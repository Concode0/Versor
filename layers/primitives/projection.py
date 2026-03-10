# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from .base import CliffordModule

class BladeSelector(CliffordModule):
    """Blade Selector. Filters insignificant components.

    Learns to weigh geometric grades, suppressing less relevant ones.

    Attributes:
        weights (nn.Parameter): Soft gates [Channels, Dim].
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """Sets up the selector.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Input features.
        """
        super().__init__(algebra)
        
        self.weights = nn.Parameter(torch.Tensor(channels, algebra.dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights to one (pass-through)."""
        nn.init.ones_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gates the grades.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Filtered input.
        """
        # Sigmoid gate
        w = torch.sigmoid(self.weights).unsqueeze(0)
        return x * w


class GeometricNeutralizer(CliffordModule):
    """Geometric Neutralization. Orthogonalizes Grade-0 against Grade-2 in real-time.

    Removes the component of the Grade-0 (scalar) signal that is parallel to the
    Grade-2 (bivector) subspace.

    Uses Exponential Moving Average (EMA) to maintain stable covariance statistics
    across batches, ensuring batch-independent behavior during inference.

    Attributes:
        algebra (CliffordAlgebra): The algebra instance.
        momentum (float): EMA momentum.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, momentum: float = 0.1):
        """Initialize the neutralizer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of multivector channels.
            momentum (float): EMA momentum for covariance tracking.
        """
        super().__init__(algebra)
        self.channels = channels
        self.momentum = momentum

        # Get indices for Grade-0 and Grade-2
        self.register_buffer('g0_idx', algebra.grade_masks[0].nonzero(as_tuple=False).squeeze(-1))
        self.register_buffer('g2_idx', algebra.grade_masks[2].nonzero(as_tuple=False).squeeze(-1))

        # Dimensions for Cl(3,1): Grade-0 is 1, Grade-2 is 6
        self.d0 = len(self.g0_idx)
        self.d2 = len(self.g2_idx)

        # EMA Buffers for each channel
        # We track:
        #   - Mean of scalar (Grade-0): [C, D0]
        #   - Mean of bivector (Grade-2): [C, D2]
        #   - Covariance(bivector, bivector): [C, D2, D2]
        #   - Covariance(bivector, scalar): [C, D2, D0]
        self.register_buffer('running_mean_scalar', torch.zeros(channels, self.d0))
        self.register_buffer('running_mean_bivec', torch.zeros(channels, self.d2))
        self.register_buffer('running_cov_bb', torch.eye(self.d2).unsqueeze(0).repeat(channels, 1, 1))
        self.register_buffer('running_cov_bs', torch.zeros(channels, self.d2, self.d0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neutralizes the multivector signal using EMA statistics.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Neutralized multivector.
        """
        # x: [B, C, D]
        scalar = x[..., self.g0_idx]  # [B, C, D0]
        bivec = x[..., self.g2_idx]   # [B, C, D2]
        B, C, _ = scalar.shape

        if self.training:
            # 1. Compute batch statistics
            batch_mean_s = scalar.mean(dim=0)  # [C, D0]
            batch_mean_b = bivec.mean(dim=0)   # [C, D2]

            # Center the batch
            s_centered = scalar - batch_mean_s.unsqueeze(0)
            b_centered = bivec - batch_mean_b.unsqueeze(0)

            # batch_cov_bb: [C, D2, D2]
            batch_cov_bb = torch.einsum('bci, bcj -> cij', b_centered, b_centered) / (B - 1 + 1e-8)
            # batch_cov_bs: [C, D2, D0]
            batch_cov_bs = torch.einsum('bci, bcj -> cij', b_centered, s_centered) / (B - 1 + 1e-8)

            # 2. Update EMA buffers
            self.running_mean_scalar = (1 - self.momentum) * self.running_mean_scalar + self.momentum * batch_mean_s
            self.running_mean_bivec = (1 - self.momentum) * self.running_mean_bivec + self.momentum * batch_mean_b
            self.running_cov_bb = (1 - self.momentum) * self.running_cov_bb + self.momentum * batch_cov_bb
            self.running_cov_bs = (1 - self.momentum) * self.running_cov_bs + self.momentum * batch_cov_bs

            # Use batch stats during training
            cur_mean_s = batch_mean_s
            cur_mean_b = batch_mean_b
            cur_cov_bb = batch_cov_bb
            cur_cov_bs = batch_cov_bs
        else:
            # Use EMA stats during inference
            cur_mean_s = self.running_mean_scalar
            cur_mean_b = self.running_mean_bivec
            cur_cov_bb = self.running_cov_bb
            cur_cov_bs = self.running_cov_bs

        # 3. Perform Projection
        # Solve: cur_cov_bb * W = cur_cov_bs  => W = inv(cur_cov_bb) * cur_cov_bs
        # Use pseudo-inverse for stability
        if cur_cov_bb.device.type == 'mps':
            inv_bb = torch.linalg.pinv(cur_cov_bb.cpu()).to(cur_cov_bb.device)
        else:
            inv_bb = torch.linalg.pinv(cur_cov_bb)

        weights = torch.matmul(inv_bb, cur_cov_bs)

        # Center based on current means
        b_centered = bivec - cur_mean_b.unsqueeze(0)

        # Projection: [B, C, D0]
        projection = torch.einsum('bci, cij -> bcj', b_centered, weights)

        # Neutralized scalar
        scalar_n = scalar - projection

        # 4. Construct the output multivector
        out = x.clone()
        out[..., self.g0_idx] = scalar_n
        return out
