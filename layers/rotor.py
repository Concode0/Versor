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
from layers.base import CliffordModule

class RotorLayer(CliffordModule):
    """Learnable rotor layer for sandwich-product transformation.

    Learns R = exp(-B/2) and applies the isometry x' = RxR~.
    Preserves origin, lengths, and angles.

    Attributes:
        channels (int): Number of rotors.
        bivector_weights (nn.Parameter): Learnable B coefficients.
        use_decomposition (bool): If True, use power iteration decomposition.
        decomp_k (int, optional): Number of simple components for decomposition.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        use_decomposition: bool = False,
        decomp_k: int = None
    ):
        """Initialize the rotor layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of features.
            use_decomposition (bool): If True, use bivector decomposition.
                Reference: Pence et al. (2025), arXiv:2507.11688v1
            decomp_k (int, optional): Number of simple components for decomposition.
        """
        super().__init__(algebra)
        self.channels = channels
        self.use_decomposition = use_decomposition
        self.decomp_k = decomp_k

        # Use algebra's precomputed grade masks for bivector indices
        bv_mask = algebra.grade_masks[2]
        self.register_buffer('bivector_indices', bv_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_bivectors = len(self.bivector_indices)

        self.bivector_weights = nn.Parameter(torch.Tensor(channels, self.num_bivectors))

        # Rotor cache for eval mode
        self._cached_R = None
        self._cached_R_rev = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with near-identity rotations."""
        nn.init.normal_(self.bivector_weights, std=0.01)

    def _compute_rotors(self, device, dtype):
        """Compute R and R~ from bivector weights."""
        B = torch.zeros(self.channels, self.algebra.dim, device=device, dtype=dtype)
        indices = self.bivector_indices.unsqueeze(0).expand(self.channels, -1)
        B.scatter_(1, indices, self.bivector_weights)

        if self.use_decomposition:
            R = self.algebra.exp_decomposed(
                -0.5 * B, use_decomposition=True, k=self.decomp_k
            )
        else:
            R = self.algebra.exp(-0.5 * B)

        R_rev = self.algebra.reverse(R)
        return R, R_rev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the sandwich product x' = RxR~.

        Caches rotors during eval mode for faster inference.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Rotated input.
        """
        if not self.training and self._cached_R is not None:
            R, R_rev = self._cached_R, self._cached_R_rev
        else:
            R, R_rev = self._compute_rotors(x.device, x.dtype)
            if not self.training:
                self._cached_R = R
                self._cached_R_rev = R_rev

        R_expanded = R.unsqueeze(0)
        R_rev_expanded = R_rev.unsqueeze(0)

        Rx = self.algebra.geometric_product(R_expanded, x)
        res = self.algebra.geometric_product(Rx, R_rev_expanded)

        return res

    def train(self, mode: bool = True):
        """Override to invalidate rotor cache when switching to train mode."""
        if mode:
            self._cached_R = None
            self._cached_R_rev = None
        return super().train(mode)

    def prune_bivectors(self, threshold: float = 1e-4) -> int:
        """Zero out bivector weights below the threshold.

        Args:
            threshold (float): Cutoff magnitude.

        Returns:
            int: Number of pruned parameters.
        """
        with torch.no_grad():
            mask = torch.abs(self.bivector_weights) >= threshold
            num_pruned = (~mask).sum().item()
            self.bivector_weights.data.mul_(mask.float())
        return num_pruned

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity regularization on bivector weights."""
        return torch.norm(self.bivector_weights, p=1)