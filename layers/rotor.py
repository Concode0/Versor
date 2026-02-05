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
    """A Learnable Rotor Layer.

    Learns a set of rotors R = exp(-B/2) to perform geometric rotations on the input.
    Rotors preserve the origin, lengths, and angles (isometries).

    Attributes:
        channels (int): Number of independent rotors to learn.
        bivector_indices (torch.Tensor): Indices of bivector basis elements in the algebra.
        num_bivectors (int): Count of bivector components.
        bivector_weights (nn.Parameter): Learnable coefficients for B [Channels, Num_Bivectors].
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """Initializes the Rotor Layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of channels (features) to rotate independently.
        """
        super().__init__(algebra)
        self.channels = channels
        
        # We learn the bivector B directly.
        self.bivector_indices = self._get_bivector_indices()
        self.num_bivectors = len(self.bivector_indices)
        
        # Weights: [Channels, Num_Bivectors]
        self.bivector_weights = nn.Parameter(torch.Tensor(channels, self.num_bivectors))
        
        self.reset_parameters()
        
    def _get_bivector_indices(self) -> torch.Tensor:
        """Identifies indices corresponding to Grade-2 elements (bivectors)."""
        indices = []
        for i in range(self.algebra.dim):
            # Count set bits (grade)
            cnt = 0
            temp = i
            while temp > 0:
                if temp & 1: cnt += 1
                temp >>= 1
            if cnt == 2:
                indices.append(i)
        return torch.tensor(indices, device=self.algebra.device, dtype=torch.long)

    def reset_parameters(self):
        """Initializes bivector weights to small random values (near identity rotation)."""
        nn.init.normal_(self.bivector_weights, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the rotor transformation x -> R x R~.

        Args:
            x (torch.Tensor): Input multivectors [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Rotated multivectors [Batch, Channels, Dim].
        """
        # 1. Construct Bivector B from weights
        B = torch.zeros(self.channels, self.algebra.dim, device=x.device, dtype=x.dtype)
        
        # Scatter weights into correct bivector indices
        indices = self.bivector_indices.unsqueeze(0).expand(self.channels, -1)
        B.scatter_(1, indices, self.bivector_weights)
        
        # 2. Compute Rotor R = exp(-B/2)
        # Using Taylor series or optimized exp map
        R = self.algebra.exp(-0.5 * B)
        
        # 3. Compute Reverse R_rev
        R_rev = self.algebra.reverse(R)
        
        # 4. Apply Sandwich Product: R * x * R_rev
        # R is [Channels, Dim], x is [Batch, Channels, Dim]
        
        R_expanded = R.unsqueeze(0) # [1, C, D]
        R_rev_expanded = R_rev.unsqueeze(0) # [1, C, D]
        
        # Rx
        Rx = self.algebra.geometric_product(R_expanded, x)
        
        # (Rx)R_rev
        res = self.algebra.geometric_product(Rx, R_rev_expanded)
        
        return res

    def prune_bivectors(self, threshold: float = 1e-4) -> int:
        """Prunes bivector weights with small magnitudes (Geometric Sparsity).

        Args:
            threshold (float): Magnitude threshold for pruning.

        Returns:
            int: Number of pruned parameters.
        """
        with torch.no_grad():
            mask = torch.abs(self.bivector_weights) >= threshold
            num_pruned = (~mask).sum().item()
            self.bivector_weights.data.mul_(mask.float())
        return num_pruned

    def sparsity_loss(self) -> torch.Tensor:
        """Computes the geometric sparsity loss (L1 norm of bivectors).

        Encourages the learning of simpler rotations (fewer active planes).

        Returns:
            torch.Tensor: Scalar loss.
        """
        return torch.norm(self.bivector_weights, p=1)
