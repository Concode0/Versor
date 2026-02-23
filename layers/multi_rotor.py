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

class MultiRotorLayer(CliffordModule):
    """Multi-rotor layer with weighted superposition: x' = sum_k w_k R_k x R~_k.

    Replaces rigid single-rotor rotations with a flexible superposition.

    Attributes:
        channels (int): Input features.
        num_rotors (int): Number of overlapping rotors.
        use_decomposition (bool): If True, use power iteration decomposition.
        decomp_k (int, optional): Number of simple components for decomposition.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_rotors: int = 8,
        use_decomposition: bool = False,
        decomp_k: int = None
    ):
        """Initialize the multi-rotor layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Input features.
            num_rotors (int): Parallel heads.
            use_decomposition (bool): If True, use bivector decomposition.
                Reference: Pence et al. (2025), arXiv:2507.11688v1
            decomp_k (int, optional): Number of simple components for decomposition.
        """
        super().__init__(algebra)
        self.channels = channels
        self.num_rotors = num_rotors
        self.use_decomposition = use_decomposition
        self.decomp_k = decomp_k

        # Use algebra's precomputed grade masks for bivector indices
        bv_mask = algebra.grade_masks[2]
        self.register_buffer('bivector_indices', bv_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_bivectors = len(self.bivector_indices)

        # Overlapping rotors
        self.rotor_bivectors = nn.Parameter(torch.Tensor(num_rotors, self.num_bivectors))

        # Mixing weights
        self.weights = nn.Parameter(torch.Tensor(channels, num_rotors))

        # Rotor cache for eval mode
        self._cached_R = None
        self._cached_R_rev = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with small rotations and uniform weights."""
        nn.init.normal_(self.rotor_bivectors, std=0.01)
        nn.init.xavier_uniform_(self.weights)

    def _compute_rotors(self, device, dtype):
        """Compute R and R~ from bivector weights."""
        B = torch.zeros(self.num_rotors, self.algebra.dim, device=device, dtype=dtype)
        indices = self.bivector_indices.unsqueeze(0).expand(self.num_rotors, -1)
        B.scatter_(1, indices, self.rotor_bivectors)

        if self.use_decomposition:
            R = self.algebra.exp_decomposed(
                -0.5 * B, use_decomposition=True, k=self.decomp_k
            )
        else:
            R = self.algebra.exp(-0.5 * B)  # [K, D]
        R_rev = self.algebra.reverse(R)
        return R, R_rev

    def forward(self, x: torch.Tensor, return_invariants: bool = False) -> torch.Tensor:
        """Apply weighted multi-rotor transformation.

        Caches rotors during eval mode for faster inference.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].
            return_invariants (bool): If True, returns grade norms.

        Returns:
            torch.Tensor: Transformed output.
        """
        if not self.training and self._cached_R is not None:
            R, R_rev = self._cached_R, self._cached_R_rev
        else:
            R, R_rev = self._compute_rotors(x.device, x.dtype)
            if not self.training:
                self._cached_R = R
                self._cached_R_rev = R_rev

        # Sandwich Product
        x_expanded = x.unsqueeze(2)
        R_expanded = R.view(1, 1, self.num_rotors, -1)
        R_rev_expanded = R_rev.view(1, 1, self.num_rotors, -1)

        Rx = self.algebra.geometric_product(R_expanded, x_expanded)
        rotated_x = self.algebra.geometric_product(Rx, R_rev_expanded)

        # Superposition
        out = torch.einsum('ck,bckd->bcd', self.weights, rotated_x)

        if return_invariants:
            return self.algebra.get_grade_norms(out)

        return out

    def train(self, mode: bool = True):
        """Override to invalidate rotor cache when switching to train mode."""
        if mode:
            self._cached_R = None
            self._cached_R_rev = None
        return super().train(mode)

    def sparsity_loss(self) -> torch.Tensor:
        """Computes the L1 sparsity loss for rotor bivectors and weights."""
        return torch.norm(self.rotor_bivectors, p=1) + torch.norm(self.weights, p=1)