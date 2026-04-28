# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Multi-versor superposition layers with universal grade parameterization.

Implements versor-based transformations using weighted sums of sandwich products.
"""

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from core.validation import check_channels, check_multivector

from .base import CliffordModule


class MultiRotorLayer(CliffordModule):
    """Multi-versor layer with weighted superposition: x' = sum_k w_k hat(V_k) x V_k^{-1}.

    For grade=2 (default): each V_k = exp(-B_k/2) is a rotor, reducing to
    x' = sum_k w_k R_k x R~_k.
    For grade=k: each V_k is a grade-k versor applied via the general versor product.

    The exp strategy is controlled by ``algebra.exp_policy``.

    Attributes:
        channels (int): Input features.
        num_rotors (int): Number of overlapping versors.
        grade (int): Grade of the learnable parameters. Default 2 (rotors).
        rotor_grade_weights (nn.Parameter): Grade-k coefficients [num_rotors, num_grade_elements].
        weights (nn.Parameter): Mixing weights [channels, num_rotors].
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_rotors: int = 8,
        grade: int = 2,
    ):
        """Initialize Multi-Versor Layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Input features.
            num_rotors (int): Number of parallel versor heads.
            grade (int): Grade of the learnable parameter.
                grade=2 (default): bivectors → rotors via exp(-B/2), Spin group.
                grade=k: general grade-k versor product.
        """
        super().__init__(algebra)
        self.channels = channels
        self.num_rotors = num_rotors
        self.grade = grade

        grade_mask = algebra.grade_masks[grade]
        self.register_buffer("grade_indices", grade_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_grade_elements = len(self.grade_indices)

        self.rotor_grade_weights = nn.Parameter(torch.Tensor(num_rotors, self.num_grade_elements))
        if grade == 2:
            self.rotor_grade_weights._manifold = "spin"

        # Mixing weights (Euclidean — intentionally untagged)
        self.weights = nn.Parameter(torch.Tensor(channels, num_rotors))

        # Versor cache for eval mode
        self._cached_V_left = None
        self._cached_V_right = None

        self.reset_parameters()

    # --- Backward-compat aliases (grade == 2 usage) ---

    @property
    def bivector_indices(self):
        return self.grade_indices

    @property
    def num_bivectors(self):
        return self.num_grade_elements

    @property
    def rotor_bivectors(self):
        return self.rotor_grade_weights

    # ---------------------------------------------------

    def reset_parameters(self):
        """Initialize with small transforms and uniform mixing weights."""
        nn.init.normal_(self.rotor_grade_weights, std=0.01)
        nn.init.xavier_uniform_(self.weights)

    def _compute_versors(self, device, dtype):
        """Compute left and right factors for all K versors.

        For grade=2: left = R_k = exp(-B_k/2), right = R~_k.
        For grade=k: left = hat(V_k), right = V_k^{-1}.

        Returns:
            Tuple[Tensor, Tensor]: (V_left [K, dim], V_right [K, dim])
        """
        V = torch.zeros(self.num_rotors, self.algebra.dim, device=device, dtype=dtype)
        indices = self.grade_indices.unsqueeze(0).expand(self.num_rotors, -1)
        V.scatter_(1, indices, self.rotor_grade_weights)

        if self.grade == 2:
            R = self.algebra.exp(-0.5 * V)  # [K, D]
            return R, self.algebra.reverse(R)
        else:
            norm = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            V = V / norm
            return self.algebra.grade_involution(V), self.algebra.blade_inverse(V)

    def forward(self, x: torch.Tensor, return_invariants: bool = False) -> torch.Tensor:
        """Apply weighted multi-versor superposition.

        Caches versors during eval mode for faster inference.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].
            return_invariants (bool): If True, returns per-grade norms instead of output.

        Returns:
            torch.Tensor: Transformed output [Batch, Channels, Dim].
        """
        check_multivector(x, self.algebra, "MultiRotorLayer input")
        check_channels(x, self.channels, "MultiRotorLayer input")

        if not self.training and self._cached_V_left is not None:
            V_left, V_right = self._cached_V_left, self._cached_V_right
        else:
            V_left, V_right = self._compute_versors(x.device, x.dtype)
            if not self.training:
                self._cached_V_left = V_left
                self._cached_V_right = V_right

        # Action-matrix sandwich: build K matrices once, apply via einsum
        versored_x = self.algebra.multi_rotor_sandwich(
            V_left,
            x,
            V_right,
        )  # [B, C, K, D]

        # Weighted superposition
        out = torch.einsum("ck,bcke->bce", self.weights, versored_x)

        if return_invariants:
            return self.algebra.get_grade_norms(out)

        return out

    def train(self, mode: bool = True):
        """Invalidate versor cache when switching to train mode."""
        if mode:
            self._cached_V_left = None
            self._cached_V_right = None
        return super().train(mode)

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss for versor weights and mixing weights."""
        return torch.norm(self.rotor_grade_weights, p=1) + torch.norm(self.weights, p=1)
