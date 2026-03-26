# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from core.validation import check_multivector, check_channels
from .base import CliffordModule


class ReflectionLayer(CliffordModule):
    """Learnable reflection layer via unit vectors.

    Each channel learns a unit vector n_c and applies the reflection
    x'_c = -n_c x_c n_c^{-1}. This is the fundamental odd versor
    transformation - rotors (even versors) are compositions of two
    reflections.

    The learned vectors are projected to unit norm before each reflection.
    For Euclidean signature, the projection is simple L2 normalization.
    For mixed signature, the projection normalizes by |<n ~n>_0|.

    Attributes:
        channels (int): Number of reflection vectors.
        vector_weights (nn.Parameter): Learnable grade-1 coefficients [C, n].
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """Initialize the reflection layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of features.
        """
        super().__init__(algebra)
        self.channels = channels

        # Grade-1 indices: 2^0, 2^1, ..., 2^(n-1)
        g1_mask = algebra.grade_masks[1]
        self.register_buffer('vector_indices', g1_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_vectors = algebra.n

        self.vector_weights = nn.Parameter(torch.Tensor(channels, self.num_vectors))

        # Cache for eval mode
        self._cached_n = None
        self._cached_n_inv = None

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize with random unit-ish vectors."""
        nn.init.normal_(self.vector_weights, std=1.0)
        # Normalize to unit vectors
        with torch.no_grad():
            norms = self.vector_weights.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.vector_weights.div_(norms)

    def _build_vectors(self, device, dtype):
        """Build full multivectors from vector weights and normalize.

        Returns:
            Tuple of (n, n_inv) each [C, dim].
        """
        C = self.channels
        n = torch.zeros(C, self.algebra.dim, device=device, dtype=dtype)
        indices = self.vector_indices.unsqueeze(0).expand(C, -1)
        n.scatter_(1, indices, self.vector_weights.to(dtype=dtype))

        # Normalize: n_hat = n / sqrt(|<n ~n>_0|)
        n_sq = self.algebra.norm_sq(n)  # [C, 1]
        scale = n_sq.abs().clamp(min=1e-12).sqrt()
        n = n / scale

        n_inv = self.algebra.blade_inverse(n)
        return n, n_inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-channel reflections: x'_c = -n_c x_c n_c^{-1}.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Reflected input [Batch, Channels, Dim].
        """
        check_multivector(x, self.algebra, "ReflectionLayer input")
        check_channels(x, self.channels, "ReflectionLayer input")

        if not self.training and self._cached_n is not None:
            n, n_inv = self._cached_n, self._cached_n_inv
        else:
            n, n_inv = self._build_vectors(x.device, x.dtype)
            if not self.training:
                self._cached_n = n
                self._cached_n_inv = n_inv

        # grade_involution(n) = -n for grade-1 vectors
        n_hat = -n  # [C, dim]

        # Per-channel reflection via two GPs: (-n) * x * n^{-1}
        # Use per_channel_sandwich with n_hat as "R" and n_inv as "R_rev"
        return self.algebra.per_channel_sandwich(n_hat, x, n_inv)

    def train(self, mode: bool = True):
        """Override to invalidate cache when switching to train mode."""
        if mode:
            self._cached_n = None
            self._cached_n_inv = None
        return super().train(mode)

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity regularization on vector weights."""
        return torch.norm(self.vector_weights, p=1)
