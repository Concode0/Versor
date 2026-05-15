# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn

from core.foundation.layout import GradeLayout
from core.foundation.manifold import MANIFOLD_SPHERE, tag_manifold
from core.foundation.module import CliffordModule
from core.planning.action import reflection_vector_matrix
from core.runtime.algebra import CliffordAlgebra

from ._utils import (
    cache_matches,
    check_multivector_storage,
    dense_from_indices,
    grade_indices,
    layout_metric_signs,
    require_positive_int,
    resolve_layer_layout,
)


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

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        compact_output: bool = True,
    ):
        """Initialize the reflection layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of features.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        self.input_layout = resolve_layer_layout(algebra, layout=input_layout, grades=input_grades)
        self.output_layout = (
            resolve_layer_layout(algebra, layout=output_layout, grades=output_grades)
            if output_layout is not None or output_grades is not None
            else self.input_layout
        )
        self.compact_output = bool(compact_output)

        self.register_buffer("vector_indices", grade_indices(algebra, 1, name="vector grade"))
        self.num_vectors = self.vector_indices.numel()
        self.vector_layout = algebra.layout((1,))

        self.vector_weights = nn.Parameter(torch.Tensor(self.channels, self.num_vectors))
        tag_manifold(self.vector_weights, MANIFOLD_SPHERE)

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
        weights = self.vector_weights.to(device=device, dtype=dtype)
        n = dense_from_indices(weights, self.vector_indices, self.algebra.dim)

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
        is_compact = check_multivector_storage(
            x,
            self.algebra,
            channels=self.channels,
            name="ReflectionLayer input",
            layout=self.input_layout,
            allow_dense=self.input_layout is None or self.input_layout.dim == self.algebra.dim,
        )
        if is_compact:
            return self._forward_compact(x)
        if not hasattr(self.algebra, "per_channel_sandwich"):
            raise ValueError(
                "ReflectionLayer dense execution requires CliffordAlgebra; declare input_grades for compact use."
            )

        cache = (
            (self._cached_n, self._cached_n_inv)
            if self._cached_n is not None and self._cached_n_inv is not None
            else None
        )
        if not self.training and cache_matches(cache, x):
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

    def _forward_compact(self, x: torch.Tensor) -> torch.Tensor:
        """Apply compact reflection through the induced vector action."""
        if self.input_layout is None:
            raise ValueError("ReflectionLayer compact input requires input_layout or input_grades")
        output_layout = self.output_layout if self.output_layout is not None else self.input_layout

        normals = self.vector_weights.to(device=x.device, dtype=x.dtype)
        signs = layout_metric_signs(self.vector_layout, device=x.device, dtype=x.dtype)
        norm_sq = (normals * normals * signs).sum(dim=-1, keepdim=True)
        scale = norm_sq.abs().clamp_min(1e-12).sqrt()
        normals = normals / scale
        matrix = reflection_vector_matrix(normals, vector_layout=self.vector_layout, eps=self.algebra.eps_sq)
        return self.algebra.planned_linear_action(
            x,
            matrix,
            input_layout=self.input_layout,
            output_layout=output_layout,
            input_compact=True,
            compact_output=self.compact_output,
        )

    def train(self, mode: bool = True):
        """Override to invalidate cache when switching to train mode."""
        if mode:
            self._cached_n = None
            self._cached_n_inv = None
        return super().train(mode)

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity regularization on vector weights."""
        return torch.norm(self.vector_weights, p=1)
