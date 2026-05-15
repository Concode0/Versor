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

from core.foundation.layout import GradeLayout
from core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from core.foundation.module import CliffordModule
from core.planning.action import bivector_vector_generator, reflection_vector_matrix
from core.runtime.accessors import materialize_dense
from core.runtime.algebra import CliffordAlgebra

from ._utils import (
    cache_matches,
    check_multivector_storage,
    compact_grade_norms,
    dense_from_indices,
    grade_indices,
    layout_metric_signs,
    require_positive_int,
    resolve_layer_layout,
)


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
        *,
        input_grades=None,
        output_grades=None,
        input_layout: GradeLayout = None,
        output_layout: GradeLayout = None,
        compact_output: bool = True,
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
        self.channels = require_positive_int(channels, "channels")
        self.num_rotors = require_positive_int(num_rotors, "num_rotors")
        self.grade = int(grade)
        self.input_layout = resolve_layer_layout(algebra, layout=input_layout, grades=input_grades)
        self.output_layout = (
            resolve_layer_layout(algebra, layout=output_layout, grades=output_grades)
            if output_layout is not None or output_grades is not None
            else self.input_layout
        )
        self.compact_output = bool(compact_output)

        self.register_buffer("grade_indices", grade_indices(algebra, self.grade))
        self.num_grade_elements = self.grade_indices.numel()
        self.parameter_layout = algebra.layout((self.grade,))

        self.rotor_grade_weights = nn.Parameter(torch.Tensor(self.num_rotors, self.num_grade_elements))
        if self.grade == 2:
            tag_manifold(self.rotor_grade_weights, MANIFOLD_SPIN)

        # Mixing weights (Euclidean — intentionally untagged)
        self.weights = nn.Parameter(torch.Tensor(self.channels, self.num_rotors))

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
        weights = self.rotor_grade_weights.to(device=device, dtype=dtype)
        V = dense_from_indices(weights, self.grade_indices, self.algebra.dim)

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
        is_compact = check_multivector_storage(
            x,
            self.algebra,
            channels=self.channels,
            name="MultiRotorLayer input",
            layout=self.input_layout,
            allow_dense=self.input_layout is None or self.input_layout.dim == self.algebra.dim,
        )
        if is_compact:
            out = self._forward_compact(x)
            if return_invariants:
                output_layout = self.output_layout if self.output_layout is not None else self.input_layout
                return compact_grade_norms(self.algebra, out, output_layout)
            return out
        if not hasattr(self.algebra, "multi_rotor_sandwich"):
            raise ValueError(
                "MultiRotorLayer dense execution requires CliffordAlgebra; declare input_grades for compact use."
            )

        cache = (
            (self._cached_V_left, self._cached_V_right)
            if self._cached_V_left is not None and self._cached_V_right is not None
            else None
        )
        if not self.training and cache_matches(cache, x):
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
        weights = self.weights.to(device=x.device, dtype=x.dtype)
        out = torch.einsum("ck,...cke->...ce", weights, versored_x)

        if return_invariants:
            return self.algebra.get_grade_norms(out)

        return out

    def _forward_compact(self, x: torch.Tensor) -> torch.Tensor:
        """Apply compact weighted superposition of induced versor actions."""
        if self.input_layout is None:
            raise ValueError("MultiRotorLayer compact input requires input_layout or input_grades")
        output_layout = self.output_layout if self.output_layout is not None else self.input_layout

        rotor_values = self.rotor_grade_weights.to(device=x.device, dtype=x.dtype)
        if self.grade == 2:
            generator = bivector_vector_generator(rotor_values, bivector_layout=self.parameter_layout)
            matrices = torch.matrix_exp(generator)
        elif self.grade == 1:
            signs = layout_metric_signs(self.parameter_layout, device=x.device, dtype=x.dtype)
            norm_sq = (rotor_values * rotor_values * signs).sum(dim=-1, keepdim=True)
            scale = norm_sq.abs().clamp_min(1e-12).sqrt()
            normals = rotor_values / scale
            matrices = reflection_vector_matrix(normals, vector_layout=self.parameter_layout, eps=self.algebra.eps_sq)
        else:
            raise ValueError("MultiRotorLayer compact execution currently supports grade=1 and grade=2 versors")

        outputs = []
        for rotor_index in range(self.num_rotors):
            matrix = matrices[rotor_index].unsqueeze(0).expand(self.channels, -1, -1)
            outputs.append(
                self.algebra.planned_linear_action(
                    x,
                    matrix,
                    input_layout=self.input_layout,
                    output_layout=output_layout,
                    input_compact=True,
                    compact_output=True,
                )
            )
        stacked = torch.stack(outputs, dim=-2)
        weights = self.weights.to(device=x.device, dtype=x.dtype)
        out = torch.einsum("ck,...ckd->...cd", weights, stacked)
        if self.compact_output:
            return out
        return materialize_dense(self.algebra, out, layout=output_layout)

    def train(self, mode: bool = True):
        """Invalidate versor cache when switching to train mode."""
        if mode:
            self._cached_V_left = None
            self._cached_V_right = None
        return super().train(mode)

    def sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity loss for versor weights and mixing weights."""
        return torch.norm(self.rotor_grade_weights, p=1) + torch.norm(self.weights, p=1)
