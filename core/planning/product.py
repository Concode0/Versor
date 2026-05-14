# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Static grade-path plans for sparse high-dimensional Clifford products.

This module describes the lower-level AoT shape needed by high-dimensional
sparse execution: input grades are declared or inferred at construction time,
all basis interactions are expanded once, and forward execution is only gather,
multiply, and indexed reduction over the required output grade lanes.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn

from core.foundation.basis import (
    GradeProductOp,
    basis_index_tuple_for_grades,
    basis_indices_tensor,
    expand_output_grades,
    normalize_grades,
    operation_coefficient,
)
from core.foundation.layout import AlgebraSpec
from core.planning.layouts import ProductRequest
from core.planning.tree import GradePlanTree, build_grade_plan_tree


class GradeProductPlan:
    """AoT basis interaction plan for one grade-restricted bilinear product."""

    def __init__(
        self,
        *,
        p: int,
        q: int,
        r: int,
        op: GradeProductOp,
        left_grades: tuple[int, ...],
        right_grades: tuple[int, ...],
        output_grades: tuple[int, ...],
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        output_indices: torch.Tensor,
        output_positions: torch.Tensor,
        left_compact_positions: torch.Tensor,
        right_compact_positions: torch.Tensor,
        coefficients: torch.Tensor,
        active_output_indices: torch.Tensor,
        tree: GradePlanTree,
    ):
        self.spec = AlgebraSpec(p, q, r)
        self.op = op
        self.left_layout = self.spec.layout(left_grades)
        self.right_layout = self.spec.layout(right_grades)
        self.output_layout = self.spec.layout(output_grades)
        self.left_indices = left_indices
        self.right_indices = right_indices
        self.output_indices = output_indices
        self.output_positions = output_positions
        self.left_compact_positions = left_compact_positions
        self.right_compact_positions = right_compact_positions
        self.coefficients = coefficients
        self.active_output_indices = active_output_indices
        self.tree = tree

    @property
    def p(self) -> int:
        return self.spec.p

    @property
    def q(self) -> int:
        return self.spec.q

    @property
    def r(self) -> int:
        return self.spec.r

    @property
    def left_grades(self) -> tuple[int, ...]:
        return self.left_layout.grades

    @property
    def right_grades(self) -> tuple[int, ...]:
        return self.right_layout.grades

    @property
    def output_grades(self) -> tuple[int, ...]:
        return self.output_layout.grades

    @property
    def n(self) -> int:
        return self.p + self.q + self.r

    @property
    def dim(self) -> int:
        return 1 << self.n

    @property
    def pair_count(self) -> int:
        return int(self.left_indices.numel())

    @property
    def output_dim(self) -> int:
        return int(self.active_output_indices.numel())

    @property
    def is_empty(self) -> bool:
        return self.pair_count == 0

    @property
    def density(self) -> float:
        if self.tree.estimated_pairs == 0:
            return 0.0
        return self.pair_count / self.tree.estimated_pairs


def build_grade_product_plan(
    p: int,
    q: int = 0,
    r: int = 0,
    *,
    left_grades: Iterable[int],
    right_grades: Iterable[int],
    output_grades: Optional[Iterable[int]] = None,
    op: GradeProductOp = "gp",
    device=None,
    dtype: torch.dtype = torch.float32,
) -> GradeProductPlan:
    """Build an exact static basis-pair plan for a grade-restricted operation."""
    spec = AlgebraSpec(int(p), int(q), int(r))
    tree = build_grade_plan_tree(
        spec,
        left_grades=left_grades,
        right_grades=right_grades,
        output_grades=output_grades,
        op=op,
    )
    return build_grade_product_plan_from_tree(tree, device=device, dtype=dtype)


def build_grade_product_plan_from_request(
    request: ProductRequest,
    *,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> GradeProductPlan:
    """Build a plan from a normalized product request."""
    tree = build_grade_plan_tree(
        request.spec,
        left_grades=request.left_grades,
        right_grades=request.right_grades,
        output_grades=request.output_grades,
        op=request.op,
    )
    return build_grade_product_plan_from_tree(
        tree,
        device=request.device if device is None else device,
        dtype=request.dtype if dtype is None else dtype,
    )


def build_grade_product_plan_from_tree(
    tree: GradePlanTree,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> GradeProductPlan:
    """Lower a planner tree into flat Torch gather/reduce buffers."""
    spec = tree.spec
    p, q, r = spec.p, spec.q, spec.r
    n = spec.n
    left_grade_tuple = tree.left_grades
    right_grade_tuple = tree.right_grades
    output_grade_tuple = tree.output_grades

    left_basis_by_grade = {grade: basis_index_tuple_for_grades(n, (grade,)) for grade in left_grade_tuple}
    right_basis_by_grade = {grade: basis_index_tuple_for_grades(n, (grade,)) for grade in right_grade_tuple}
    left_position_by_index = {
        index: position for position, index in enumerate(spec.layout(left_grade_tuple).basis_indices)
    }
    right_position_by_index = {
        index: position for position, index in enumerate(spec.layout(right_grade_tuple).basis_indices)
    }
    active_outputs = basis_index_tuple_for_grades(n, output_grade_tuple)
    output_position_by_index = {index: position for position, index in enumerate(active_outputs)}

    plan_left: list[int] = []
    plan_right: list[int] = []
    plan_output: list[int] = []
    plan_positions: list[int] = []
    plan_left_compact_positions: list[int] = []
    plan_right_compact_positions: list[int] = []
    plan_coefficients: list[float] = []

    for path in tree.paths:
        for left_index in left_basis_by_grade[path.left_grade]:
            for right_index in right_basis_by_grade[path.right_grade]:
                output_index = left_index ^ right_index
                output_position = output_position_by_index.get(output_index)
                if output_position is None:
                    continue
                coefficient = operation_coefficient(left_index, right_index, p, q, r, tree.op)
                if coefficient == 0.0:
                    continue
                plan_left.append(left_index)
                plan_right.append(right_index)
                plan_output.append(output_index)
                plan_positions.append(output_position)
                plan_left_compact_positions.append(left_position_by_index[left_index])
                plan_right_compact_positions.append(right_position_by_index[right_index])
                plan_coefficients.append(coefficient)

    return GradeProductPlan(
        p=p,
        q=q,
        r=r,
        op=tree.op,
        left_grades=left_grade_tuple,
        right_grades=right_grade_tuple,
        output_grades=output_grade_tuple,
        left_indices=basis_indices_tensor(plan_left, n=n, role="left product basis indices", device=device),
        right_indices=basis_indices_tensor(plan_right, n=n, role="right product basis indices", device=device),
        output_indices=basis_indices_tensor(plan_output, n=n, role="output product basis indices", device=device),
        output_positions=torch.tensor(plan_positions, dtype=torch.long, device=device),
        left_compact_positions=torch.tensor(plan_left_compact_positions, dtype=torch.long, device=device),
        right_compact_positions=torch.tensor(plan_right_compact_positions, dtype=torch.long, device=device),
        coefficients=torch.tensor(plan_coefficients, dtype=dtype, device=device),
        active_output_indices=basis_indices_tensor(active_outputs, n=n, role="active output basis indices", device=device),
        tree=tree,
    )


class GradeProductExecutor(nn.Module):
    """Compile-friendly grade-restricted product using a static interaction plan.

    ``forward`` returns compact output lanes ordered by ``active_output_indices``.
    ``forward_dense`` is an explicit materialization helper for parity checks and
    dense callers.
    """

    def __init__(self, plan: GradeProductPlan):
        super().__init__()
        self.p = plan.p
        self.q = plan.q
        self.r = plan.r
        self.n = plan.n
        self.dim = plan.dim
        self.op = plan.op
        self.left_grades = plan.left_grades
        self.right_grades = plan.right_grades
        self.output_grades = plan.output_grades
        self.left_layout = plan.left_layout
        self.right_layout = plan.right_layout
        self.output_layout = plan.output_layout
        self._output_dim = plan.output_dim
        self._pair_count = plan.pair_count
        self.register_buffer("left_indices", plan.left_indices, persistent=False)
        self.register_buffer("right_indices", plan.right_indices, persistent=False)
        self.register_buffer("output_indices", plan.output_indices, persistent=False)
        self.register_buffer("output_positions", plan.output_positions, persistent=False)
        self.register_buffer("coefficients", plan.coefficients, persistent=False)
        self.register_buffer("active_output_indices", plan.active_output_indices, persistent=False)
        self.register_buffer("left_compact_positions", plan.left_compact_positions, persistent=False)
        self.register_buffer("right_compact_positions", plan.right_compact_positions, persistent=False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def pair_count(self) -> int:
        return self._pair_count

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return compact grade-lane output for full dense input tensors."""
        if left.shape[-1] != self.dim:
            raise ValueError(f"left last dimension must be {self.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.dim:
            raise ValueError(f"right last dimension must be {self.dim}, got {right.shape[-1]}")

        left_b, right_b = torch.broadcast_tensors(left, right)
        left_terms = torch.index_select(left_b, -1, self.left_indices)
        right_terms = torch.index_select(right_b, -1, self.right_indices)
        coefficients = self.coefficients.to(dtype=torch.promote_types(left_terms.dtype, right_terms.dtype))
        terms = left_terms * right_terms * coefficients

        output = terms.new_zeros(*terms.shape[:-1], self.output_dim)
        return output.index_add(-1, self.output_positions, terms)

    def forward_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return compact output for inputs already stored in this plan's compact layouts."""
        if left.shape[-1] != self.left_layout.dim:
            raise ValueError(f"left compact dimension must be {self.left_layout.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.right_layout.dim:
            raise ValueError(f"right compact dimension must be {self.right_layout.dim}, got {right.shape[-1]}")

        left_b, right_b = torch.broadcast_tensors(left, right)
        left_terms = torch.index_select(left_b, -1, self.left_compact_positions)
        right_terms = torch.index_select(right_b, -1, self.right_compact_positions)
        coefficients = self.coefficients.to(dtype=torch.promote_types(left_terms.dtype, right_terms.dtype))
        terms = left_terms * right_terms * coefficients

        output = terms.new_zeros(*terms.shape[:-1], self.output_dim)
        return output.index_add(-1, self.output_positions, terms)

    def forward_pairwise_compact(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Pairwise compact product for sequence-style bilinear scoring.

        ``left`` is ``[..., left_items, left_layout.dim]`` and ``right`` is
        ``[..., right_items, right_layout.dim]``. The result is
        ``[..., left_items, right_items, output_layout.dim]``.
        """
        if left.shape[-1] != self.left_layout.dim:
            raise ValueError(f"left compact dimension must be {self.left_layout.dim}, got {left.shape[-1]}")
        if right.shape[-1] != self.right_layout.dim:
            raise ValueError(f"right compact dimension must be {self.right_layout.dim}, got {right.shape[-1]}")

        prefix = torch.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        left = left.expand(*prefix, *left.shape[-2:])
        right = right.expand(*prefix, *right.shape[-2:])

        left_terms = torch.index_select(left, -1, self.left_compact_positions)
        right_terms = torch.index_select(right, -1, self.right_compact_positions)
        coefficients = self.coefficients.to(dtype=torch.promote_types(left_terms.dtype, right_terms.dtype))
        terms = left_terms.unsqueeze(-2) * right_terms.unsqueeze(-3) * coefficients

        output = terms.new_zeros(*terms.shape[:-1], self.output_dim)
        return output.index_add(-1, self.output_positions, terms)

    def forward_dense(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Return a full ``[..., 2**n]`` dense tensor for dense-kernel parity checks."""
        compact = self.forward(left, right)
        output = compact.new_zeros(*compact.shape[:-1], self.dim)
        return output.index_copy(-1, self.active_output_indices, compact)
