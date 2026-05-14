# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Planner-only grade tree metadata.

The tree here is not a runtime backend. It groups declared grade routes before
they are lowered into flat Torch executor buffers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

from core.foundation.basis import GradeProductOp, expand_output_grades, normalize_grades, product_output_grades
from core.foundation.layout import AlgebraSpec


@dataclass(frozen=True)
class GradePathNode:
    """One homogeneous left-grade/right-grade route in a product plan."""

    path_index: int
    left_grade: int
    right_grade: int
    output_grades: tuple[int, ...]
    left_dim: int
    right_dim: int

    @property
    def estimated_pairs(self) -> int:
        """Upper-bound number of basis pairs before metric zero pruning."""
        return self.left_dim * self.right_dim


@dataclass(frozen=True)
class GradePlanTree:
    """Planner-side grouping for a grade-restricted product."""

    spec: AlgebraSpec
    op: GradeProductOp
    left_grades: tuple[int, ...]
    right_grades: tuple[int, ...]
    output_grades: tuple[int, ...]
    paths: tuple[GradePathNode, ...]
    chunk_pair_limit: Optional[int] = None

    @property
    def path_count(self) -> int:
        """Number of active homogeneous product routes."""
        return len(self.paths)

    @property
    def estimated_pairs(self) -> int:
        """Upper-bound number of basis pairs across all paths."""
        return sum(path.estimated_pairs for path in self.paths)

    @property
    def estimated_chunks(self) -> int:
        """Number of planner chunks implied by ``chunk_pair_limit``."""
        if self.chunk_pair_limit is None or self.chunk_pair_limit <= 0:
            return 1 if self.paths else 0
        return sum(math.ceil(path.estimated_pairs / self.chunk_pair_limit) for path in self.paths)

    def path_for_grades(self, left_grade: int, right_grade: int) -> Optional[GradePathNode]:
        """Return the active path for a homogeneous grade pair."""
        for path in self.paths:
            if path.left_grade == left_grade and path.right_grade == right_grade:
                return path
        return None


def build_grade_plan_tree(
    spec: AlgebraSpec,
    *,
    left_grades: Iterable[int],
    right_grades: Iterable[int],
    output_grades: Optional[Iterable[int]] = None,
    op: GradeProductOp = "gp",
    chunk_pair_limit: Optional[int] = None,
) -> GradePlanTree:
    """Build planner metadata for grade route grouping."""
    left = normalize_grades(left_grades, spec.n, name="left_grades")
    right = normalize_grades(right_grades, spec.n, name="right_grades")
    output = (
        expand_output_grades(left, right, spec.n, op=op)
        if output_grades is None
        else normalize_grades(output_grades, spec.n, name="output_grades")
    )
    output_set = set(output)

    paths = []
    for left_grade in left:
        left_dim = _grade_dim(spec.n, left_grade)
        for right_grade in right:
            route_outputs = product_output_grades(left_grade, right_grade, spec.n, op=op)
            route_outputs = tuple(grade for grade in route_outputs if grade in output_set)
            if not route_outputs:
                continue
            paths.append(
                GradePathNode(
                    path_index=len(paths),
                    left_grade=left_grade,
                    right_grade=right_grade,
                    output_grades=route_outputs,
                    left_dim=left_dim,
                    right_dim=_grade_dim(spec.n, right_grade),
                )
            )

    return GradePlanTree(
        spec=spec,
        op=op,
        left_grades=left,
        right_grades=right,
        output_grades=output,
        paths=tuple(paths),
        chunk_pair_limit=chunk_pair_limit,
    )


def _grade_dim(n: int, grade: int) -> int:
    if grade < 0 or grade > n:
        return 0
    grade = min(grade, n - grade)
    numerator = 1
    denominator = 1
    for i in range(1, grade + 1):
        numerator *= n - grade + i
        denominator *= i
    return numerator // denominator
