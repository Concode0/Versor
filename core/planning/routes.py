# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Static module-route descriptors for compile-time optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch.nn as nn

from core.foundation.layout import GradeLayout
from core.foundation.module import AlgebraLike

_MISSING = object()


@dataclass(frozen=True)
class ModuleOptimizationPlan:
    """Static metadata for one algebra-aware module in a composed model."""

    path: str
    module_type: str
    operators: tuple[str, ...]
    input_grades: Optional[tuple[int, ...]]
    output_grades: Optional[tuple[int, ...]]
    parameter_grades: Optional[tuple[int, ...]]
    score_grades: Optional[tuple[int, ...]]
    basis_dim: int
    dense_dim: int
    compact: bool
    dense_only_reason: Optional[str] = None

    @property
    def compression_ratio(self) -> float:
        """Return active basis lanes divided by dense basis lanes."""
        if self.dense_dim == 0:
            return 1.0
        return self.basis_dim / self.dense_dim

    def uses_grade(self, grade: int) -> bool:
        """Return whether the plan mentions ``grade`` in any static grade slot."""
        grade = int(grade)
        grade_sets = (self.input_grades, self.output_grades, self.parameter_grades, self.score_grades)
        return any(grades is not None and grade in grades for grades in grade_sets)


@dataclass(frozen=True)
class ModuleOptimizationIssue:
    """Static coverage issue for one algebra-aware module."""

    path: str
    module_type: str
    reason: str
    has_planned_descendants: bool


@dataclass(frozen=True)
class ModuleOptimizationReport:
    """Optimization-plan coverage for a composed module tree."""

    plans: tuple[ModuleOptimizationPlan, ...]
    issues: tuple[ModuleOptimizationIssue, ...]

    @property
    def compact_plans(self) -> tuple[ModuleOptimizationPlan, ...]:
        """Return plans that use compact basis lanes."""
        return tuple(plan for plan in self.plans if plan.compact)

    @property
    def dense_only_plans(self) -> tuple[ModuleOptimizationPlan, ...]:
        """Return plans that are explicitly dense-only."""
        return tuple(plan for plan in self.plans if not plan.compact and plan.dense_only_reason is not None)

    @property
    def unplanned_leaf_modules(self) -> tuple[ModuleOptimizationIssue, ...]:
        """Return algebra-aware modules that have no plan and no planned descendants."""
        return tuple(issue for issue in self.issues if not issue.has_planned_descendants)

    def assert_no_unplanned_leaves(self) -> None:
        """Raise if any algebra-aware leaf module is invisible to the planner."""
        leaves = self.unplanned_leaf_modules
        if not leaves:
            return
        details = ", ".join(f"{issue.path}:{issue.module_type}" for issue in leaves)
        raise AssertionError(f"Unplanned algebra-aware leaf modules: {details}")


def module_optimization_plan(module: nn.Module, *, path: str = "") -> Optional[ModuleOptimizationPlan]:
    """Return static optimization metadata for one module.

    The collector is layer-agnostic. Modules can either implement
    ``optimization_plan(path=...)`` or expose simple static attributes such as
    ``layout``, ``feature_layout``, ``score_grades``, and
    ``optimization_operators``.
    """
    custom_plan = _custom_plan(module, path)
    if custom_plan is not None:
        return custom_plan

    algebra = _module_algebra(module)
    if algebra is None:
        return None

    layout = _declared_layout(module)
    operators = _operator_tuple(getattr(module, "optimization_operators", ()))
    parameter_grades = _parameter_grades(module)
    score_grades = _score_grades(module)
    dense_only_reason = getattr(module, "optimization_dense_only_reason", None)
    if layout is None and not operators and parameter_grades is None and score_grades is None and dense_only_reason is None:
        return None

    default_grades = _grades_from_layout(layout)
    input_grades = _grade_attr(module, "optimization_input_grades", default_grades)
    output_grades = _grade_attr(module, "optimization_output_grades", default_grades)
    compact = layout is not None
    return ModuleOptimizationPlan(
        path=path or "<root>",
        module_type=module.__class__.__name__,
        operators=operators,
        input_grades=input_grades,
        output_grades=output_grades,
        parameter_grades=parameter_grades,
        score_grades=score_grades,
        basis_dim=_basis_dim(algebra, layout),
        dense_dim=int(algebra.dim),
        compact=compact,
        dense_only_reason=None if compact else dense_only_reason,
    )


def collect_module_optimization_plans(
    module: nn.Module,
    *,
    compact_only: bool = False,
) -> tuple[ModuleOptimizationPlan, ...]:
    """Collect static optimization metadata from a composed module tree."""
    plans = []
    for path, child in module.named_modules():
        plan = module_optimization_plan(child, path=path)
        if plan is None:
            continue
        if compact_only and not plan.compact:
            continue
        plans.append(plan)
    return tuple(plans)


def inspect_module_optimization(module: nn.Module) -> ModuleOptimizationReport:
    """Return static optimization plans plus coverage issues for a module tree."""
    modules = tuple(module.named_modules())
    plans = []
    plan_by_path: dict[str, ModuleOptimizationPlan] = {}

    for path, child in modules:
        normalized_path = path or "<root>"
        plan = module_optimization_plan(child, path=normalized_path)
        if plan is None:
            continue
        plans.append(plan)
        plan_by_path[normalized_path] = plan

    issues = []
    for path, child in modules:
        normalized_path = path or "<root>"
        if normalized_path in plan_by_path or _module_algebra(child) is None:
            continue
        has_planned_descendants = any(_is_descendant(plan.path, normalized_path) for plan in plans)
        reason = (
            "container has planned descendants but no direct route"
            if has_planned_descendants
            else "algebra-aware module exposes no static optimization metadata"
        )
        issues.append(
            ModuleOptimizationIssue(
                path=normalized_path,
                module_type=child.__class__.__name__,
                reason=reason,
                has_planned_descendants=has_planned_descendants,
            )
        )

    return ModuleOptimizationReport(plans=tuple(plans), issues=tuple(issues))


def _custom_plan(module: nn.Module, path: str) -> Optional[ModuleOptimizationPlan]:
    plan_fn = getattr(module, "optimization_plan", None)
    if plan_fn is None or not callable(plan_fn):
        return None
    try:
        plan = plan_fn(path=path or "<root>")
    except TypeError:
        plan = plan_fn()
    if plan is None:
        return None
    if not isinstance(plan, ModuleOptimizationPlan):
        raise TypeError(f"{module.__class__.__name__}.optimization_plan() must return ModuleOptimizationPlan or None")
    return plan


def _module_algebra(module: nn.Module) -> Optional[AlgebraLike]:
    algebra = getattr(module, "algebra", None)
    if algebra is None:
        algebra = getattr(module, "_algebra", None)
    if algebra is None or not hasattr(algebra, "planner") or not hasattr(algebra, "dim"):
        return None
    return algebra


def _declared_layout(module: nn.Module) -> Optional[GradeLayout]:
    for attr in ("optimization_layout", "layout", "feature_layout"):
        layout = getattr(module, attr, None)
        if layout is not None:
            return layout
    return None


def _grades_from_layout(layout: Optional[GradeLayout]) -> Optional[tuple[int, ...]]:
    if layout is None:
        return None
    return tuple(int(grade) for grade in layout.grades)


def _grade_attr(
    module: nn.Module,
    attr: str,
    default: Optional[tuple[int, ...]],
) -> Optional[tuple[int, ...]]:
    value = getattr(module, attr, _MISSING)
    if value is _MISSING:
        return default
    return _grade_tuple(value)


def _score_grades(module: nn.Module) -> Optional[tuple[int, ...]]:
    value = getattr(module, "optimization_score_grades", _MISSING)
    if value is not _MISSING:
        return _grade_tuple(value)

    grades = getattr(module, "score_grades", None)
    if grades is not None:
        return _grade_tuple(grades)

    score_layout = getattr(module, "_score_layout", None)
    return _grades_from_layout(score_layout)


def _parameter_grades(module: nn.Module) -> Optional[tuple[int, ...]]:
    value = getattr(module, "optimization_parameter_grades", _MISSING)
    if value is not _MISSING:
        return _grade_tuple(value)
    if hasattr(module, "grade"):
        return (int(getattr(module, "grade")),)
    return None


def _grade_tuple(grades) -> Optional[tuple[int, ...]]:
    if grades is None:
        return None
    if isinstance(grades, GradeLayout):
        return _grades_from_layout(grades)
    if isinstance(grades, int):
        return (int(grades),)
    if isinstance(grades, Iterable):
        return tuple(int(grade) for grade in grades)
    return (int(grades),)


def _operator_tuple(operators) -> tuple[str, ...]:
    if operators is None:
        return ()
    if isinstance(operators, str):
        return (operators,)
    return tuple(str(operator) for operator in operators)


def _basis_dim(algebra: AlgebraLike, layout: Optional[GradeLayout]) -> int:
    if layout is not None:
        return int(layout.dim)
    return int(algebra.dim)


def _is_descendant(path: str, parent_path: str) -> bool:
    if parent_path == "<root>":
        return path != "<root>"
    return path.startswith(parent_path + ".")
