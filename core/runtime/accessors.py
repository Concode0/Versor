# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Functional accessors for algebra layout, grade, and storage metadata."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from core.foundation.basis import normalize_grades, operation_coefficient, reverse_sign
from core.foundation.layout import AlgebraSpec, GradeLayout
from core.planning.policy import FULL_LAYOUT_MAX_N, validate_layout_cost, warn_full_layout_fallback


def resolve_layout(
    algebra,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    mv=None,
    allow_full: bool = True,
    warn_full: bool = True,
) -> GradeLayout:
    """Resolve static grade layout metadata without inspecting tensor values."""
    spec = AlgebraSpec.from_algebra(algebra)
    if layout is not None:
        _check_layout_spec(spec, layout, "layout")
        if grades is not None and layout.grades != normalize_grades(grades, spec.n, name="grades"):
            raise ValueError("layout and grades disagree")
        return validate_layout_cost(algebra, layout)

    if grades is not None:
        return validate_layout_cost(algebra, spec.layout(grades))

    if _is_multivector(mv) and getattr(mv, "layout", None) is not None:
        mv_layout = mv.layout
        _check_layout_spec(spec, mv_layout, "mv.layout")
        return validate_layout_cost(algebra, mv_layout)

    default_grades = getattr(algebra, "_default_grades", None)
    if default_grades is not None:
        cached = getattr(algebra, "_default_layout", None)
        if cached is not None:
            _check_layout_spec(spec, cached, "default_layout")
            return validate_layout_cost(algebra, cached, role="default_layout")
        resolved = spec.layout(default_grades)
        if hasattr(algebra, "_default_layout"):
            algebra._default_layout = resolved
        return validate_layout_cost(algebra, resolved, role="default_layout")

    if not allow_full or not bool(getattr(algebra, "allow_full_layout_products", True)):
        raise ValueError("No grade layout is available. Declare active grades or configure default_grades.")
    if spec.n > FULL_LAYOUT_MAX_N:
        raise ValueError(
            f"Implicit full Cl({spec.p},{spec.q},{spec.r}) layout is disabled for n>{FULL_LAYOUT_MAX_N}. "
            "Declare active grades or configure default_grades."
        )
    if warn_full:
        warn_full_layout_fallback(algebra)
    return validate_layout_cost(algebra, spec.layout(range(spec.n + 1)), role="full_layout")


def default_layout(algebra) -> GradeLayout:
    """Return the algebra default layout using the central fallback policy."""
    return resolve_layout(algebra)


def grade_indices(algebra, grades: Iterable[int], *, device=None) -> torch.Tensor:
    """Return canonical dense basis indices for ``grades``."""
    if device is None:
        device = getattr(algebra, "device", None)
    return resolve_layout(algebra, grades=grades, warn_full=False).indices_tensor(device=device)


def hermitian_signs(
    algebra,
    layout: Optional[GradeLayout] = None,
    *,
    grades: Optional[Iterable[int]] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return Hermitian metric signs for a dense or compact layout."""
    resolved = resolve_layout(algebra, layout=layout, grades=grades)
    if device is None:
        device = getattr(algebra, "device", None)
    if dtype is None:
        dtype = getattr(algebra, "dtype", torch.float32)

    dense_signs = getattr(algebra, "_hermitian_signs", None)
    if dense_signs is not None:
        indices = resolved.indices_tensor(device=dense_signs.device)
        signs = torch.index_select(dense_signs, -1, indices)
        return signs.to(device=device, dtype=dtype)

    values = [_hermitian_sign_for_index(algebra, index) for index in resolved.basis_indices]
    return torch.tensor(values, dtype=dtype, device=device)


def compact_values(
    algebra,
    value,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> tuple[torch.Tensor, GradeLayout]:
    """Return compact values plus layout for a tensor or ``Multivector``."""
    resolved = resolve_layout(algebra, layout=layout, grades=grades, mv=value)
    if _is_multivector(value):
        _check_algebra(algebra, value.algebra)
        if value.is_compact:
            return resolved.convert(value.values, value.layout), resolved
        return resolved.compact(value.coefficients), resolved

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor or Multivector-like value, got {type(value)!r}")
    if value.shape[-1] == resolved.dim:
        return value, resolved
    if value.shape[-1] == resolved.dense_dim:
        return resolved.compact(value), resolved
    raise ValueError(f"value last dimension must be {resolved.dim} compact or {resolved.dense_dim} dense")


def materialize_dense(
    algebra,
    value,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Return dense coefficients subject to the central full-layout policy."""
    if _is_multivector(value):
        _check_algebra(algebra, value.algebra)
        if not value.is_compact:
            return value.coefficients
        _check_dense_materialization_allowed(algebra)
        return value.layout.dense(value.values)

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor or Multivector-like value, got {type(value)!r}")
    if value.shape[-1] == getattr(algebra, "dim"):
        return value
    resolved = resolve_layout(algebra, layout=layout, grades=grades)
    if value.shape[-1] != resolved.dim:
        raise ValueError(f"value compact last dimension must be {resolved.dim}, got {value.shape[-1]}")
    _check_dense_materialization_allowed(algebra)
    return resolved.dense(value)


def as_multivector(
    algebra,
    value,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
):
    """Wrap a tensor or return an existing ``Multivector``."""
    from core.runtime.multivector import Multivector

    if isinstance(value, Multivector):
        _check_algebra(algebra, value.algebra)
        if layout is None and grades is None:
            return value
        resolved = resolve_layout(algebra, layout=layout, grades=grades, mv=value)
        return value.with_layout(resolved)

    if layout is None and grades is None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected Tensor or Multivector-like value, got {type(value)!r}")
        return Multivector(algebra, value)

    resolved = resolve_layout(algebra, layout=layout, grades=grades)
    if value.shape[-1] == resolved.dim:
        return Multivector(algebra, values=value, layout=resolved)
    return Multivector(algebra, tensor=value, layout=resolved)


def _hermitian_sign_for_index(algebra, index: int) -> float:
    grade = int(index).bit_count()
    grade_sign = -1.0 if grade % 2 else 1.0
    metric_sign = operation_coefficient(index, index, algebra.p, algebra.q, algebra.r, "gp")
    return grade_sign * reverse_sign(index) * metric_sign


def _check_layout_spec(spec: AlgebraSpec, layout: GradeLayout, name: str) -> None:
    if layout.spec != spec:
        raise ValueError(f"{name} signature {layout.spec} does not match algebra signature {spec}")


def _check_algebra(expected, actual) -> None:
    lhs = (expected.p, expected.q, expected.r)
    rhs = (actual.p, actual.q, actual.r)
    if lhs != rhs:
        raise ValueError(f"Algebra mismatch: Cl{lhs} vs Cl{rhs}")


def _check_dense_materialization_allowed(algebra) -> None:
    if not bool(getattr(algebra, "allow_full_layout_products", True)):
        raise ValueError("Dense materialization is disabled for this algebra. Keep compact values.")
    if getattr(algebra, "n", 0) > FULL_LAYOUT_MAX_N:
        raise ValueError(
            f"Dense materialization is disabled for n>{FULL_LAYOUT_MAX_N}. "
            "Keep compact values or declare a smaller active layout."
        )
    warn_full_layout_fallback(algebra)


def _is_multivector(value) -> bool:
    return hasattr(value, "algebra") and hasattr(value, "layout") and hasattr(value, "is_compact")
