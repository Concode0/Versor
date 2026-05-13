# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Metric definitions for Clifford algebras.

Provides distances, norms, and inner products that respect
the metric signature.
"""

from typing import Iterable, Optional

import torch

from core.foundation.layout import GradeLayout
from core.foundation.module import AlgebraLike
from core.runtime.accessors import compact_values
from core.runtime.accessors import hermitian_signs as _layout_hermitian_signs


def _hermitian_signs(
    algebra: AlgebraLike,
    layout: Optional[GradeLayout] = None,
    *,
    grades: Optional[Iterable[int]] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return Hermitian sign tensor for a dense or compact layout.

    The Hermitian inner product on Cl(p,q) is:
        <A, B>_H = sum_I (conj_sign_I * metric_sign_I) * a_I * b_I

    This is precomputed as ``conj_signs * diagonal(cayley_signs)``
    and registered as a buffer on the algebra in ``_init_derived_tables()``.

    Returns:
        Sign tensor [Dim] with values +1, -1, or 0 (null blades).
    """
    return _layout_hermitian_signs(algebra, layout=layout, grades=grades, device=device, dtype=dtype)


def inner_product(algebra: AlgebraLike, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute the scalar product via projection onto grade 0.

    Computes <A B>_0.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): First multivector [Batch, Dim].
        B (torch.Tensor): Second multivector [Batch, Dim].

    Returns:
        torch.Tensor: Scalar part [Batch, 1].
    """
    return algebra.projected_geometric_product(A, B, output_grades=(0,), compact_output=True)


def induced_norm(algebra: AlgebraLike, A: torch.Tensor) -> torch.Tensor:
    """Compute the induced norm respecting the metric signature.

    Computes ||A|| = sqrt(|<A ~A>_0|).

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): Multivector [Batch, Dim].

    Returns:
        torch.Tensor: Norm [Batch, 1].
    """
    A_rev = algebra.reverse(A)
    # Scalar product <A A~>_0
    sq_norm = inner_product(algebra, A, A_rev)

    # In mixed signatures, sq_norm can be negative.
    # We return sqrt(|sq_norm|)
    return torch.sqrt(torch.abs(sq_norm))


def geometric_distance(algebra: AlgebraLike, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Computes geometric distance.

    dist(A, B) = ||A - B||.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): First multivector.
        B (torch.Tensor): Second multivector.

    Returns:
        torch.Tensor: Distance.
    """
    diff = A - B
    return induced_norm(algebra, diff)


def grade_purity(algebra: AlgebraLike, A: torch.Tensor, grade: int) -> torch.Tensor:
    """Checks the purity of the grade by examining coefficient energy.

    Purity = ||<A>_k||^2 / ||A||^2.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): Multivector [..., Dim].
        grade (int): Target grade.

    Returns:
        torch.Tensor: Purity score [0, 1].
    """
    # Project to grade
    A_k = algebra.grade_projection(A, grade)

    # Compute energies (using standard squared norm of coefficients for stability)
    energy_k = (A_k**2).sum(dim=-1)
    energy_total = (A**2).sum(dim=-1).clamp(min=algebra.eps)

    return energy_k / energy_total


def mean_active_grade(algebra: AlgebraLike, A: torch.Tensor) -> torch.Tensor:
    """Average grade. Identifies the grade where the majority of the energy resides.

    Mean Grade = Sum(k * ||<A>_k||^2) / ||A||^2.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): Multivector.

    Returns:
        torch.Tensor: Average grade index.
    """
    energy_total = (A**2).sum(dim=-1).clamp(min=algebra.eps)
    weighted_sum = torch.zeros_like(energy_total)

    for k in range(algebra.n + 1):
        A_k = algebra.grade_projection(A, k)
        energy_k = (A_k**2).sum(dim=-1)
        weighted_sum += k * energy_k

    return weighted_sum / energy_total


# Hermitian Metrics for Mixed-Signature Algebras
#
# In Cl(p,q) with q > 0, the standard norm <A~A>_0 can be negative
# because basis blades involving negative-signature dimensions square
# to -1. This breaks gradient-based optimization.
#
# The Hermitian inner product uses the algebraically proper formula:
#
#   <A, B>_H = <bar{A} B>_0 = Sum_I (conj_sign_I * metric_sign_I) * a_I * b_I
#
# where conj_sign_I is the Clifford conjugation sign and metric_sign_I
# is the basis blade self-product sign. We precompute these signs once
# via _hermitian_signs(). For Euclidean algebras Cl(p,0), all signs are
# +1 and this reduces to the simple coefficient inner product.
#
# Additionally, we provide the Clifford conjugate (bar involution)
# and the signature-aware trace form for algebraic computations.


def clifford_conjugate(algebra: AlgebraLike, mv: torch.Tensor) -> torch.Tensor:
    """Clifford conjugation (bar involution).

    Combines reversion with grade involution:
        A_bar_k = (-1)^k * (-1)^{k(k-1)/2} * A_k

    This is the natural *-involution on Cl(p,q). Useful for
    algebraic computations (e.g., spinor norms, Lipschitz groups).

    Args:
        algebra: The algebra instance.
        mv: Multivector [..., Dim].

    Returns:
        Conjugated multivector [..., Dim].
    """
    return algebra.clifford_conjugation(mv)


def hermitian_inner_product(
    algebra: AlgebraLike,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Hermitian inner product on Cl(p,q): <bar{A} B>_0.

    <A, B>_H = Sum_I (conj_sign_I * metric_sign_I) * a_I * b_I

    Uses precomputed sign arrays so that the result equals the scalar
    part of the geometric product of the Clifford conjugate of A with B.
    For Euclidean algebras (q=0), all signs are +1 and this reduces to
    the simple coefficient inner product Sum a_I b_I.

    Args:
        algebra: The algebra instance.
        A: First multivector [..., Dim].
        B: Second multivector [..., Dim].

    Returns:
        Scalar inner product [..., 1].
    """
    A_values, B_values, resolved = _aligned_pair_values(
        algebra,
        A,
        B,
        layout=layout,
        left_layout=left_layout,
        right_layout=right_layout,
        grades=grades,
        left_grades=left_grades,
        right_grades=right_grades,
    )
    return _hermitian_inner_values(algebra, A_values, B_values, resolved)


def hermitian_norm(
    algebra: AlgebraLike,
    A: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Hermitian norm: ||A||_H = sqrt(|<A, A>_H|).

    Always real and non-negative for any signature.
    Uses abs() since the signed inner product can produce negative
    self-products in mixed-signature algebras.

    Args:
        algebra: The algebra instance.
        A: Multivector [..., Dim].

    Returns:
        Norm [..., 1]. Always >= 0.
    """
    values, resolved = compact_values(algebra, A, layout=layout, grades=grades)
    sq = _hermitian_inner_values(algebra, values, values, resolved)
    return torch.sqrt(torch.abs(sq))


def hermitian_distance(
    algebra: AlgebraLike,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Hermitian distance: d_H(A, B) = ||A - B||_H.

    Positive-definite metric distance for any signature.
    Satisfies: non-negativity, symmetry, triangle inequality, identity.

    Args:
        algebra: The algebra instance.
        A: First multivector [..., Dim].
        B: Second multivector [..., Dim].

    Returns:
        Distance [..., 1]. Always >= 0.
    """
    A_values, B_values, resolved = _aligned_pair_values(
        algebra,
        A,
        B,
        layout=layout,
        left_layout=left_layout,
        right_layout=right_layout,
        grades=grades,
        left_grades=left_grades,
        right_grades=right_grades,
    )
    diff = A_values - B_values
    sq = _hermitian_inner_values(algebra, diff, diff, resolved)
    return torch.sqrt(torch.abs(sq))


def hermitian_angle(
    algebra: AlgebraLike,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Hermitian angle between multivectors.

    cos(theta) = <A, B>_H / (||A||_H * ||B||_H)

    Args:
        algebra: The algebra instance.
        A: First multivector [..., Dim].
        B: Second multivector [..., Dim].

    Returns:
        Angle in radians [..., 1].
    """
    A_values, B_values, resolved = _aligned_pair_values(
        algebra,
        A,
        B,
        layout=layout,
        left_layout=left_layout,
        right_layout=right_layout,
        grades=grades,
        left_grades=left_grades,
        right_grades=right_grades,
    )
    signs = _signs_like(algebra, resolved, A_values, B_values)
    ip = (signs * A_values * B_values).sum(dim=-1, keepdim=True)
    sq_a = (signs * A_values * A_values).sum(dim=-1, keepdim=True)
    sq_b = (signs * B_values * B_values).sum(dim=-1, keepdim=True)
    # Use sqrt(sq_a * sq_b) instead of sqrt(sq_a)*sqrt(sq_b) to avoid
    # float32 precision loss from two separate sqrt operations.
    denom = torch.sqrt(torch.abs(sq_a) * torch.abs(sq_b)).clamp(min=algebra.eps)
    cos_theta = ip / denom
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


def grade_hermitian_norm(
    algebra: AlgebraLike,
    A: torch.Tensor,
    grade: int,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Hermitian norm restricted to a single grade.

    ||<A>_k||_H = sqrt(Sum_{I: |I|=k} a_I**2)

    Measures the energy contribution of a specific grade
    in a signature-independent way.

    Args:
        algebra: The algebra instance.
        A: Multivector [..., Dim].
        grade: Target grade.

    Returns:
        Grade-specific norm [..., 1].
    """
    values, source_layout = compact_values(algebra, A, layout=layout, grades=grades)
    grade_layout = algebra.layout((int(grade),))
    grade_values = grade_layout.convert(values, source_layout)
    sq = _hermitian_inner_values(algebra, grade_values, grade_values, grade_layout)
    return torch.sqrt(torch.abs(sq))


def hermitian_grade_spectrum(
    algebra: AlgebraLike,
    A: torch.Tensor,
    *,
    layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Full Hermitian grade spectrum.

    Returns |<A_k, A_k>_H| for each grade k = 0, ..., n.
    Uses abs() to ensure non-negative values in mixed signatures.

    Args:
        algebra: The algebra instance.
        A: Multivector [..., Dim].

    Returns:
        Grade energies [..., n+1]. Each entry >= 0.
    """
    values, source_layout = compact_values(algebra, A, layout=layout, grades=grades)
    spectrum = []
    for k in range(algebra.n + 1):
        grade_layout = algebra.layout((k,))
        grade_values = grade_layout.convert(values, source_layout)
        sq = _hermitian_inner_values(algebra, grade_values, grade_values, grade_layout)
        spectrum.append(torch.abs(sq))
    return torch.cat(spectrum, dim=-1)


def _aligned_pair_values(
    algebra: AlgebraLike,
    A,
    B,
    *,
    layout: Optional[GradeLayout] = None,
    left_layout: Optional[GradeLayout] = None,
    right_layout: Optional[GradeLayout] = None,
    grades: Optional[Iterable[int]] = None,
    left_grades: Optional[Iterable[int]] = None,
    right_grades: Optional[Iterable[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, GradeLayout]:
    """Compact two values into one static layout without dense materialization."""
    shared_left_layout = left_layout if left_layout is not None else layout
    shared_right_layout = right_layout if right_layout is not None else layout
    shared_left_grades = left_grades if left_grades is not None else grades
    shared_right_grades = right_grades if right_grades is not None else grades

    A_values, A_layout = compact_values(algebra, A, layout=shared_left_layout, grades=shared_left_grades)
    B_values, B_layout = compact_values(algebra, B, layout=shared_right_layout, grades=shared_right_grades)
    resolved = A_layout if A_layout == B_layout else _union_layout(algebra, A_layout, B_layout)
    if A_layout != resolved:
        A_values = resolved.convert(A_values, A_layout)
    if B_layout != resolved:
        B_values = resolved.convert(B_values, B_layout)
    return A_values, B_values, resolved


def _union_layout(algebra: AlgebraLike, left: GradeLayout, right: GradeLayout) -> GradeLayout:
    basis = set(left.basis_indices).union(right.basis_indices)
    grades = sorted({index.bit_count() for index in basis})
    return algebra.layout(grades)


def _hermitian_inner_values(
    algebra: AlgebraLike,
    A_values: torch.Tensor,
    B_values: torch.Tensor,
    layout: GradeLayout,
) -> torch.Tensor:
    signs = _signs_like(algebra, layout, A_values, B_values)
    return (signs * A_values * B_values).sum(dim=-1, keepdim=True)


def _signs_like(
    algebra: AlgebraLike,
    layout: GradeLayout,
    A_values: torch.Tensor,
    B_values: torch.Tensor,
) -> torch.Tensor:
    dtype = torch.promote_types(A_values.dtype, B_values.dtype)
    return _hermitian_signs(algebra, layout=layout, device=A_values.device, dtype=dtype)


def signature_trace_form(algebra: AlgebraLike, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Signature-aware trace form: <~A B>_0.

    The standard Clifford algebra scalar product. NOT positive-definite
    in mixed signatures. Use hermitian_inner_product for optimization.

    This form is signature-aware and useful for:
    - Rotor normalization (R~R = 1)
    - Versor validation
    - Spinor norm computation

    Args:
        algebra: The algebra instance.
        A: First multivector [..., Dim].
        B: Second multivector [..., Dim].

    Returns:
        Scalar trace form [..., 1]. Can be negative in mixed signatures.
    """
    A_rev = algebra.reverse(A)
    prod = algebra.geometric_product(A_rev, B)
    return prod[..., 0:1]


def signature_norm_squared(algebra: AlgebraLike, A: torch.Tensor) -> torch.Tensor:
    """Signature-aware squared norm: <A~A>_0.

    Can be negative in mixed-signature algebras. Returns the raw value
    without absolute value, preserving causal structure information.

    For Cl(n,0): always non-negative.
    For Cl(p,q) with q>0: sign encodes causal character.

    Args:
        algebra: The algebra instance.
        A: Multivector [..., Dim].

    Returns:
        Signed squared norm [..., 1].
    """
    return signature_trace_form(algebra, A, A)
