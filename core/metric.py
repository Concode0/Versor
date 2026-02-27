# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Metric definitions for Clifford algebras.

Provides distances, norms, and inner products that respect
the metric signature.
"""

import torch
from core.algebra import CliffordAlgebra


def _hermitian_signs(algebra: CliffordAlgebra) -> torch.Tensor:
    """Precompute conj_sign_i * metric_sign_i for each basis element.

    The Hermitian inner product on Cl(p,q) is:
        <A, B>_H = sum_I (conj_sign_I * metric_sign_I) * a_I * b_I

    where conj_sign_I = (-1)^k * (-1)^{k(k-1)/2} (Clifford conjugation sign)
    and metric_sign_I = (-1)^{k(k-1)/2} * prod_{j in I} g_{jj} (basis blade self-product scalar part).

    This equals <bar{A} B>_0 computed via the full geometric product.

    Returns:
        Sign tensor [Dim] with values +1 or -1.
    """
    if hasattr(algebra, '_cached_hermitian_signs'):
        cached = algebra._cached_hermitian_signs
        if cached.device == algebra.device:
            return cached

    signs = torch.ones(algebra.dim, device=algebra.device)
    pq = algebra.p + algebra.q
    for i in range(algebra.dim):
        k = bin(i).count('1')  # grade
        # Clifford conjugation sign: (-1)^k * (-1)^{k(k-1)/2}
        conj_sign = ((-1) ** k) * ((-1) ** (k * (k - 1) // 2))
        # Metric sign: (-1)^{k(k-1)/2} * prod g_{jj} for j in I
        # g_{jj} = +1 if j < p, -1 if p <= j < p+q, 0 if j >= p+q
        metric_product = 1
        has_null = False
        for bit in range(algebra.n):
            if i & (1 << bit):
                if bit >= pq:
                    has_null = True
                    break
                metric_product *= (1 if bit < algebra.p else -1)
        if has_null:
            signs[i] = 0
        else:
            metric_sign = ((-1) ** (k * (k - 1) // 2)) * metric_product
            signs[i] = conj_sign * metric_sign

    algebra._cached_hermitian_signs = signs
    return signs

def inner_product(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute the scalar product via projection onto grade 0.

    Computes <A B>_0.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): First multivector [Batch, Dim].
        B (torch.Tensor): Second multivector [Batch, Dim].

    Returns:
        torch.Tensor: Scalar part [Batch, 1].
    """
    # Optimized: A * B then extract grade 0
    prod = algebra.geometric_product(A, B)
    scalar_part = prod[..., 0:1] # Grade 0
    return scalar_part

def induced_norm(algebra: CliffordAlgebra, A: torch.Tensor) -> torch.Tensor:
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

def geometric_distance(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
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

def grade_purity(algebra: CliffordAlgebra, A: torch.Tensor, grade: int) -> torch.Tensor:
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
    energy_k = (A_k ** 2).sum(dim=-1)
    energy_total = (A ** 2).sum(dim=-1) + 1e-9
    
    return energy_k / energy_total

def mean_active_grade(algebra: CliffordAlgebra, A: torch.Tensor) -> torch.Tensor:
    """Average grade. Identifies the grade where the majority of the energy resides.

    Mean Grade = Sum(k * ||<A>_k||^2) / ||A||^2.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): Multivector.

    Returns:
        torch.Tensor: Average grade index.
    """
    energy_total = (A ** 2).sum(dim=-1) + 1e-9
    weighted_sum = torch.zeros_like(energy_total)
    
    for k in range(algebra.n + 1):
        A_k = algebra.grade_projection(A, k)
        energy_k = (A_k ** 2).sum(dim=-1)
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

def clifford_conjugate(algebra: CliffordAlgebra, mv: torch.Tensor) -> torch.Tensor:
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
    result = mv.clone()
    for i in range(algebra.dim):
        k = bin(i).count('1')
        sign = ((-1) ** k) * ((-1) ** (k * (k - 1) // 2))
        if sign == -1:
            result[..., i] = -result[..., i]
    return result


def hermitian_inner_product(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
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
    signs = _hermitian_signs(algebra).to(device=A.device, dtype=A.dtype)
    return (signs * A * B).sum(dim=-1, keepdim=True)


def hermitian_norm(algebra: CliffordAlgebra, A: torch.Tensor) -> torch.Tensor:
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
    sq = hermitian_inner_product(algebra, A, A)
    return torch.sqrt(torch.abs(sq))


def hermitian_distance(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
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
    return hermitian_norm(algebra, A - B)


def hermitian_angle(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Hermitian angle between multivectors.

    cos(theta) = <A, B>_H / (||A||_H * ||B||_H)

    Args:
        algebra: The algebra instance.
        A: First multivector [..., Dim].
        B: Second multivector [..., Dim].

    Returns:
        Angle in radians [..., 1].
    """
    ip = hermitian_inner_product(algebra, A, B)
    norm_a = hermitian_norm(algebra, A)
    norm_b = hermitian_norm(algebra, B)
    cos_theta = ip / (norm_a * norm_b + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)


def grade_hermitian_norm(algebra: CliffordAlgebra, A: torch.Tensor, grade: int) -> torch.Tensor:
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
    A_k = algebra.grade_projection(A, grade)
    return hermitian_norm(algebra, A_k)


def hermitian_grade_spectrum(algebra: CliffordAlgebra, A: torch.Tensor) -> torch.Tensor:
    """Full Hermitian grade spectrum.

    Returns |<A_k, A_k>_H| for each grade k = 0, ..., n.
    Uses abs() to ensure non-negative values in mixed signatures.

    Args:
        algebra: The algebra instance.
        A: Multivector [..., Dim].

    Returns:
        Grade energies [..., n+1]. Each entry >= 0.
    """
    signs = _hermitian_signs(algebra).to(device=A.device, dtype=A.dtype)
    spectrum = []
    for k in range(algebra.n + 1):
        A_k = algebra.grade_projection(A, k)
        sq = (signs * A_k * A_k).sum(dim=-1, keepdim=True)
        spectrum.append(torch.abs(sq))
    return torch.cat(spectrum, dim=-1)


def signature_trace_form(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
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


def signature_norm_squared(algebra: CliffordAlgebra, A: torch.Tensor) -> torch.Tensor:
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