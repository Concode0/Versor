# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

"""Geometric Metric Utilities.

Provides functions to compute geometric distances, norms, and inner products
respecting the metric signature of the algebra.
"""

import torch
from core.algebra import CliffordAlgebra

def inner_product(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Computes the scalar product (A . B) or <A B>_0.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): First multivector [Batch, Dim].
        B (torch.Tensor): Second multivector [Batch, Dim].

    Returns:
        torch.Tensor: Scalar part of the geometric product [Batch, 1].
    """
    # Optimized: A * B then extract grade 0
    # Actually, <AB>_0 = sum(A_i * B_j * <e_i e_j>_0)
    # <e_i e_j>_0 is non-zero only if i == j.
    # e_i * e_i = +/- 1 based on signature.
    
    # We can compute this efficiently by element-wise mult with metric signature
    # Get the sign of e_k * e_k
    
    # Access pre-computed metric diagonal if possible, or compute on fly
    # e_i * e_i sign is on the diagonal of the cayley_signs?
    # No, cayley_signs[i, i] is sign(e_i * e_i).
    
    # Extract diagonal signs
    # indices: [0, 1, ..., dim-1]
    # cayley_signs: [Dim, Dim]
    # We want cayley_signs[k, k] for all k
    
    # However, cayley_signs is private/cached. We can recompute or expose it.
    # Ideally, CliffordAlgebra should expose a metric_signature tensor.
    
    # Let's use algebra.geometric_product for correctness and generality first.
    prod = algebra.geometric_product(A, B)
    scalar_part = prod[..., 0:1] # Grade 0
    return scalar_part

def induced_norm(algebra: CliffordAlgebra, A: torch.Tensor) -> torch.Tensor:
    """Computes the metric-induced norm ||A|| = sqrt(|<A ~A>_0|).

    This norm respects the indefinite signature (e.g. spacetime interval).

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
    """Computes the distance between two multivectors induced by the algebra metric.

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
    """Computes the energy ratio of a specific grade k.

    Purity = ||<A>_k||^2 / ||A||^2

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        A (torch.Tensor): Multivector [..., Dim].
        grade (int): Target grade.

    Returns:
        torch.Tensor: Purity score [0, 1].
    """
    # Project to grade
    A_k = algebra.grade_projection(A, grade)
    
    # Compute energies (using standard Euclidean norm for coefficient magnitude, 
    # or induced norm? Purity usually refers to coefficient mass).
    # Let's use standard squared norm of coefficients for stability.
    energy_k = (A_k ** 2).sum(dim=-1)
    energy_total = (A ** 2).sum(dim=-1) + 1e-9
    
    return energy_k / energy_total

def mean_active_grade(algebra: CliffordAlgebra, A: torch.Tensor) -> torch.Tensor:
    """Computes the weighted mean grade of the multivector.

    Mean Grade = Sum(k * ||<A>_k||^2) / ||A||^2

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
