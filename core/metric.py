# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Metric definitions. Where geometry meets measurement.

Provides distances, norms, and inner products that actually respect
the metric signature, unlike standard linear algebra.
"""

import torch
from core.algebra import CliffordAlgebra

def inner_product(algebra: CliffordAlgebra, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """The scalar product. Projection onto the scalar part.

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
    """The real magnitude. Respects spacetime signature.

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