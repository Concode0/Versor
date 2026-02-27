# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Bivector decomposition using power iteration.

This module implements differentiable bivector decomposition from:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1 [cs.LG]

The key insight is that general bivectors can be decomposed into sums of
simple bivectors, which can then be exponentiated using closed-form expressions
instead of scaling-and-squaring.
"""

import torch
import math
from typing import Tuple, List, Optional


def ga_power_iteration(
    algebra,
    b: torch.Tensor,
    v_init: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GA Power Iteration to find a simple bivector projection.

    Implements Algorithm 2 from Pence et al. (2025). This finds the dominant
    simple bivector component of a general bivector using power iteration,
    avoiding eigendecomposition while remaining fully differentiable.

    Reference:
        Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
        from Irreducibles." arXiv:2507.11688v1 [cs.LG], Algorithm 2, page 5

    Algorithm:
        1. Initialize random vector v
        2. Iterate: v <- b _| v, then normalize v
        3. Converge when ||v - v_prev|| < threshold
        4. Compute u = (b _| v) / ||(b _| v)||
        5. Return simple bivector b_s = sigma(u ^ v) where sigma = ||b||

    Args:
        algebra: CliffordAlgebra instance.
        b (torch.Tensor): Bivector to decompose [..., dim].
        v_init (torch.Tensor, optional): Initial vector. Random if None.
        threshold (float): Convergence threshold for ||v - v_prev||.
        max_iterations (int): Maximum iterations before stopping.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - b_s: Simple bivector projection [..., dim]
            - v: Converged vector [..., dim]
    """
    batch_shape = b.shape[:-1]
    device = b.device
    dtype = b.dtype

    # Initialize random vector in grade-1 subspace if not provided
    if v_init is None:
        v_raw = torch.randn(*batch_shape, algebra.n, device=device, dtype=dtype)
        v = algebra.embed_vector(v_raw)
    else:
        v = v_init

    # Normalize initial vector
    v_norm = v.norm(dim=-1, keepdim=True)
    v = v / (v_norm + 1e-10)

    # Power iteration
    for iteration in range(max_iterations):
        v_prev = v

        # Apply bivector: v <- b _| v
        # Note: Algorithm 2 line 4 shows single contraction per iteration
        v = algebra.right_contraction(b, v)

        # Normalize
        v_norm = v.norm(dim=-1, keepdim=True)
        v = v / (v_norm + 1e-10)

        # # Check convergence
        # diff = (v - v_prev).norm(dim=-1)
        # if (diff < threshold).all():
        #     pass

    # Compute u = (b _| v) / ||(b _| v)||
    u = algebra.right_contraction(b, v)
    u_norm = u.norm(dim=-1, keepdim=True)
    u = u / (u_norm + 1e-10)

    # Compute simple bivector: b_s = sigma(u ^ v)
    # where sigma = ||b|| is the bivector norm
    sigma = b.norm(dim=-1, keepdim=True)
    b_s = sigma * algebra.wedge(u, v)

    return b_s, v


def differentiable_invariant_decomposition(
    algebra,
    b: torch.Tensor,
    k: Optional[int] = None,
    threshold: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Decomposes a bivector into simple bivector components.

    Implements Algorithm 1 from Pence et al. (2025). This iteratively projects
    out simple bivector components using power iteration, avoiding eigendecomposition
    while remaining fully differentiable.

    Reference:
        Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
        from Irreducibles." arXiv:2507.11688v1 [cs.LG], Algorithm 1, page 5

    Algorithm:
        1. For i = 1 to k:
        2.   Find simple bivector b_i using power iteration
        3.   Subtract from residual: b <- b - b_i
        4.   Stop if ||b|| < threshold

    Args:
        algebra: CliffordAlgebra instance.
        b (torch.Tensor): Bivector to decompose [..., dim].
        k (int, optional): Number of components. Auto-determined if None.
        threshold (float): Stop when residual norm < threshold.
        max_iterations (int): Max iterations per power iteration.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]:
            - decomp: List of simple bivectors [b_1, b_2, ..., b_k]
            - vectors: List of corresponding vectors [v_1, v_2, ..., v_k]
    """
    # Determine maximum number of components: k_max = n(n-1)/2
    n = algebra.n
    k_max = (n * (n - 1)) // 2

    if k is None:
        k = k_max
    else:
        k = min(k, k_max)

    decomp = []
    vectors = []
    residual = b.clone()

    for i in range(k):
        # # Check if residual is negligible
        # residual_norm = residual.norm(dim=-1)
        # if (residual_norm < threshold).all():
        #     break

        # Project out next simple bivector
        b_i, v_i = ga_power_iteration(
            algebra, residual, threshold=threshold, max_iterations=max_iterations
        )

        decomp.append(b_i)
        vectors.append(v_i)

        # Subtract from residual
        residual = residual - b_i

    return decomp, vectors


def exp_simple_bivector(algebra, b: torch.Tensor) -> torch.Tensor:
    """Exponentiates a simple bivector using the algebra's closed-form expression.

    Delegates to ``algebra._exp_bivector_closed(b)`` which correctly handles
    all three signature regimes (elliptic, hyperbolic, parabolic) instead of
    assuming Euclidean L2 norm.

    Reference:
        Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
        from Irreducibles." arXiv:2507.11688v1 [cs.LG], Equation 5, page 5

    Args:
        algebra: CliffordAlgebra instance.
        b (torch.Tensor): Simple bivector [..., dim].

    Returns:
        torch.Tensor: Rotor exp(b) [..., dim].
    """
    return algebra._exp_bivector_closed(b)


def exp_decomposed(
    algebra,
    b: torch.Tensor,
    use_decomposition: bool = True,
    k: Optional[int] = None,
    threshold: float = 1e-6,
    max_iterations: int = 100
) -> torch.Tensor:
    """Exponentiates a bivector using optional decomposition.

    If use_decomposition is True, decomposes the bivector into simple components
    and exponentiates each using closed form, then composes the results.
    Otherwise, falls back to standard scaling-and-squaring method.

    Reference:
        Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
        from Irreducibles." arXiv:2507.11688v1 [cs.LG]

    Algorithm:
        1. Decompose: [b_1, b_2, ..., b_k] <- decomposition(b)
        2. Exponentiate each: R_i <- exp(b_i) using closed form
        3. Compose: R <- R_1 * R_2 * ... * R_k

    Args:
        algebra: CliffordAlgebra instance.
        b (torch.Tensor): Bivector to exponentiate [..., dim].
        use_decomposition (bool): If True, use decomposition method.
        k (int, optional): Number of components for decomposition.
        threshold (float): Convergence threshold.
        max_iterations (int): Max iterations for power iteration.

    Returns:
        torch.Tensor: Rotor exp(b) [..., dim].
    """
    if not use_decomposition:
        # Fall back to standard method
        return algebra.exp(b)

    # Decompose bivector
    decomp, _ = differentiable_invariant_decomposition(
        algebra, b, k=k, threshold=threshold, max_iterations=max_iterations
    )

    # Handle case where decomposition is empty (zero bivector)
    if len(decomp) == 0:
        result = torch.zeros_like(b)
        result[..., 0] = 1.0  # Identity rotor
        return result

    # Exponentiate each component using closed form
    rotors = [exp_simple_bivector(algebra, b_i) for b_i in decomp]

    # Compose rotors via geometric product: R = R_1 * R_2 * ... * R_k
    result = rotors[0]
    for R_i in rotors[1:]:
        result = algebra.geometric_product(result, R_i)

    return result
