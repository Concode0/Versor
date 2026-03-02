# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Bivector decomposition via GA power iteration.

Decomposes a general bivector into simple (blade) components that can each
be exponentiated with the closed-form formula.

Reference:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1 [cs.LG]
"""

import torch
from typing import Tuple, List, Optional


def ga_power_iteration(
    algebra,
    b: torch.Tensor,
    v_init: Optional[torch.Tensor] = None,
    threshold: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the dominant simple bivector component via power iteration.

    Implements Algorithm 2 from Pence et al. (2025).  Iterates
    ``v <- (b _| v) / ||b _| v||`` until convergence, then recovers
    the simple projection ``b_s = sigma * (u ^ v)``.

    Args:
        algebra: CliffordAlgebra instance.
        b: Bivector to decompose [..., dim].
        v_init: Initial grade-1 vector (random if None).
        threshold: Convergence tolerance on ``||v - v_prev||``.
        max_iterations: Iteration cap.

    Returns:
        (b_s, v) where b_s is the simple projection and v the converged
        vector, both shaped [..., dim].
    """
    batch_shape = b.shape[:-1]
    device = b.device
    dtype = b.dtype

    if v_init is None:
        v_raw = torch.randn(*batch_shape, algebra.n, device=device, dtype=dtype)
        v = algebra.embed_vector(v_raw)
    else:
        v = v_init

    v_norm = v.norm(dim=-1, keepdim=True)
    v = v / v_norm.clamp(min=1e-6)

    for _ in range(max_iterations):
        v_prev = v
        v = algebra.right_contraction(b, v)
        v_norm = v.norm(dim=-1, keepdim=True)
        v = v / v_norm.clamp(min=1e-6)

        if (v - v_prev).norm(dim=-1).max() < threshold:
            break

    u = algebra.right_contraction(b, v)
    u_norm = u.norm(dim=-1, keepdim=True)
    u = u / u_norm.clamp(min=1e-6)

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
    """Decompose a bivector into simple components via greedy projection.

    Implements Algorithm 1 from Pence et al. (2025).  Iteratively
    extracts the dominant simple component and subtracts it from the
    residual.

    Args:
        algebra: CliffordAlgebra instance.
        b: Bivector [..., dim].
        k: Number of components (auto = n(n-1)/2 if None).
        threshold: Stop when residual norm falls below this.
        max_iterations: Per-component power iteration cap.

    Returns:
        (decomp, vectors): lists of simple bivectors and their
        associated vectors.
    """
    n = algebra.n
    k_max = (n * (n - 1)) // 2
    k = min(k, k_max) if k is not None else k_max

    decomp: List[torch.Tensor] = []
    vectors: List[torch.Tensor] = []
    residual = b.clone()

    for _ in range(k):
        if residual.norm(dim=-1).max() < threshold:
            break

        b_i, v_i = ga_power_iteration(
            algebra, residual, threshold=threshold, max_iterations=max_iterations
        )
        decomp.append(b_i)
        vectors.append(v_i)
        residual = residual - b_i

    return decomp, vectors


def exp_simple_bivector(algebra, b: torch.Tensor) -> torch.Tensor:
    """Closed-form exponential of a simple bivector.

    Delegates to ``algebra._exp_bivector_closed`` which handles all
    three signature regimes (elliptic, hyperbolic, parabolic).

    Args:
        algebra: CliffordAlgebra instance.
        b: Simple bivector [..., dim].

    Returns:
        Rotor exp(b) [..., dim].
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
    """Exponentiate a bivector via decomposition into simple components.

    When ``use_decomposition`` is True the bivector is decomposed into
    simple blades (via ``differentiable_invariant_decomposition``), each
    is exponentiated in closed form, and the rotors are composed via
    geometric product.

    During training the power iteration loop is **detached** (run in
    forward-only mode) so that gradients do not flow through the
    normalization divisions.  Gradients instead flow through the
    closed-form exp of each component and the final GP composition.
    This is stable for all bivector magnitudes.

    Args:
        algebra: CliffordAlgebra instance.
        b: Bivector [..., dim].
        use_decomposition: Enable decomposition (False -> ``algebra.exp``).
        k: Number of simple components (auto if None).
        threshold: Convergence threshold.
        max_iterations: Power iteration cap.

    Returns:
        Rotor exp(b) [..., dim].
    """
    if not use_decomposition:
        return algebra.exp(b)

    # Detach for decomposition (power iteration is not differentiable)
    # then re-project the original bivector onto the discovered planes.
    with torch.no_grad():
        decomp, _ = differentiable_invariant_decomposition(
            algebra, b.detach(), k=k, threshold=threshold,
            max_iterations=max_iterations
        )

    if len(decomp) == 0:
        result = torch.zeros_like(b)
        result[..., 0] = 1.0
        return result

    bv_mask = algebra.grade_masks[2]
    if bv_mask.device != b.device:
        bv_mask = bv_mask.to(b.device)

    rotors = []
    residual = b
    for b_i_detached in decomp:
        # Plane direction (unit simple bivector) — detached
        plane_norm = b_i_detached.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        plane_dir = b_i_detached / plane_norm  # detached unit plane

        # Project the live bivector onto this plane
        bv_live = residual[..., bv_mask]
        plane_bv = plane_dir[..., bv_mask]
        coeff = (bv_live * plane_bv).sum(dim=-1, keepdim=True)

        b_i_live = coeff * plane_dir
        residual = residual - b_i_live

        rotors.append(exp_simple_bivector(algebra, b_i_live))

    result = rotors[0]
    for R_i in rotors[1:]:
        result = algebra.geometric_product(result, R_i)

    return result
