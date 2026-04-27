# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Bivector decomposition via GA power iteration.

Decomposes a general bivector into simple (blade) components that can each
be exponentiated with the closed-form formula.

Reference:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1 [cs.LG]
"""

import enum
import torch
from typing import Tuple, List, Optional


class ExpPolicy(enum.Enum):
    """Policy controlling how ``CliffordAlgebra.exp()`` handles bivectors.

    Both policies share the same dispatch (closed-form for n <= 3,
    compiled-safe decomposition for n >= 4) and differ only in the
    power-iteration budget used inside the decomposition path.

    - ``BALANCED`` -- knee-point iteration count, cost-efficient default.
    - ``PRECISE``  -- saturation iteration count, reaches the dtype noise
      floor at the cost of roughly 2x BALANCED's wall time.
    """
    BALANCED = "balanced"
    PRECISE = "precise"


# Power-iteration step counts for the compiled-safe decomposed exp path,
# keyed by (policy, dtype). Calibrated from separated benchmarks.
#   bf16  : noise floor ~3e-3 reached by k~8;  saturated by k~16
#   fp32  : knee at k~24 (~3e-7), floor ~3e-8 by k~64
#   fp64  : knee at k~48 (~1e-12), floor ~1e-14 by k~64; 128 = conservative cap
_ITER_BASE = {
    ExpPolicy.BALANCED: {
        torch.bfloat16: 8, torch.float16: 8,
        torch.float32: 24, torch.float64: 48,
    },
    ExpPolicy.PRECISE: {
        torch.bfloat16: 16, torch.float16: 16,
        torch.float32: 64, torch.float64: 128,
    },
}
# Added to the base count when n >= 6 to compensate for the slight error
# growth observed between n=4 and n=6 in benchmarks (e.g. fp32:
# 3.46e-8 -> 8.73e-8 at fixed iters).
_ITER_N_BUMP = {ExpPolicy.BALANCED: 8, ExpPolicy.PRECISE: 16}


def resolve_fixed_iterations(policy: ExpPolicy, dtype: torch.dtype, n: int) -> int:
    """Return the (policy, dtype, n)-keyed power-iteration count.

    Used by ``CliffordAlgebra`` at init (and on policy change) to pin a
    static iteration budget matched to the algebra's working precision
    and dimension.
    """
    base_table = _ITER_BASE[policy]
    base = base_table.get(dtype, base_table[torch.float32])
    bump = _ITER_N_BUMP[policy] if n >= 6 else 0
    return base + bump


def _seed_vector(algebra, b: torch.Tensor) -> torch.Tensor:
    """Deterministic grade-1 seed for power iteration.

    Probes ``b`` with a uniform unit vector ``(1/sqrt(n)) * sum_i e_i`` via
    right-contraction. The probe lives in ``b``'s column space, so it has
    non-zero overlap with the dominant eigenvector unless that eigenvector
    is exactly orthogonal to ``(1, ..., 1)`` -- a measure-zero case for
    which we fall back to the uniform vector itself.
    """
    batch_shape = b.shape[:-1]
    device, dtype = b.device, b.dtype
    n = algebra.n

    uniform = torch.full(
        (*batch_shape, n), 1.0 / (n ** 0.5), device=device, dtype=dtype
    )
    v_uniform = algebra.embed_vector(uniform)

    probe = algebra.right_contraction(b, v_uniform)
    probe_norm = probe.norm(dim=-1, keepdim=True)
    return torch.where(probe_norm > algebra.eps, probe, v_uniform)


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
        algebra (CliffordAlgebra): CliffordAlgebra instance.
        b: Bivector to decompose [..., dim].
        v_init: Initial grade-1 vector (random if None).
        threshold: Convergence tolerance on ``||v - v_prev||``.
        max_iterations: Iteration cap.

    Returns:
        (b_s, v) where b_s is the simple projection and v the converged
        vector, both shaped [..., dim].
    """
    if v_init is None:
        v = _seed_vector(algebra, b)
    else:
        v = v_init

    v_norm = v.norm(dim=-1, keepdim=True)
    v = v / v_norm.clamp(min=algebra.eps)

    for _ in range(max_iterations):
        v_prev = v
        v = algebra.right_contraction(b, v)
        v_norm = v.norm(dim=-1, keepdim=True)
        v = v / v_norm.clamp(min=algebra.eps)

        if (v - v_prev).norm(dim=-1).max() < threshold:
            break

    u = algebra.right_contraction(b, v)
    u_norm = u.norm(dim=-1, keepdim=True)
    u = u / u_norm.clamp(min=algebra.eps)

    b_s = u_norm * algebra.wedge(u, v)

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
        algebra (CliffordAlgebra): CliffordAlgebra instance.
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
        algebra (CliffordAlgebra): CliffordAlgebra instance.
        b: Simple bivector [..., dim].

    Returns:
        Rotor exp(b) [..., dim].
    """
    return algebra._exp_bivector_closed(b)


def _power_iteration_compiled_safe(
    algebra,
    b: torch.Tensor,
    fixed_iterations: int = 20,
) -> torch.Tensor:
    """Compile-safe power iteration for dominant simple bivector.

    Runs exactly ``fixed_iterations`` steps with no early exit.
    Converged elements are frozen via ``torch.where`` so redundant
    iterations are harmless.

    Args:
        algebra: CliffordAlgebra instance.
        b: Bivector [..., dim].
        fixed_iterations: Number of iterations (no early exit).

    Returns:
        b_s: Dominant simple bivector projection [..., dim].
    """
    v = _seed_vector(algebra, b)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=algebra.eps)

    for _ in range(fixed_iterations):
        v_prev = v
        v_new = algebra.right_contraction(b, v)
        v_new = v_new / v_new.norm(dim=-1, keepdim=True).clamp(min=algebra.eps)

        # Freeze converged elements (no CPU sync -- purely tensor ops)
        converged = (v_new - v_prev).norm(dim=-1, keepdim=True) < 1e-6
        v = torch.where(converged, v_prev, v_new)

    u = algebra.right_contraction(b, v)
    u_norm = u.norm(dim=-1, keepdim=True)
    u = u / u_norm.clamp(min=algebra.eps)

    # sigma is the eigenvalue (projection onto this plane), NOT the full norm
    b_s = u_norm * algebra.wedge(u, v)

    return b_s


def _decompose_compiled_safe(
    algebra,
    b: torch.Tensor,
    k: Optional[int] = None,
    fixed_iterations: int = 20,
) -> List[torch.Tensor]:
    """Compile-safe greedy bivector decomposition.

    Runs exactly ``k`` extraction steps (default ``n // 2``).
    Negligible residuals are masked via ``torch.where`` instead of
    early-exit.

    Args:
        algebra: CliffordAlgebra instance.
        b: Bivector [..., dim].
        k: Number of simple components (default ``n // 2``).
        fixed_iterations: Power iteration steps per component.

    Returns:
        List of k simple bivector tensors [..., dim].
    """
    n = algebra.n
    k = k if k is not None else n // 2
    k = max(k, 1)

    decomp: List[torch.Tensor] = []
    residual = b

    for _ in range(k):
        b_i = _power_iteration_compiled_safe(
            algebra, residual, fixed_iterations=fixed_iterations
        )
        # Mask: zero out extraction when residual is already negligible
        active = residual.norm(dim=-1, keepdim=True) > algebra.eps
        b_i = b_i * active.to(b_i.dtype)

        decomp.append(b_i)
        residual = residual - b_i

    return decomp


def compiled_safe_decomposed_exp(
    algebra,
    b: torch.Tensor,
    k: Optional[int] = None,
    fixed_iterations: int = 20,
) -> torch.Tensor:
    """Compile-safe decomposed exponential -- no CPU sync.

    Decomposes ``b`` into simple blades under ``torch.no_grad()``,
    re-projects the live (gradient-carrying) bivector onto each
    discovered plane, exponentiates each in closed form, and composes
    via geometric product.

    Args:
        algebra: CliffordAlgebra instance.
        b: Bivector [..., dim].
        k: Number of simple components (default ``n // 2``).
        fixed_iterations: Power iteration steps per component.

    Returns:
        Rotor exp(b) [..., dim].
    """
    n = algebra.n
    k_actual = k if k is not None else n // 2
    k_actual = max(k_actual, 1)

    # Identity rotor fallback
    identity = torch.zeros_like(b)
    identity[..., 0] = 1.0

    # Decompose (no grad -- power iteration not differentiable)
    with torch.no_grad():
        decomp = _decompose_compiled_safe(
            algebra, b.detach(), k=k_actual, fixed_iterations=fixed_iterations
        )

    bv_mask = algebra.grade_masks[2]

    # Re-project live bivector and compose rotors
    result = identity
    residual = b
    for b_i_detached in decomp:
        plane_norm = b_i_detached.norm(dim=-1, keepdim=True).clamp(min=algebra.eps_sq)
        plane_dir = b_i_detached / plane_norm

        bv_live = residual[..., bv_mask]
        plane_bv = plane_dir[..., bv_mask]
        coeff = (bv_live * plane_bv).sum(dim=-1, keepdim=True)

        b_i_live = coeff * plane_dir
        residual = residual - b_i_live

        R_i = algebra._exp_bivector_closed(b_i_live)
        result = algebra.geometric_product(result, R_i)

    return result
