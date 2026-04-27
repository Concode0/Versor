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
import warnings
import torch
from typing import Tuple, List, Optional


class ExpPolicy(enum.Enum):
    """Policy controlling how ``CliffordAlgebra.exp()`` handles bivectors.

    - ``AUTO``  -- closed-form for n <= 3 (all bivectors simple),
                   compiled-safe decomposition for n >= 4.
    - ``FAST``  -- closed-form only (``_exp_bivector_closed``). Ignores
                   residual error for non-simple bivectors. Fastest path.
    - ``EXACT`` -- always use compiled-safe decomposition. Exact for all
                   bivectors, ``torch.compile``-safe (no CPU sync).
    """
    AUTO = "auto"
    FAST = "fast"
    EXACT = "exact"


# Default power-iteration step counts for the compiled-safe decomposed exp
# path, keyed by tensor dtype. Picked at the cost/benefit knee of a sweep
# over (n in {4,5,6}, magnitude in {0.1, 0.5, 1.0}) — beyond these counts a
# 2x cost buys less than ~3 decimal places of additional accuracy.
#   bfloat16 - mantissa-limited noise floor ~1e-2; saturates at k~8
#   float32  - n=4 saturates ~3e-7 by k~24; knee at k=32
#   float64  - knee at k~96 (full machine eps would need k>=256, prohibitive)
_DTYPE_FIXED_ITERATIONS = {
    torch.bfloat16: 16,
    torch.float16: 16,
    torch.float32: 32,
    torch.float64: 96,
}
_DEFAULT_FIXED_ITERATIONS = 32  # fallback for unknown dtypes


def resolve_fixed_iterations(dtype: torch.dtype) -> int:
    """Return the dtype-keyed default power-iteration count.

    Used by ``CliffordAlgebra`` at init to pin a static iteration budget
    matched to the algebra's working precision.
    """
    return _DTYPE_FIXED_ITERATIONS.get(dtype, _DEFAULT_FIXED_ITERATIONS)


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
    batch_shape = b.shape[:-1]
    device = b.device
    dtype = b.dtype

    if v_init is None:
        v_raw = torch.randn(*batch_shape, algebra.n, device=device, dtype=dtype)
        v = algebra.embed_vector(v_raw)
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


def exp_decomposed(
    algebra,
    b: torch.Tensor,
    use_decomposition: bool = True,
    k: Optional[int] = None,
    threshold: float = 1e-6,
    max_iterations: int = 100
) -> torch.Tensor:
    """Exponentiate a bivector via decomposition into simple components.

    .. deprecated::
        Use ``algebra.exp(b)`` with ``algebra.exp_policy`` set instead.
        This function is kept for backward compatibility.

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
        algebra (CliffordAlgebra): CliffordAlgebra instance.
        b: Bivector [..., dim].
        use_decomposition: Enable decomposition (False -> ``algebra._exp_bivector_closed``).
        k: Number of simple components (auto if None).
        threshold: Convergence threshold.
        max_iterations: Power iteration cap.

    Returns:
        Rotor exp(b) [..., dim].
    """
    warnings.warn(
        "exp_decomposed() is deprecated. Set algebra.exp_policy and use "
        "algebra.exp() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not use_decomposition:
        return algebra._exp_bivector_closed(b)

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
        # Plane direction (unit simple bivector) -- detached
        plane_norm = b_i_detached.norm(dim=-1, keepdim=True).clamp(min=algebra.eps_sq)
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
    batch_shape = b.shape[:-1]
    device = b.device
    dtype = b.dtype

    v_raw = torch.randn(*batch_shape, algebra.n, device=device, dtype=dtype)
    v = algebra.embed_vector(v_raw)
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
