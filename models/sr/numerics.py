# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Safe numerical primitives for the SR pipeline.

Centralizes clamping, guarding, and conditioning logic so that
individual pipeline stages do not each reinvent stability hacks.
"""

import logging
import signal

import numpy as np
import sympy

logger = logging.getLogger(__name__)

# Maximum safe argument for np.exp / float64 (~1.7e308)
_EXP_MAX = 700.0
# Maximum safe |theta| for cosh/sinh (cosh(700) ~ 1e304)
_HYPER_MAX_THETA = 350.0


def safe_exp(x, max_val=_EXP_MAX):
    """np.exp with input clamping to prevent overflow."""
    return np.exp(np.clip(x, -max_val, max_val))


def safe_log(x, eps=1e-30):
    """np.log with a floor to prevent -inf / NaN."""
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), eps))


def safe_cosh(x, max_val=_HYPER_MAX_THETA):
    """np.cosh with input clamping to prevent overflow."""
    return np.cosh(np.clip(x, -max_val, max_val))


def safe_sinh(x, max_val=_HYPER_MAX_THETA):
    """np.sinh with input clamping to prevent overflow."""
    return np.sinh(np.clip(x, -max_val, max_val))


def safe_inv_sqrt_diag(diag_vals, eps=1e-10):
    """1/sqrt(d) for each element, returning 0 where d <= eps.

    Used for the normalized Laplacian in spectral clustering where
    disconnected nodes give zero degree.
    """
    diag_vals = np.asarray(diag_vals, dtype=np.float64)
    safe = np.where(diag_vals > eps, diag_vals, 1.0)
    inv_sqrt = 1.0 / np.sqrt(safe)
    return np.where(diag_vals > eps, inv_sqrt, 0.0)


def safe_lstsq(A, y, rcond=None):
    """Least-squares solve with column normalization for conditioning.

    Normalizes each column of A to unit variance before solving,
    then rescales the solution weights accordingly.

    Returns:
        weights: shape [n_cols] — rescaled to match original A.
    """
    col_scales = np.linalg.norm(A, axis=0)
    col_scales = np.where(col_scales < 1e-12, 1.0, col_scales)
    A_norm = A / col_scales
    weights_norm = np.linalg.lstsq(A_norm, y, rcond=rcond)[0]
    return weights_norm / col_scales


def safe_simplify(expr, timeout_sec=5):
    """sympy.simplify with a timeout to prevent hangs on complex expressions.

    Falls back to the original expression if simplification times out
    or raises an error.
    """
    def _handler(signum, frame):
        raise TimeoutError("sympy.simplify timed out")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        result = sympy.simplify(expr)
    except (TimeoutError, Exception):
        logger.debug(f"safe_simplify timed out or failed, returning original expr")
        result = expr
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return result


def clamp_theta(theta, max_val=_HYPER_MAX_THETA):
    """Clamp a scalar angle for safe use in cosh/sinh."""
    return max(min(float(theta), max_val), -max_val)
