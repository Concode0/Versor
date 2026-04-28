"""Shared utilities for the SR pipeline.

Extracted from unbender.py, translator.py, implicit.py, estimator.py,
and grouper.py to eliminate duplication.
"""

import logging
import signal

import numpy as np
import sympy
import torch

logger = logging.getLogger(__name__)


# Module mapping for sympy.lambdify to ensure numpy ufuncs work on compound exprs.
# cosh/sinh use safe wrappers to prevent overflow on large theta values.
from models.sr.numerics import safe_cosh, safe_exp, safe_sinh

LAMBDIFY_MODULES = [
    {
        "log": np.log,
        "sqrt": np.sqrt,
        "Abs": np.abs,
        "sign": np.sign,
        "exp": safe_exp,
        "sin": np.sin,
        "cos": np.cos,
        "cosh": safe_cosh,
        "sinh": safe_sinh,
    },
    "numpy",
]


def make_lambdify_fn(symbols, expr):
    """Create a numpy-compatible callable from a sympy expression."""
    return sympy.lambdify(symbols, expr, modules=LAMBDIFY_MODULES)


def safe_sympy_solve(expr, var, timeout_sec=5):
    """sympy.solve with timeout and validation."""

    def handler(signum, frame):
        raise TimeoutError("sympy.solve timed out")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)
    try:
        solutions = sympy.solve(expr, var)
    except (TimeoutError, Exception):
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    if not solutions:
        return None

    # Pick simplest real solution
    for sol in solutions:
        if not sol.has(sympy.I):
            return sol
    return solutions[0]  # fallback


def safe_float(val, default=0.5):
    """Convert a value to float, replacing NaN/inf with default."""
    f = float(val) if not isinstance(val, float) else val
    if not np.isfinite(f):
        return default
    return f


def standardize(data, min_std=1e-8):
    """Zero-mean, unit-variance standardization (numpy or torch)."""
    if isinstance(data, torch.Tensor):
        mu = data.mean(0)
        std = data.std(0).clamp(min=min_std)
        return (data - mu) / std
    else:
        mu = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < min_std, 1.0, std)
        return (data - mu) / std


def subsample(data, max_size=500):
    """Subsample data to max_size rows. Works with numpy and torch."""
    if isinstance(data, torch.Tensor):
        if data.shape[0] > max_size:
            idx = torch.randperm(data.shape[0], device=data.device)[:max_size]
            return data[idx]
        return data
    else:
        if data.shape[0] > max_size:
            idx = np.random.default_rng().choice(data.shape[0], size=max_size, replace=False)
            return data[idx]
        return data


def safe_svd(X):
    """SVD with error handling. Returns (S, Vt) or (None, None)."""
    try:
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        return S, Vt
    except np.linalg.LinAlgError:
        return None, None


def evaluate_terms(terms, X_np):
    """Evaluate RotorTerms on numpy data. Standalone version without translator.

    Pads arguments with zeros if the callable expects more args than X_np has columns.
    Skips terms that produce non-finite values.
    """
    y_hat = np.zeros(X_np.shape[0])
    for t in terms:
        if t.fn is None:
            continue
        n_expected = t.fn.__code__.co_argcount
        n_vars = X_np.shape[1]
        args = [X_np[:, i] for i in range(min(n_vars, n_expected))]
        args.extend([np.zeros(X_np.shape[0])] * (n_expected - len(args)))
        val = t.weight * np.asarray(t.fn(*args), dtype=np.float64)
        val = np.broadcast_to(val, (X_np.shape[0],))
        if np.all(np.isfinite(val)):
            y_hat += val
    return y_hat


def safe_metric_search(data, device, default_n, num_probes=4, probe_epochs=40, micro_batch_size=64, max_p=None):
    """Run MetricSearch with fallback to default Cl(min(n,4),0,0).

    Consolidates the identical MetricSearch try/except pattern used in
    grouper.py (3x) and unbender.py (1x).

    Args:
        data: Torch tensor for MetricSearch.
        device: Computation device.
        default_n: Used for fallback: p = min(default_n, 4).
        num_probes: Number of probes for MetricSearch.
        probe_epochs: Epochs per probe.
        micro_batch_size: Batch size for probes.
        max_p: Optional upper bound for p (clamping).

    Returns:
        (p, q, r) tuple.
    """
    from models.sr.errors import MetricSearchError

    try:
        from core.analysis import MetricSearch

        searcher = MetricSearch(
            device=device,
            num_probes=num_probes,
            probe_epochs=probe_epochs,
            micro_batch_size=micro_batch_size,
        )
        p, q, r = searcher.search(data)
        n = p + q + r
        if n < 2:
            p = max(p, 2 - n + p)
        if max_p is not None and p > max_p:
            logger.info(f"MetricSearch clamped p: {p} -> {max_p}")
            p = max_p
        return p, q, r
    except Exception as e:
        logger.warning(f"MetricSearch failed: {e}, using Cl({min(default_n, 4)},0,0)")
        return min(default_n, 4), 0, 0


def safe_evaluate_term(fn, X_np, var_indices=None):
    """Safely evaluate a lambdified term function on data.

    Catches domain errors (log of negative, overflow, etc.) and returns
    None if the evaluation produces non-finite values.

    Args:
        fn: Callable from sympy.lambdify.
        X_np: Input data array.
        var_indices: Column indices to select from X_np (or None for all).

    Returns:
        np.ndarray of predictions, or None if evaluation failed.
    """
    if fn is None:
        return None
    try:
        X = X_np[:, var_indices] if var_indices is not None else X_np
        n_expected = fn.__code__.co_argcount
        n_vars = X.shape[1]
        args = [X[:, i] for i in range(min(n_vars, n_expected))]
        args.extend([np.zeros(X.shape[0])] * (n_expected - len(args)))
        result = np.asarray(fn(*args), dtype=np.float64)
        result = np.broadcast_to(result, (X.shape[0],)).copy()
        if np.all(np.isfinite(result)):
            return result
        return None
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None
