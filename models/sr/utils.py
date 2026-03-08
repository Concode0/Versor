"""Shared utilities for the SR pipeline.

Extracted from unbender.py, translator.py, implicit.py, estimator.py,
and grouper.py to eliminate duplication.
"""

import signal

import numpy as np
import sympy
import torch


# Module mapping for sympy.lambdify to ensure numpy ufuncs work on compound exprs
LAMBDIFY_MODULES = [{"log": np.log, "sqrt": np.sqrt, "Abs": np.abs,
                      "sign": np.sign, "exp": np.exp, "sin": np.sin,
                      "cos": np.cos}, "numpy"]


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
