# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Feynman Symbolic Regression Benchmark Dataset.

15 curated equations from the Feynman Lectures on Physics.
Each equation maps k scalar inputs to one scalar output.
The goal is to discover the underlying symbolic structure.

Reference: Udrescu & Tegmark (2019), arXiv:1905.11483
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Per-equation samplers (each returns x: [n, k] and y: [n])
# ---------------------------------------------------------------------------

def _gen_I_12_1(n, rng):
    """F1 + F2"""
    x = rng.uniform(1, 5, (n, 2))
    return x, x[:, 0] + x[:, 1]


def _gen_I_26_2(n, rng):
    """arcsin(n * sin(theta2))"""
    n_var = rng.uniform(0.5, 0.99, n)
    theta2 = rng.uniform(0.1, 0.9, n)
    x = np.column_stack([n_var, theta2])
    return x, np.arcsin(n_var * np.sin(theta2))


def _gen_I_6_20(n, rng):
    """Gaussian: exp(-(theta-theta1)^2/(2*sigma^2)) / sqrt(2*pi*sigma^2)"""
    theta = rng.uniform(-3, 3, n)
    theta1 = rng.uniform(-3, 3, n)
    sigma = rng.uniform(0.1, 3, n)
    x = np.column_stack([theta, theta1, sigma])
    y = np.exp(-(theta - theta1) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    return x, y


def _gen_I_24_6(n, rng):
    """m * (omega^2 + omega0^2) / 4"""
    m = rng.uniform(1, 5, n)
    omega = rng.uniform(1, 5, n)
    omega0 = rng.uniform(1, 5, n)
    x = np.column_stack([m, omega, omega0])
    return x, m * (omega ** 2 + omega0 ** 2) / 4


def _gen_I_34_1(n, rng):
    """omega0 / (1 - v/c)  [omega0, v, c]"""
    omega0 = rng.uniform(1, 5, n)
    beta = rng.uniform(0.01, 0.9, n)   # v/c
    c = rng.uniform(1, 3, n)
    v = beta * c
    x = np.column_stack([omega0, v, c])
    return x, omega0 / (1 - beta)


def _gen_I_37_4(n, rng):
    """I1 + I2 + 2*sqrt(I1*I2)*cos(delta)"""
    I1 = rng.uniform(1, 5, n)
    I2 = rng.uniform(1, 5, n)
    delta = rng.uniform(-1, 1, n)
    x = np.column_stack([I1, I2, delta])
    return x, I1 + I2 + 2 * np.sqrt(I1 * I2) * np.cos(delta)


def _gen_I_43_43(n, rng):
    """mu * T / q"""
    x = rng.uniform(1, 5, (n, 3))
    return x, x[:, 0] * x[:, 1] / x[:, 2]


def _gen_I_12_2(n, rng):
    """q1*q2 / (4*pi*eps*r^2)"""
    x = rng.uniform(1, 5, (n, 4))
    y = x[:, 0] * x[:, 1] / (4 * np.pi * x[:, 2] * x[:, 3] ** 2)
    return x, y


def _gen_I_15_1(n, rng):
    """(x - u*t) / sqrt(1 - (u/c)^2)  [x, u, t, c]"""
    x_pos = rng.uniform(1, 5, n)
    beta = rng.uniform(0.01, 0.9, n)   # u/c
    c = rng.uniform(1, 3, n)
    u = beta * c
    t = rng.uniform(0.1, 2, n)
    x = np.column_stack([x_pos, u, t, c])
    return x, (x_pos - u * t) / np.sqrt(1 - beta ** 2)


def _gen_I_50_26(n, rng):
    """x1 * (cos(omega*t) + alpha*cos^2(omega*t))"""
    x1 = rng.uniform(1, 5, n)
    omega = rng.uniform(1, 5, n)
    t = rng.uniform(0.1, 2, n)
    alpha = rng.uniform(0.1, 1, n)
    x = np.column_stack([x1, omega, t, alpha])
    ct = np.cos(omega * t)
    return x, x1 * (ct + alpha * ct ** 2)


def _gen_II_2_42(n, rng):
    """kappa * dT * A / d  [kappa, dT, A, d]"""
    x = rng.uniform(1, 5, (n, 4))
    return x, x[:, 0] * x[:, 1] * x[:, 2] / x[:, 3]


def _gen_I_34_27(n, rng):
    """omega0 * (1 + v*cos(theta)/c) / sqrt(1-(v/c)^2)  [omega0, v, theta, c]"""
    omega0 = rng.uniform(1, 5, n)
    beta = rng.uniform(0.01, 0.9, n)   # v/c
    theta = rng.uniform(0, 2 * np.pi, n)
    c = rng.uniform(1, 3, n)
    v = beta * c
    x = np.column_stack([omega0, v, theta, c])
    y = omega0 * (1 + v * np.cos(theta) / c) / np.sqrt(1 - beta ** 2)
    return x, y


def _gen_III_4_33(n, rng):
    """h*omega / (exp(h*omega/(k*T)) - 1)  [h, omega, k, T]
    Ratio h*omega/(k*T) controlled to be in (0.1, 5) for stability.
    """
    h = rng.uniform(0.1, 1.0, n)
    omega = rng.uniform(1.0, 5.0, n)
    ratio = rng.uniform(0.1, 5.0, n)   # h*omega / (k*T)
    k = rng.uniform(0.5, 2.0, n)
    T = h * omega / (ratio * k)        # ensures ratio is correct
    x = np.column_stack([h, omega, k, T])
    return x, h * omega / (np.exp(ratio) - 1.0)


def _gen_I_9_18(n, rng):
    """G*m1*m2 / r^2  [G, m1, m2, r]  (r = r2 - r1 > 0)"""
    x = rng.uniform(1, 5, (n, 4))
    return x, x[:, 0] * x[:, 1] * x[:, 2] / x[:, 3] ** 2


def _gen_II_11_27(n, rng):
    """n0 * exp(-m*u^2 / (2*k*T))  [n0, m, u, k, T]"""
    n0 = rng.uniform(1, 5, n)
    m = rng.uniform(0.1, 1, n)
    u = rng.uniform(0.1, 2, n)
    k = rng.uniform(0.5, 5, n)
    T = rng.uniform(1, 10, n)
    x = np.column_stack([n0, m, u, k, T])
    return x, n0 * np.exp(-m * u ** 2 / (2 * k * T))


# ---------------------------------------------------------------------------
# Equation registry
# ---------------------------------------------------------------------------

FEYNMAN_EQUATIONS = {
    "I.12.1":  {"n_vars": 2, "fn": _gen_I_12_1,   "desc": "F1 + F2"},
    "I.26.2":  {"n_vars": 2, "fn": _gen_I_26_2,   "desc": "arcsin(n*sin(theta2))"},
    "I.6.20":  {"n_vars": 3, "fn": _gen_I_6_20,   "desc": "Gaussian distribution"},
    "I.24.6":  {"n_vars": 3, "fn": _gen_I_24_6,   "desc": "m*(omega^2+omega0^2)/4"},
    "I.34.1":  {"n_vars": 3, "fn": _gen_I_34_1,   "desc": "omega0/(1-v/c)"},
    "I.37.4":  {"n_vars": 3, "fn": _gen_I_37_4,   "desc": "I1+I2+2*sqrt(I1*I2)*cos(delta)"},
    "I.43.43": {"n_vars": 3, "fn": _gen_I_43_43,  "desc": "mu*T/q"},
    "I.12.2":  {"n_vars": 4, "fn": _gen_I_12_2,   "desc": "q1*q2/(4*pi*eps*r^2)"},
    "I.15.1":  {"n_vars": 4, "fn": _gen_I_15_1,   "desc": "(x-u*t)/sqrt(1-(u/c)^2)"},
    "I.50.26": {"n_vars": 4, "fn": _gen_I_50_26,  "desc": "x1*(cos(w*t)+alpha*cos^2(w*t))"},
    "II.2.42": {"n_vars": 4, "fn": _gen_II_2_42,  "desc": "kappa*(T2-T1)*A/d"},
    "I.34.27": {"n_vars": 4, "fn": _gen_I_34_27,  "desc": "omega0*(1+v*cos(theta)/c)/sqrt(1-(v/c)^2)"},
    "III.4.33":{"n_vars": 4, "fn": _gen_III_4_33, "desc": "h*omega/(exp(h*omega/(k*T))-1)"},
    "I.9.18":  {"n_vars": 4, "fn": _gen_I_9_18,   "desc": "G*m1*m2/r^2"},
    "II.11.27":{"n_vars": 5, "fn": _gen_II_11_27, "desc": "n0*exp(-m*u^2/(2*k*T))"},
}


# ---------------------------------------------------------------------------
# Dataset and loaders
# ---------------------------------------------------------------------------

class FeynmanDataset(Dataset):
    """Holds normalised (x, y) pairs for one Feynman equation.

    Attributes:
        x (Tensor): Inputs [N, k].
        y (Tensor): Targets [N, 1].
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _load_or_generate(
    equation: str,
    n_samples: int,
    noise: float,
    cache_dir: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load from cache if available, else generate synthetic data.

    Returns:
        x: [n_samples, k] float32 tensor
        y: [n_samples, 1] float32 tensor
    """
    if cache_dir:
        eq_tag = equation.replace(".", "_")
        cache_path = os.path.join(cache_dir, f"{eq_tag}_{n_samples}.pt")
        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            return data["x"], data["y"]

    eq_spec = FEYNMAN_EQUATIONS[equation]
    rng = np.random.default_rng(seed)
    x_np, y_np = eq_spec["fn"](n_samples, rng)

    if noise > 0:
        y_np = y_np + rng.normal(0, noise * float(np.std(y_np)), len(y_np))

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1)   # [N, 1]

    # Filter NaN / Inf that might arise from edge cases
    valid = torch.isfinite(x).all(dim=1) & torch.isfinite(y).squeeze(-1)
    x, y = x[valid], y[valid]

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"x": x, "y": y}, cache_path)

    return x, y


def get_feynman_loaders(
    equation: str = "I.6.20",
    n_samples: int = 10000,
    batch_size: int = 64,
    noise: float = 0.0,
    cache_dir: str = "./data/feynman",
    seed: int = 42,
) -> tuple:
    """Build train / val / test DataLoaders for one Feynman equation.

    Split: 70 / 15 / 15.  Normalisation is computed on the training split.

    Returns:
        train_loader, val_loader, test_loader,
        x_mean [k], x_std [k], y_mean scalar, y_std scalar
    """
    if equation not in FEYNMAN_EQUATIONS:
        raise ValueError(f"Unknown equation '{equation}'. "
                         f"Available: {list(FEYNMAN_EQUATIONS)}")

    x, y = _load_or_generate(equation, n_samples, noise, cache_dir, seed)
    N = len(x)

    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)

    # Deterministic split (data was generated with fixed seed)
    train_x, train_y = x[:n_train], y[:n_train]
    val_x,   val_y   = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    test_x,  test_y  = x[n_train + n_val :], y[n_train + n_val :]

    # Normalise using training statistics
    x_mean = train_x.mean(0)
    x_std  = train_x.std(0).clamp(min=1e-6)
    y_mean = train_y.mean()
    y_std  = train_y.std().clamp(min=1e-6)

    def _norm_x(t): return (t - x_mean) / x_std
    def _norm_y(t): return (t - y_mean) / y_std

    train_ds = FeynmanDataset(_norm_x(train_x), _norm_y(train_y))
    val_ds   = FeynmanDataset(_norm_x(val_x),   _norm_y(val_y))
    test_ds  = FeynmanDataset(_norm_x(test_x),  _norm_y(test_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std
