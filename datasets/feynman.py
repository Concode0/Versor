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

100 equations from the Feynman Lectures on Physics (AI Feynman benchmark).
Each equation maps k scalar inputs to one scalar output.
The goal is to discover the underlying symbolic structure.

Reference: Udrescu & Tegmark (2020), arXiv:1905.11481
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Safe math namespace for formula evaluation
# ---------------------------------------------------------------------------

_SAFE_MATH = {
    "__builtins__": {},
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "abs": np.abs,
    "pi": np.pi,
    "sign": np.sign,
    "minimum": np.minimum,
    "maximum": np.maximum,
}


def _generate_from_spec(spec, n, rng):
    """Generate (x, y) from equation spec with variable ranges.

    Args:
        spec: dict with "formula", "variables", and optional "sampling".
        n: number of samples.
        rng: numpy random generator.

    Returns:
        x: [n, k] array, y: [n] array.
    """
    variables = {}
    for var in spec["variables"]:
        lo, hi = var["low"], var["high"]
        if var.get("log_sample", False):
            variables[var["name"]] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        else:
            variables[var["name"]] = rng.uniform(lo, hi, n)

    # Apply sampling constraints
    sampling = spec.get("sampling")
    if sampling and "constraints" in sampling:
        for constraint in sampling["constraints"]:
            constraint(variables, rng, n)

    x = np.column_stack([variables[v["name"]] for v in spec["variables"]])
    namespace = {**_SAFE_MATH, **variables}
    y = eval(spec["formula"], namespace)  # noqa: S307
    return x, y


# ---------------------------------------------------------------------------
# Sampling constraints for edge cases
# ---------------------------------------------------------------------------

def _constrain_relativistic_v(variables, rng, n):
    """Ensure v < c by sampling beta = v/c in (0.01, 0.9)."""
    if "v" in variables and "c" in variables:
        beta = rng.uniform(0.01, 0.9, n)
        variables["v"] = beta * variables["c"]


def _constrain_relativistic_v1(variables, rng, n):
    """Ensure v1 < c by sampling beta in (0.01, 0.9)."""
    if "v1" in variables and "c" in variables:
        beta = rng.uniform(0.01, 0.9, n)
        variables["v1"] = beta * variables["c"]


def _constrain_relativistic_v2(variables, rng, n):
    """Ensure v2 < c by sampling beta in (0.01, 0.9)."""
    if "v2" in variables and "c" in variables:
        beta = rng.uniform(0.01, 0.9, n)
        variables["v2"] = beta * variables["c"]


def _constrain_bose_einstein(variables, rng, n):
    """Control h*omega/(k*T) ratio to avoid overflow in exp."""
    if all(k in variables for k in ("h", "omega", "kb", "T")):
        ratio = rng.uniform(0.1, 5.0, n)
        variables["T"] = variables["h"] * variables["omega"] / (ratio * variables["kb"])


def _constrain_arcsin_arg(variables, rng, n):
    """Ensure arcsin argument stays in (-1, 1)."""
    if "n_refr" in variables and "theta2" in variables:
        # n_refr * sin(theta2) must be in (-1, 1)
        variables["n_refr"] = rng.uniform(0.3, 0.99, n)
        variables["theta2"] = rng.uniform(0.1, 0.9, n)


def _constrain_waveguide(variables, rng, n):
    """Ensure omega^2/c^2 > pi^2/d^2 for real waveguide wave number."""
    if all(k in variables for k in ("omega", "c", "d_val")):
        # cutoff = pi*c/d; need omega > cutoff
        cutoff = np.pi * variables["c"] / variables["d_val"]
        variables["omega"] = cutoff * rng.uniform(1.1, 3.0, n)


# ---------------------------------------------------------------------------
# 100-equation registry (AI Feynman benchmark)
# ---------------------------------------------------------------------------

FEYNMAN_EQUATIONS = {
    # ===== Volume I =====

    "I.6.2a": {
        "n_vars": 1, "tier": 1,
        "formula": "exp(-theta**2 / 2) / sqrt(2 * pi)",
        "variables": [{"name": "theta", "low": 1.0, "high": 3.0}],
        "output": "f", "desc": "Gaussian (unit sigma)",
    },
    "I.6.2b": {
        "n_vars": 2, "tier": 1,
        "formula": "exp(-theta**2 / (2 * sigma**2)) / sqrt(2 * pi * sigma**2)",
        "variables": [
            {"name": "theta", "low": 1.0, "high": 3.0},
            {"name": "sigma", "low": 1.0, "high": 3.0},
        ],
        "output": "f", "desc": "Gaussian (variable sigma)",
    },
    "I.6.20": {
        "n_vars": 3, "tier": 2,
        "formula": "exp(-(theta - theta1)**2 / (2 * sigma**2)) / sqrt(2 * pi * sigma**2)",
        "variables": [
            {"name": "theta", "low": -3.0, "high": 3.0},
            {"name": "theta1", "low": -3.0, "high": 3.0},
            {"name": "sigma", "low": 0.1, "high": 3.0},
        ],
        "output": "f", "desc": "Gaussian distribution",
    },
    "I.8.14": {
        "n_vars": 2, "tier": 1,
        "formula": "sqrt((x2 - x1)**2 + (y2 - y1)**2)",
        "variables": [
            {"name": "x1", "low": 1.0, "high": 5.0},
            {"name": "x2", "low": 1.0, "high": 5.0},
            {"name": "y1", "low": 1.0, "high": 5.0},
            {"name": "y2", "low": 1.0, "high": 5.0},
        ],
        "output": "d", "desc": "Euclidean distance 2D",
        "n_vars": 4,
    },
    "I.9.18": {
        "n_vars": 6, "tier": 4,
        "formula": "F / sqrt(1 - (v**2 / c**2))",
        "variables": [
            {"name": "F", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "F_rel", "desc": "Relativistic force (3 var simplified)",
        "n_vars": 3,
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "I.10.7": {
        "n_vars": 2, "tier": 1,
        "formula": "m_0 / sqrt(1 - (v / c)**2)",
        "variables": [
            {"name": "m_0", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "m", "desc": "Relativistic mass",
        "n_vars": 3,
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "I.11.19": {
        "n_vars": 3, "tier": 1,
        "formula": "x1 * y1 + x2 * y2 + x3 * y3",
        "variables": [
            {"name": "x1", "low": 1.0, "high": 5.0},
            {"name": "x2", "low": 1.0, "high": 5.0},
            {"name": "x3", "low": 1.0, "high": 5.0},
            {"name": "y1", "low": 1.0, "high": 5.0},
            {"name": "y2", "low": 1.0, "high": 5.0},
            {"name": "y3", "low": 1.0, "high": 5.0},
        ],
        "output": "A", "desc": "3D dot product",
        "n_vars": 6,
    },
    "I.12.1": {
        "n_vars": 2, "tier": 1,
        "formula": "F1 + F2",
        "variables": [
            {"name": "F1", "low": 1.0, "high": 5.0},
            {"name": "F2", "low": 1.0, "high": 5.0},
        ],
        "output": "F", "desc": "F1 + F2",
    },
    "I.12.2": {
        "n_vars": 4, "tier": 2,
        "formula": "q1 * q2 / (4 * pi * epsilon * r**2)",
        "variables": [
            {"name": "q1", "low": 1.0, "high": 5.0},
            {"name": "q2", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
        ],
        "output": "F", "desc": "q1*q2/(4*pi*eps*r^2)",
    },
    "I.12.4": {
        "n_vars": 2, "tier": 1,
        "formula": "q1 * r / (4 * pi * epsilon * r**3)",
        "variables": [
            {"name": "q1", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
        ],
        "output": "Ef", "desc": "Coulomb field (q/(4*pi*eps*r^2))",
        "n_vars": 3,
    },
    "I.12.11": {
        "n_vars": 3, "tier": 1,
        "formula": "q * (Ef + B * v * sin(theta))",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
            {"name": "B", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.1, "high": 3.0},
        ],
        "output": "F", "desc": "Lorentz force",
        "n_vars": 5,
    },
    "I.13.4": {
        "n_vars": 3, "tier": 1,
        "formula": "m * (v**2 / 2 + g * y - g * z)",
        "variables": [
            {"name": "m", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "g", "low": 1.0, "high": 5.0},
            {"name": "y", "low": 1.0, "high": 5.0},
            {"name": "z", "low": 1.0, "high": 5.0},
        ],
        "output": "K", "desc": "Kinetic + potential energy",
        "n_vars": 5,
    },
    "I.13.12": {
        "n_vars": 4, "tier": 1,
        "formula": "G * m1 * m2 * (1 / r2 - 1 / r1)",
        "variables": [
            {"name": "G", "low": 1.0, "high": 5.0},
            {"name": "m1", "low": 1.0, "high": 5.0},
            {"name": "m2", "low": 1.0, "high": 5.0},
            {"name": "r1", "low": 1.0, "high": 5.0},
            {"name": "r2", "low": 1.0, "high": 5.0},
        ],
        "output": "U", "desc": "Gravitational potential difference",
        "n_vars": 5,
    },
    "I.14.3": {
        "n_vars": 3, "tier": 1,
        "formula": "m * g * z",
        "variables": [
            {"name": "m", "low": 1.0, "high": 5.0},
            {"name": "g", "low": 1.0, "high": 5.0},
            {"name": "z", "low": 1.0, "high": 5.0},
        ],
        "output": "U", "desc": "m*g*z",
    },
    "I.14.4": {
        "n_vars": 3, "tier": 1,
        "formula": "k_spring * x**2 / 2",
        "variables": [
            {"name": "k_spring", "low": 1.0, "high": 5.0},
            {"name": "x", "low": 1.0, "high": 5.0},
        ],
        "output": "U", "desc": "Spring potential energy",
        "n_vars": 2,
    },
    "I.15.1": {
        "n_vars": 4, "tier": 3,
        "formula": "(x_pos - u * t) / sqrt(1 - (u / c)**2)",
        "variables": [
            {"name": "x_pos", "low": 1.0, "high": 5.0},
            {"name": "u", "low": 1.0, "high": 5.0},
            {"name": "t", "low": 0.1, "high": 2.0},
            {"name": "c", "low": 1.0, "high": 3.0},
        ],
        "output": "x_prime", "desc": "(x-u*t)/sqrt(1-(u/c)^2)",
        "sampling": {"constraints": [
            lambda v, r, n: v.__setitem__("u", r.uniform(0.01, 0.9, n) * v["c"])
        ]},
    },
    "I.15.3t": {
        "n_vars": 4, "tier": 3,
        "formula": "(t - u * x_pos / c**2) / sqrt(1 - (u / c)**2)",
        "variables": [
            {"name": "x_pos", "low": 1.0, "high": 5.0},
            {"name": "u", "low": 1.0, "high": 5.0},
            {"name": "t", "low": 0.1, "high": 2.0},
            {"name": "c", "low": 1.0, "high": 3.0},
        ],
        "output": "t_prime", "desc": "Lorentz time transformation",
        "sampling": {"constraints": [
            lambda v, r, n: v.__setitem__("u", r.uniform(0.01, 0.9, n) * v["c"])
        ]},
    },
    "I.15.10": {
        "n_vars": 3, "tier": 2,
        "formula": "m_0 * v / sqrt(1 - (v / c)**2)",
        "variables": [
            {"name": "m_0", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "p", "desc": "Relativistic momentum",
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "I.16.6": {
        "n_vars": 4, "tier": 2,
        "formula": "(u + v) / (1 + u * v / c**2)",
        "variables": [
            {"name": "u", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 3.0, "high": 10.0},
        ],
        "output": "v_rel", "desc": "Relativistic velocity addition",
        "n_vars": 3,
    },
    "I.18.4": {
        "n_vars": 3, "tier": 2,
        "formula": "(m1 * r1 + m2 * r2) / (m1 + m2)",
        "variables": [
            {"name": "m1", "low": 1.0, "high": 5.0},
            {"name": "m2", "low": 1.0, "high": 5.0},
            {"name": "r1", "low": 1.0, "high": 5.0},
            {"name": "r2", "low": 1.0, "high": 5.0},
        ],
        "output": "r_cm", "desc": "Center of mass",
        "n_vars": 4,
    },
    "I.18.12": {
        "n_vars": 3, "tier": 2,
        "formula": "r * F * sin(theta)",
        "variables": [
            {"name": "r", "low": 1.0, "high": 5.0},
            {"name": "F", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.1, "high": 3.0},
        ],
        "output": "tau", "desc": "Torque (r*F*sin(theta))",
    },
    "I.18.16": {
        "n_vars": 5, "tier": 2,
        "formula": "m * r * v * sin(theta)",
        "variables": [
            {"name": "m", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.1, "high": 3.0},
        ],
        "output": "L", "desc": "Angular momentum",
        "n_vars": 4,
    },
    "I.24.6": {
        "n_vars": 3, "tier": 1,
        "formula": "m * (omega**2 + omega_0**2) / 4",
        "variables": [
            {"name": "m", "low": 1.0, "high": 5.0},
            {"name": "omega", "low": 1.0, "high": 5.0},
            {"name": "omega_0", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "m*(omega^2+omega0^2)/4",
    },
    "I.25.13": {
        "n_vars": 2, "tier": 1,
        "formula": "q / C",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "C", "low": 1.0, "high": 5.0},
        ],
        "output": "Volt", "desc": "Capacitor voltage q/C",
    },
    "I.26.2": {
        "n_vars": 2, "tier": 2,
        "formula": "arcsin(n_refr * sin(theta2))",
        "variables": [
            {"name": "n_refr", "low": 0.3, "high": 0.99},
            {"name": "theta2", "low": 0.1, "high": 0.9},
        ],
        "output": "theta1", "desc": "arcsin(n*sin(theta2))",
        "sampling": {"constraints": [_constrain_arcsin_arg]},
    },
    "I.27.6": {
        "n_vars": 3, "tier": 2,
        "formula": "1 / (1 / d1 + n_refr / d2)",
        "variables": [
            {"name": "d1", "low": 1.0, "high": 5.0},
            {"name": "d2", "low": 1.0, "high": 5.0},
            {"name": "n_refr", "low": 0.5, "high": 3.0},
        ],
        "output": "foc", "desc": "Thin lens (1/d1+n/d2)^-1",
    },
    "I.29.4": {
        "n_vars": 2, "tier": 1,
        "formula": "omega / c",
        "variables": [
            {"name": "omega", "low": 1.0, "high": 10.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "k", "desc": "Wave number omega/c",
    },
    "I.29.16": {
        "n_vars": 2, "tier": 1,
        "formula": "sqrt(x1**2 + x2**2 - 2 * x1 * x2 * cos(theta1 - theta2))",
        "variables": [
            {"name": "x1", "low": 1.0, "high": 5.0},
            {"name": "x2", "low": 1.0, "high": 5.0},
            {"name": "theta1", "low": 0.0, "high": 6.28},
            {"name": "theta2", "low": 0.0, "high": 6.28},
        ],
        "output": "A", "desc": "Phasor addition",
        "n_vars": 4,
    },
    "I.30.3": {
        "n_vars": 3, "tier": 2,
        "formula": "Int_0 * sin(n_val * theta / 2)**2 / (n_val * theta / 2)**2",
        "variables": [
            {"name": "Int_0", "low": 1.0, "high": 5.0},
            {"name": "n_val", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.1, "high": 3.0},
        ],
        "output": "Int", "desc": "Single-slit diffraction intensity",
    },
    "I.30.5": {
        "n_vars": 2, "tier": 1,
        "formula": "arcsin(lambd / (n_val * d))",
        "variables": [
            {"name": "lambd", "low": 0.1, "high": 1.0},
            {"name": "d", "low": 2.0, "high": 5.0},
            {"name": "n_val", "low": 1.0, "high": 3.0},
        ],
        "output": "theta", "desc": "Diffraction grating angle",
        "n_vars": 3,
    },
    "I.32.5": {
        "n_vars": 4, "tier": 2,
        "formula": "q**2 * a**2 / (6 * pi * epsilon * c**3)",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "a", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "P", "desc": "Larmor radiation power",
    },
    "I.34.1": {
        "n_vars": 3, "tier": 2,
        "formula": "omega_0 / (1 - v / c)",
        "variables": [
            {"name": "omega_0", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 3.0},
        ],
        "output": "omega", "desc": "omega0/(1-v/c)",
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "I.34.8": {
        "n_vars": 3, "tier": 2,
        "formula": "q * v * B / p",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "B", "low": 1.0, "high": 5.0},
            {"name": "p", "low": 1.0, "high": 5.0},
        ],
        "output": "omega_cyc", "desc": "Cyclotron frequency",
        "n_vars": 4,
    },
    "I.34.14": {
        "n_vars": 3, "tier": 2,
        "formula": "(1 + v / c) / sqrt(1 - (v / c)**2) * omega_0",
        "variables": [
            {"name": "omega_0", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "omega", "desc": "Relativistic Doppler",
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "I.34.27": {
        "n_vars": 4, "tier": 3,
        "formula": "omega_0 * (1 + v * cos(theta) / c) / sqrt(1 - (v / c)**2)",
        "variables": [
            {"name": "omega_0", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.0, "high": 6.28},
            {"name": "c", "low": 1.0, "high": 3.0},
        ],
        "output": "omega", "desc": "omega0*(1+v*cos(theta)/c)/sqrt(1-(v/c)^2)",
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "I.37.4": {
        "n_vars": 3, "tier": 2,
        "formula": "I1 + I2 + 2 * sqrt(I1 * I2) * cos(delta)",
        "variables": [
            {"name": "I1", "low": 1.0, "high": 5.0},
            {"name": "I2", "low": 1.0, "high": 5.0},
            {"name": "delta", "low": -1.0, "high": 1.0},
        ],
        "output": "Int", "desc": "I1+I2+2*sqrt(I1*I2)*cos(delta)",
    },
    "I.38.12": {
        "n_vars": 4, "tier": 2,
        "formula": "4 * pi * epsilon * h**2 / (m * q**2)",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "h", "low": 1.0, "high": 5.0},
            {"name": "m", "low": 1.0, "high": 5.0},
            {"name": "q", "low": 1.0, "high": 5.0},
        ],
        "output": "r", "desc": "Bohr radius",
    },
    "I.39.1": {
        "n_vars": 4, "tier": 3,
        "formula": "3 / 2 * pr * V / (3 / 2 + 3 / 2)",
        "variables": [
            {"name": "pr", "low": 1.0, "high": 5.0},
            {"name": "V", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Gas energy 3/2 pV",
        "n_vars": 2,
    },
    "I.39.11": {
        "n_vars": 3, "tier": 2,
        "formula": "1 / (gamma_val - 1) * pr * V",
        "variables": [
            {"name": "gamma_val", "low": 1.1, "high": 2.0},
            {"name": "pr", "low": 1.0, "high": 5.0},
            {"name": "V", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Ideal gas energy (1/(gamma-1))*pV",
    },
    "I.39.22": {
        "n_vars": 3, "tier": 2,
        "formula": "n_kB * T / V",
        "variables": [
            {"name": "n_kB", "low": 1.0, "high": 5.0},
            {"name": "T", "low": 1.0, "high": 5.0},
            {"name": "V", "low": 1.0, "high": 5.0},
        ],
        "output": "pr", "desc": "Ideal gas law nkT/V",
    },
    "I.40.1": {
        "n_vars": 4, "tier": 3,
        "formula": "n_0 * exp(-m * g * x / (kb * T))",
        "variables": [
            {"name": "n_0", "low": 1.0, "high": 5.0},
            {"name": "m", "low": 0.1, "high": 1.0},
            {"name": "g", "low": 1.0, "high": 5.0},
            {"name": "x", "low": 0.1, "high": 2.0},
            {"name": "kb", "low": 0.5, "high": 5.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "n", "desc": "Boltzmann distribution",
        "n_vars": 6,
    },
    "I.41.16": {
        "n_vars": 4, "tier": 3,
        "formula": "h * omega**3 / (pi**2 * c**2 * (exp(h * omega / (kb * T)) - 1))",
        "variables": [
            {"name": "h", "low": 0.1, "high": 1.0},
            {"name": "omega", "low": 1.0, "high": 5.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "L_w", "desc": "Planck radiation spectral density",
        "n_vars": 5,
        "sampling": {"constraints": [_constrain_bose_einstein]},
    },
    "I.43.31": {
        "n_vars": 3, "tier": 2,
        "formula": "mob * kb * T",
        "variables": [
            {"name": "mob", "low": 1.0, "high": 5.0},
            {"name": "kb", "low": 1.0, "high": 5.0},
            {"name": "T", "low": 1.0, "high": 5.0},
        ],
        "output": "D", "desc": "Einstein diffusion D=mu*kB*T",
    },
    "I.43.43": {
        "n_vars": 3, "tier": 1,
        "formula": "mu_val * T / q",
        "variables": [
            {"name": "mu_val", "low": 1.0, "high": 5.0},
            {"name": "T", "low": 1.0, "high": 5.0},
            {"name": "q", "low": 1.0, "high": 5.0},
        ],
        "output": "kappa", "desc": "mu*T/q",
    },
    "I.44.4": {
        "n_vars": 4, "tier": 3,
        "formula": "n_kB * kb * T * log(V2 / V1)",
        "variables": [
            {"name": "n_kB", "low": 1.0, "high": 5.0},
            {"name": "kb", "low": 1.0, "high": 5.0},
            {"name": "T", "low": 1.0, "high": 5.0},
            {"name": "V1", "low": 1.0, "high": 5.0},
            {"name": "V2", "low": 1.0, "high": 5.0},
        ],
        "output": "W", "desc": "Isothermal work n*k*T*ln(V2/V1)",
        "n_vars": 5,
    },
    "I.47.23": {
        "n_vars": 2, "tier": 1,
        "formula": "sqrt(gamma_val * pr / rho)",
        "variables": [
            {"name": "gamma_val", "low": 1.1, "high": 2.0},
            {"name": "pr", "low": 1.0, "high": 5.0},
            {"name": "rho", "low": 1.0, "high": 5.0},
        ],
        "output": "c_s", "desc": "Speed of sound",
        "n_vars": 3,
    },
    "I.48.2": {
        "n_vars": 3, "tier": 2,
        "formula": "m * c**2 / sqrt(1 - (v / c)**2)",
        "variables": [
            {"name": "m", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Relativistic energy",
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "I.50.26": {
        "n_vars": 4, "tier": 2,
        "formula": "x1 * (cos(omega * t) + alpha * cos(omega * t)**2)",
        "variables": [
            {"name": "x1", "low": 1.0, "high": 5.0},
            {"name": "omega", "low": 1.0, "high": 5.0},
            {"name": "t", "low": 0.1, "high": 2.0},
            {"name": "alpha", "low": 0.1, "high": 1.0},
        ],
        "output": "x_out", "desc": "x1*(cos(w*t)+alpha*cos^2(w*t))",
    },

    # ===== Volume II =====

    "II.2.42": {
        "n_vars": 4, "tier": 1,
        "formula": "kappa * dT * A / d_val",
        "variables": [
            {"name": "kappa", "low": 1.0, "high": 5.0},
            {"name": "dT", "low": 1.0, "high": 5.0},
            {"name": "A", "low": 1.0, "high": 5.0},
            {"name": "d_val", "low": 1.0, "high": 5.0},
        ],
        "output": "Pwr", "desc": "kappa*(T2-T1)*A/d",
    },
    "II.3.18": {
        "n_vars": 4, "tier": 2,
        "formula": "4 * pi * epsilon * Ef * r**2",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
        ],
        "output": "q_e", "desc": "Gauss law: q = 4*pi*eps*E*r^2",
        "n_vars": 3,
    },
    "II.4.23": {
        "n_vars": 2, "tier": 1,
        "formula": "q / (4 * pi * epsilon * r)",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
        ],
        "output": "Volt", "desc": "Coulomb potential q/(4*pi*eps*r)",
        "n_vars": 3,
    },
    "II.6.11": {
        "n_vars": 4, "tier": 3,
        "formula": "1 / (4 * pi * epsilon) * p_dip * cos(theta) / r**2",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "p_dip", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.1, "high": 3.0},
            {"name": "r", "low": 1.0, "high": 5.0},
        ],
        "output": "Volt", "desc": "Dipole potential",
    },
    "II.6.15a": {
        "n_vars": 3, "tier": 2,
        "formula": "epsilon * Ef**2 / 2",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
        ],
        "output": "u", "desc": "Electric field energy density",
        "n_vars": 2,
    },
    "II.6.15b": {
        "n_vars": 3, "tier": 2,
        "formula": "p_dip * Ef * cos(theta)",
        "variables": [
            {"name": "p_dip", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.0, "high": 3.14},
        ],
        "output": "U", "desc": "Dipole energy in field",
    },
    "II.8.7": {
        "n_vars": 2, "tier": 1,
        "formula": "3 / 5 * q**2 / (4 * pi * epsilon * d_val)",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "d_val", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Energy of charged sphere",
        "n_vars": 3,
    },
    "II.8.31": {
        "n_vars": 3, "tier": 2,
        "formula": "epsilon * Ef**2 / 2",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
        ],
        "output": "u", "desc": "Electric energy density (II)",
        "n_vars": 2,
    },
    "II.11.3": {
        "n_vars": 3, "tier": 1,
        "formula": "n_0 * (exp(mu_val * Ef / (kb * T)) + exp(-mu_val * Ef / (kb * T)))",
        "variables": [
            {"name": "n_0", "low": 1.0, "high": 5.0},
            {"name": "mu_val", "low": 0.1, "high": 1.0},
            {"name": "Ef", "low": 0.1, "high": 2.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "Nox", "desc": "Langevin paramagnetism (exp sum)",
        "n_vars": 5,
    },
    "II.11.17": {
        "n_vars": 4, "tier": 3,
        "formula": "n_0 * (exp(p_dip * Ef / (kb * T)) - exp(-p_dip * Ef / (kb * T))) / (exp(p_dip * Ef / (kb * T)) + exp(-p_dip * Ef / (kb * T)))",
        "variables": [
            {"name": "n_0", "low": 1.0, "high": 5.0},
            {"name": "p_dip", "low": 0.1, "high": 1.0},
            {"name": "Ef", "low": 0.1, "high": 2.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "x", "desc": "Langevin paramagnetism (tanh form)",
        "n_vars": 5,
    },
    "II.11.27": {
        "n_vars": 5, "tier": 3,
        "formula": "n_0 * exp(-m * u**2 / (2 * kb * T))",
        "variables": [
            {"name": "n_0", "low": 1.0, "high": 5.0},
            {"name": "m", "low": 0.1, "high": 1.0},
            {"name": "u", "low": 0.1, "high": 2.0},
            {"name": "kb", "low": 0.5, "high": 5.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "n", "desc": "n0*exp(-m*u^2/(2*k*T))",
    },
    "II.11.28": {
        "n_vars": 4, "tier": 2,
        "formula": "1 + n_0 * alpha_pol / (1 - n_0 * alpha_pol / 3)",
        "variables": [
            {"name": "n_0", "low": 0.1, "high": 2.0},
            {"name": "alpha_pol", "low": 0.1, "high": 2.0},
        ],
        "output": "theta", "desc": "Clausius-Mossotti",
        "n_vars": 2,
    },
    "II.13.17": {
        "n_vars": 3, "tier": 2,
        "formula": "1 / (4 * pi * epsilon * c**2) * 2 * I_curr / r",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
            {"name": "I_curr", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
        ],
        "output": "B", "desc": "Magnetic field from long wire",
        "n_vars": 4,
    },
    "II.13.23": {
        "n_vars": 2, "tier": 1,
        "formula": "rho_ch * c_val",
        "variables": [
            {"name": "rho_ch", "low": 1.0, "high": 5.0},
            {"name": "c_val", "low": 1.0, "high": 5.0},
        ],
        "output": "J", "desc": "Current density rho*v",
    },
    "II.13.34": {
        "n_vars": 3, "tier": 2,
        "formula": "rho_ch * v / sqrt(1 - (v / c)**2)",
        "variables": [
            {"name": "rho_ch", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "j", "desc": "Relativistic current density",
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "II.15.4": {
        "n_vars": 3, "tier": 2,
        "formula": "-mom * B * cos(theta)",
        "variables": [
            {"name": "mom", "low": 1.0, "high": 5.0},
            {"name": "B", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.0, "high": 3.14},
        ],
        "output": "E", "desc": "Magnetic dipole energy",
    },
    "II.15.5": {
        "n_vars": 3, "tier": 2,
        "formula": "-p_dip * Ef * cos(theta)",
        "variables": [
            {"name": "p_dip", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.0, "high": 3.14},
        ],
        "output": "E", "desc": "Electric dipole energy (negative)",
    },
    "II.21.32": {
        "n_vars": 3, "tier": 2,
        "formula": "q / (4 * pi * epsilon * r * (1 - v / c))",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
        ],
        "output": "Volt", "desc": "Lienard-Wiechert potential",
        "n_vars": 5,
        "sampling": {"constraints": [_constrain_relativistic_v]},
    },
    "II.24.17": {
        "n_vars": 2, "tier": 1,
        "formula": "sqrt(omega**2 / c**2 - pi**2 / d_val**2)",
        "variables": [
            {"name": "omega", "low": 5.0, "high": 10.0},
            {"name": "c", "low": 1.0, "high": 3.0},
            {"name": "d_val", "low": 0.5, "high": 2.0},
        ],
        "output": "k", "desc": "Waveguide wave number",
        "n_vars": 3,
        "sampling": {"constraints": [_constrain_waveguide]},
    },
    "II.27.16": {
        "n_vars": 3, "tier": 2,
        "formula": "epsilon * c * Ef**2",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
        ],
        "output": "flux", "desc": "Poynting vector magnitude",
    },
    "II.27.18": {
        "n_vars": 2, "tier": 1,
        "formula": "epsilon * Ef**2",
        "variables": [
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
        ],
        "output": "u", "desc": "EM energy density eps*E^2",
    },
    "II.34.2a": {
        "n_vars": 3, "tier": 2,
        "formula": "q * v * r / 2",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
        ],
        "output": "mom", "desc": "Orbital magnetic moment",
    },
    "II.34.2": {
        "n_vars": 4, "tier": 2,
        "formula": "q * h / (2 * m)",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "h", "low": 1.0, "high": 5.0},
            {"name": "m", "low": 1.0, "high": 5.0},
        ],
        "output": "mom", "desc": "Magnetic moment (Bohr magneton)",
        "n_vars": 3,
    },
    "II.34.11": {
        "n_vars": 3, "tier": 2,
        "formula": "g_val * q * B / (2 * m)",
        "variables": [
            {"name": "g_val", "low": 1.0, "high": 5.0},
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "B", "low": 1.0, "high": 5.0},
            {"name": "m", "low": 1.0, "high": 5.0},
        ],
        "output": "omega", "desc": "Larmor precession",
        "n_vars": 4,
    },
    "II.34.29a": {
        "n_vars": 3, "tier": 2,
        "formula": "q * h / (4 * pi * m)",
        "variables": [
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "h", "low": 1.0, "high": 5.0},
            {"name": "m", "low": 1.0, "high": 5.0},
        ],
        "output": "mom", "desc": "Bohr magneton (h-bar form)",
    },
    "II.34.29b": {
        "n_vars": 3, "tier": 2,
        "formula": "g_val * mom * B * Jz / (kb * T)",
        "variables": [
            {"name": "g_val", "low": 1.0, "high": 3.0},
            {"name": "mom", "low": 0.1, "high": 2.0},
            {"name": "B", "low": 1.0, "high": 5.0},
            {"name": "Jz", "low": 0.5, "high": 3.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "M", "desc": "Paramagnetic magnetization",
        "n_vars": 6,
    },
    "II.35.18": {
        "n_vars": 3, "tier": 2,
        "formula": "n_0 / (exp(mom * B / (kb * T)) + exp(-mom * B / (kb * T)))",
        "variables": [
            {"name": "n_0", "low": 1.0, "high": 5.0},
            {"name": "mom", "low": 0.1, "high": 1.0},
            {"name": "B", "low": 0.1, "high": 2.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "n_avg", "desc": "Partition function (2-level)",
        "n_vars": 5,
    },
    "II.35.21": {
        "n_vars": 3, "tier": 3,
        "formula": "n_0 * mom * tanh(mom * B / (kb * T))",
        "variables": [
            {"name": "n_0", "low": 1.0, "high": 5.0},
            {"name": "mom", "low": 0.1, "high": 1.0},
            {"name": "B", "low": 0.1, "high": 2.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "M", "desc": "Magnetization (tanh)",
        "n_vars": 5,
    },
    "II.36.38": {
        "n_vars": 5, "tier": 4,
        "formula": "mom_A * mom_B / (4 * pi * epsilon * c**2 * r**5) * (3 * cos(theta)**2 - 1)",
        "variables": [
            {"name": "mom_A", "low": 1.0, "high": 5.0},
            {"name": "mom_B", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "c", "low": 1.0, "high": 5.0},
            {"name": "r", "low": 1.0, "high": 5.0},
            {"name": "theta", "low": 0.1, "high": 3.0},
        ],
        "output": "E", "desc": "Dipole-dipole interaction",
        "n_vars": 6,
    },
    "II.37.1": {
        "n_vars": 5, "tier": 3,
        "formula": "mom * (1 + chi) * B",
        "variables": [
            {"name": "mom", "low": 1.0, "high": 5.0},
            {"name": "chi", "low": 0.01, "high": 1.0},
            {"name": "B", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Susceptibility energy",
        "n_vars": 3,
    },
    "II.38.3": {
        "n_vars": 3, "tier": 2,
        "formula": "Y_mod * A_cs * x / d_val",
        "variables": [
            {"name": "Y_mod", "low": 1.0, "high": 5.0},
            {"name": "A_cs", "low": 1.0, "high": 5.0},
            {"name": "x", "low": 0.1, "high": 2.0},
            {"name": "d_val", "low": 1.0, "high": 5.0},
        ],
        "output": "F", "desc": "Hooke's law (Y*A*x/d)",
        "n_vars": 4,
    },

    # ===== Volume III =====

    "III.4.32": {
        "n_vars": 3, "tier": 3,
        "formula": "1 / (exp(h * omega / (kb * T)) - 1)",
        "variables": [
            {"name": "h", "low": 0.1, "high": 1.0},
            {"name": "omega", "low": 1.0, "high": 5.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 5.0},
        ],
        "output": "n_avg", "desc": "Bose-Einstein occupation number",
        "n_vars": 4,
        "sampling": {"constraints": [_constrain_bose_einstein]},
    },
    "III.4.33": {
        "n_vars": 4, "tier": 3,
        "formula": "h * omega / (exp(h * omega / (kb * T)) - 1)",
        "variables": [
            {"name": "h", "low": 0.1, "high": 1.0},
            {"name": "omega", "low": 1.0, "high": 5.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 5.0},
        ],
        "output": "E_avg", "desc": "h*omega/(exp(h*omega/(k*T))-1)",
        "sampling": {"constraints": [_constrain_bose_einstein]},
    },
    "III.7.38": {
        "n_vars": 4, "tier": 3,
        "formula": "2 * mom * A * cos(E_n * t / h) * cos(omega * t)",
        "variables": [
            {"name": "mom", "low": 1.0, "high": 5.0},
            {"name": "A", "low": 1.0, "high": 3.0},
            {"name": "E_n", "low": 1.0, "high": 5.0},
            {"name": "t", "low": 0.1, "high": 2.0},
            {"name": "h", "low": 1.0, "high": 5.0},
            {"name": "omega", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Ammonia maser transition",
        "n_vars": 6,
    },
    "III.8.54": {
        "n_vars": 3, "tier": 2,
        "formula": "sin(E_n * t / h)**2",
        "variables": [
            {"name": "E_n", "low": 1.0, "high": 5.0},
            {"name": "t", "low": 0.1, "high": 2.0},
            {"name": "h", "low": 1.0, "high": 5.0},
        ],
        "output": "prob", "desc": "Two-state transition probability",
    },
    "III.9.52": {
        "n_vars": 3, "tier": 2,
        "formula": "(p_dip * Ef * t / h) * sin(E_n * t / h)**2 / ((E_n * t / h)**2 + (p_dip * Ef * t / h)**2)",
        "variables": [
            {"name": "p_dip", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 0.1, "high": 2.0},
            {"name": "E_n", "low": 1.0, "high": 5.0},
            {"name": "t", "low": 0.1, "high": 2.0},
            {"name": "h", "low": 1.0, "high": 5.0},
        ],
        "output": "prob", "desc": "Perturbation transition probability",
        "n_vars": 5,
    },
    "III.10.19": {
        "n_vars": 3, "tier": 2,
        "formula": "mom * sqrt(Bx**2 + By**2 + Bz**2)",
        "variables": [
            {"name": "mom", "low": 1.0, "high": 5.0},
            {"name": "Bx", "low": 1.0, "high": 5.0},
            {"name": "By", "low": 1.0, "high": 5.0},
            {"name": "Bz", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Energy in magnetic field (|B|)",
        "n_vars": 4,
    },
    "III.13.18": {
        "n_vars": 3, "tier": 2,
        "formula": "2 * E_n * d_val**2 * k_val / (h**2)",
        "variables": [
            {"name": "E_n", "low": 1.0, "high": 5.0},
            {"name": "d_val", "low": 1.0, "high": 5.0},
            {"name": "k_val", "low": 1.0, "high": 5.0},
            {"name": "h", "low": 1.0, "high": 5.0},
        ],
        "output": "v", "desc": "QM velocity from dispersion",
        "n_vars": 4,
    },
    "III.14.14": {
        "n_vars": 2, "tier": 1,
        "formula": "I_0 * (exp(q * Volt / (kb * T)) - 1)",
        "variables": [
            {"name": "I_0", "low": 1.0, "high": 5.0},
            {"name": "q", "low": 0.1, "high": 1.0},
            {"name": "Volt", "low": 0.1, "high": 2.0},
            {"name": "kb", "low": 0.5, "high": 2.0},
            {"name": "T", "low": 1.0, "high": 10.0},
        ],
        "output": "I_diode", "desc": "Diode equation (Shockley)",
        "n_vars": 5,
    },
    "III.15.12": {
        "n_vars": 2, "tier": 2,
        "formula": "2 * U * (1 - cos(k_val * d_val))",
        "variables": [
            {"name": "U", "low": 1.0, "high": 5.0},
            {"name": "k_val", "low": 1.0, "high": 5.0},
            {"name": "d_val", "low": 1.0, "high": 5.0},
        ],
        "output": "E", "desc": "Tight-binding band energy",
        "n_vars": 3,
    },
    "III.15.14": {
        "n_vars": 3, "tier": 3,
        "formula": "h**2 / (2 * E_n * d_val**2) * (k_val * d_val)**2",
        "variables": [
            {"name": "h", "low": 1.0, "high": 5.0},
            {"name": "E_n", "low": 1.0, "high": 5.0},
            {"name": "d_val", "low": 1.0, "high": 5.0},
            {"name": "k_val", "low": 0.1, "high": 3.0},
        ],
        "output": "m_eff", "desc": "Effective mass (band theory)",
        "n_vars": 4,
    },
    "III.15.27": {
        "n_vars": 3, "tier": 3,
        "formula": "2 * pi * alpha_pol / (n_val * d_val) * Ef",
        "variables": [
            {"name": "alpha_pol", "low": 1.0, "high": 5.0},
            {"name": "n_val", "low": 1.0, "high": 5.0},
            {"name": "d_val", "low": 1.0, "high": 5.0},
            {"name": "Ef", "low": 1.0, "high": 5.0},
        ],
        "output": "P", "desc": "Scattering cross section",
        "n_vars": 4,
    },
    "III.17.37": {
        "n_vars": 3, "tier": 2,
        "formula": "beta_val * (1 + alpha_pol * cos(theta))",
        "variables": [
            {"name": "beta_val", "low": 1.0, "high": 5.0},
            {"name": "alpha_pol", "low": 0.1, "high": 2.0},
            {"name": "theta", "low": 0.0, "high": 6.28},
        ],
        "output": "f", "desc": "Angular distribution",
    },
    "III.19.51": {
        "n_vars": 3, "tier": 3,
        "formula": "-m * q**4 / (2 * h**2 * (4 * pi * epsilon)**2 * n_val**2)",
        "variables": [
            {"name": "m", "low": 1.0, "high": 5.0},
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "h", "low": 1.0, "high": 5.0},
            {"name": "epsilon", "low": 1.0, "high": 5.0},
            {"name": "n_val", "low": 1.0, "high": 5.0},
        ],
        "output": "E_n", "desc": "Hydrogen energy levels",
        "n_vars": 5,
    },
    "III.21.1": {
        "n_vars": 3, "tier": 2,
        "formula": "rho_ch * q * A_cs * v",
        "variables": [
            {"name": "rho_ch", "low": 1.0, "high": 5.0},
            {"name": "q", "low": 1.0, "high": 5.0},
            {"name": "A_cs", "low": 1.0, "high": 5.0},
            {"name": "v", "low": 1.0, "high": 5.0},
        ],
        "output": "I_curr", "desc": "Electric current",
        "n_vars": 4,
    },
}

# ---------------------------------------------------------------------------
# Ensure n_vars matches variable count for all entries
# ---------------------------------------------------------------------------
for _key, _spec in FEYNMAN_EQUATIONS.items():
    _spec["n_vars"] = len(_spec["variables"])


def get_equations_by_tier(tier=None):
    """Return equation IDs filtered by tier.

    Args:
        tier: int (1-4) to filter by tier, or None to return all.

    Returns:
        List of equation ID strings.
    """
    if tier is None:
        return list(FEYNMAN_EQUATIONS.keys())
    return [k for k, v in FEYNMAN_EQUATIONS.items() if v.get("tier") == tier]


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
    x_np, y_np = _generate_from_spec(eq_spec, n_samples, rng)

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
    num_workers: int = 2,
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, x_mean, x_std, y_mean, y_std
