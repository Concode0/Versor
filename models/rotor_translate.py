# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Direct rotor-to-formula translation.

Translates trained rotor parameters directly into symbolic expressions
without curve_fit. Each rotor R = exp(-B/2) encodes a rotation/boost
in a plane; the sandwich product x' = RxR~ gives a closed-form
transformation that can be read off as a formula term.

Pipeline:
  1. For each layer's rotor, average bivector across channels to get representative B
  2. Map each significant bivector component to its closed-form action:
     - Elliptic (B^2 < 0): cos(theta)*x + sin(theta)*y  (rotation in xy-plane)
     - Hyperbolic (B^2 > 0): cosh(theta)*x + sinh(theta)*y  (boost)
     - Parabolic (B^2 = 0): x + theta*y  (shear/translation)
  3. Compose actions across layers
  4. Read off input->output mapping as symbolic expression
"""
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import sympy
import torch

from core.algebra import CliffordAlgebra
from models.sr_net import SRGBN

logger = logging.getLogger(__name__)


@dataclass
class SimplePlane:
    var_i: int
    var_j: int
    sig_type: str  # "elliptic" | "hyperbolic" | "parabolic"
    angle: float


@dataclass
class RotorTerm:
    planes: List[SimplePlane] = field(default_factory=list)
    weight: float = 1.0
    expr: sympy.Expr = None
    fn: callable = None


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation, NaN-safe."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom < 1e-30:
        return 0.0
    return abs(float(np.dot(a_c, b_c) / denom))


def _log_corr_divide(residual: np.ndarray, fitted: np.ndarray):
    """Check log-domain correlation to decide if division is appropriate.

    Returns (divided_result_or_None, log_correlation).
    """
    eps = 1e-8
    abs_res = np.abs(residual) + eps
    abs_fit = np.abs(fitted) + eps

    log_corr = _correlation(np.log(abs_res), np.log(abs_fit))

    if log_corr < 0.7:
        return None, log_corr

    safe_fitted = np.where(np.abs(fitted) < eps, np.sign(fitted) * eps, fitted)
    safe_fitted = np.where(safe_fitted == 0, eps, safe_fitted)
    result = residual / safe_fitted

    if not np.all(np.isfinite(result)):
        return None, log_corr

    return result, log_corr


class RotorTranslator:
    def __init__(self, algebra: CliffordAlgebra):
        self.algebra = algebra
        self.symbols = [sympy.Symbol(f"x{i+1}") for i in range(algebra.n)]

    def translate(self, model: SRGBN) -> List[RotorTerm]:
        """Analyze model rotors and return symbolic terms.

        Bivectors are [C, n_bv] per layer (single RotorLayer).
        We average across channels to get a representative bivector per layer.
        Planes involving dimensions beyond model.in_features are filtered
        (those are embedding artifacts, not real data variables).
        """
        analysis = model.get_rotor_analysis()
        n = self.algebra.n
        dim = self.algebra.dim
        n_real = getattr(model, 'in_features', n)

        # Basis signature: e_i^2 = 1 (p), -1 (q), 0 (r)
        sig = [1.0] * self.algebra.p + [-1.0] * self.algebra.q + [0.0] * self.algebra.r

        # Grade-2 basis indices
        bivector_basis_indices = [i for i in range(dim) if bin(i).count("1") == 2]

        # Map each bivector index to (i, j) pair
        bivector_mappings = []
        for idx in bivector_basis_indices:
            bits = [pos for pos in range(n) if (idx >> pos) & 1]
            bivector_mappings.append(tuple(bits))

        terms = []
        for layer_info in analysis:
            bivectors = layer_info["bivectors"]  # [C, n_bv]

            # Average across channels to get representative bivector
            B_mean = bivectors.mean(dim=0).numpy()  # [n_bv]

            planes = []
            for b_idx, val in enumerate(B_mean):
                if abs(val) < 1e-6:
                    continue

                i, j = bivector_mappings[b_idx]

                # Skip planes where BOTH variables are phantom (beyond in_features)
                if i >= n_real and j >= n_real:
                    continue

                # Plane signature: (ei^ej)^2 = -(ei^2 * ej^2)
                sq = -(sig[i] * sig[j])

                if sq < -0.5:
                    sig_type = "elliptic"
                elif sq > 0.5:
                    sig_type = "hyperbolic"
                else:
                    sig_type = "parabolic"

                planes.append(SimplePlane(i, j, sig_type, angle=float(val)))

            if not planes:
                continue

            expr = self._compose_actions(planes)

            # Zero out phantom variables (beyond in_features)
            for k in range(n_real, n):
                expr = expr.subs(self.symbols[k], 0)
            expr = sympy.simplify(expr)

            if expr == sympy.Integer(0):
                continue

            fn = sympy.lambdify(self.symbols, expr, "numpy")

            terms.append(RotorTerm(
                planes=planes,
                weight=1.0,
                expr=expr,
                fn=fn,
            ))

        return terms

    def _plane_to_action(self, plane: SimplePlane) -> sympy.Expr:
        """Closed-form sandwich product action for a single plane."""
        xi = self.symbols[plane.var_i]
        xj = self.symbols[plane.var_j]
        theta = plane.angle

        if plane.sig_type == "elliptic":
            return xi * sympy.cos(2 * theta) - xj * sympy.sin(2 * theta)
        elif plane.sig_type == "hyperbolic":
            return xi * sympy.cosh(2 * theta) + xj * sympy.sinh(2 * theta)
        else:
            return xi + 2 * theta * xj

    def _compose_actions(self, planes: List[SimplePlane]) -> sympy.Expr:
        """Combine actions from multiple planes within a rotor."""
        combined = sympy.Integer(0)
        for p in planes:
            combined += self._plane_to_action(p)
        return combined

    def to_formula(self, terms: List[RotorTerm]) -> str:
        """Assemble final formula string."""
        if not terms:
            return "y = 0"

        final_expr = sympy.Integer(0)
        for t in terms:
            final_expr += t.weight * t.expr

        return f"y = {sympy.simplify(final_expr)}"

    def translate_implicit(self, model: SRGBN, target_var_idx: int) -> List[RotorTerm]:
        """Translate model rotors in augmented (k+1)-variable implicit space.

        Same as translate() but symbols include the target variable.
        Returns F(x1,...,xk,y) expression where target_var_idx marks y.

        Args:
            model: Trained SRGBN with in_features=k+1.
            target_var_idx: Index of the target variable in augmented space.

        Returns:
            List of RotorTerms in the implicit formulation.
        """
        # Extend symbols to include the target variable
        n_total = self.algebra.n
        all_symbols = [sympy.Symbol(f"x{i+1}") for i in range(n_total)]
        if target_var_idx < len(all_symbols):
            all_symbols[target_var_idx] = sympy.Symbol("y")

        # Save and temporarily replace symbols
        orig_symbols = self.symbols
        self.symbols = all_symbols

        terms = self.translate(model)

        # Restore
        self.symbols = orig_symbols
        return terms

    def evaluate_terms(self, terms: List[RotorTerm], X_np: np.ndarray) -> np.ndarray:
        """Evaluate extracted terms on data to get predictions."""
        y_hat = np.zeros(X_np.shape[0])
        n_vars = X_np.shape[1]
        n_syms = len(self.symbols)

        for t in terms:
            if t.fn is None:
                continue

            # Ensure we pass exactly n_syms arguments by padding with zeros if n_vars < n_syms
            args = []
            for i in range(n_syms):
                if i < n_vars:
                    args.append(X_np[:, i])
                else:
                    args.append(np.zeros(X_np.shape[0]))

            y_hat += t.weight * t.fn(*args)
        return y_hat
