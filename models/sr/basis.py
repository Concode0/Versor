# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Data-adaptive nonlinear basis expansion for symbolic regression.

Augments input features with nonlinear transforms (log, 1/x, sqrt, exp)
to let the downstream polynomial-vocabulary GBN discover transcendental
relationships like power laws, exponentials, and rational functions.

Key design decisions:
  - Each transform is only added if it's numerically safe (e.g., log only
    for positive columns, exp only when max(|x|) <= 700).
  - Transforms are filtered by |corr(transform, y)| > threshold to avoid
    adding noise features.
  - Total expansion is capped at max_expansion_factor * n_original.
  - Optionally log-transforms y when it has high dynamic range and is
    all-positive, which linearizes power-law targets.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import sympy

logger = logging.getLogger(__name__)


@dataclass
class BasisExpansionResult:
    """Result of basis expansion analysis."""

    X_expanded: np.ndarray          # [N, n_expanded] expanded feature matrix
    var_name_map: dict              # col_index -> display name (e.g. "log(D)")
    var_expr_map: dict              # col_index -> sympy.Expr in original variables
    log_target: bool                # whether y was log-transformed
    n_original: int                 # original variable count
    original_var_names: list        # original variable names


def _abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation, NaN-safe."""
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    denom = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
    if denom < 1e-30:
        return 0.0
    r = float(np.dot(a_c, b_c) / denom)
    return abs(r) if np.isfinite(r) else 0.0


class BasisExpander:
    """Data-adaptive nonlinear basis expansion for SR.

    Analyzes raw data properties and selects numerically safe nonlinear
    transforms that correlate with the target. Produces an expanded
    feature matrix with a mapping from column indices to sympy expressions
    in the original variables.

    Args:
        enable_log: Include log(xi) for positive high-dynamic-range variables.
        enable_reciprocal: Include 1/xi for nonzero variables.
        enable_sqrt: Include sqrt(xi) for non-negative variables.
        enable_exp: Include exp(xi) when max(|xi|) is safe.
        log_target_auto: Auto log-transform y if positive and high range.
        corr_threshold: Minimum |corr(transform, y)| to keep a transform.
        max_expansion_factor: Cap total features at this * n_original.
        dynamic_range_threshold: min max/min ratio to trigger log transform.
        log_target_dynamic_range: min max/min ratio for y log-transform.
        exp_max_input: Maximum |xi| value for computing exp(xi) safely.
    """

    def __init__(
        self,
        enable_log=True,
        enable_reciprocal=True,
        enable_sqrt=True,
        enable_exp=True,
        log_target_auto=True,
        corr_threshold=0.05,
        max_expansion_factor=3,
        dynamic_range_threshold=100.0,
        log_target_dynamic_range=50.0,
        exp_max_input=700.0,
    ):
        self.enable_log = enable_log
        self.enable_reciprocal = enable_reciprocal
        self.enable_sqrt = enable_sqrt
        self.enable_exp = enable_exp
        self.log_target_auto = log_target_auto
        self.corr_threshold = corr_threshold
        self.max_expansion_factor = max_expansion_factor
        self.dynamic_range_threshold = dynamic_range_threshold
        self.log_target_dynamic_range = log_target_dynamic_range
        self.exp_max_input = exp_max_input

    def analyze_and_expand(self, X_raw, y_raw, var_names):
        """Analyze data properties, select transforms, expand features.

        Args:
            X_raw: np.ndarray [N, k] raw (unnormalized) input features.
            y_raw: np.ndarray [N] raw target values.
            var_names: List of k variable name strings.

        Returns:
            BasisExpansionResult with expanded features and mappings.
        """
        N, k = X_raw.shape
        y = y_raw.copy()
        var_names = list(var_names)

        # Build sympy symbols for original variables
        orig_syms = [sympy.Symbol(name) for name in var_names]

        # Check if y should be log-transformed
        log_target = False
        if self.log_target_auto:
            log_target = self._should_log_target(y)
            if log_target:
                y = np.log(np.abs(y) + 1e-30)
                logger.info("BasisExpander: log-transforming target (high dynamic range)")

        # Collect candidate transforms: (column_data, display_name, sympy_expr, corr)
        candidates = []

        for i in range(k):
            xi = X_raw[:, i]
            sym_i = orig_syms[i]

            # log(xi): positive and high dynamic range
            if self.enable_log:
                if np.all(xi > 0):
                    ratio = np.max(xi) / (np.min(xi) + 1e-30)
                    if ratio > self.dynamic_range_threshold:
                        log_xi = np.log(xi)
                        corr = _abs_correlation(log_xi, y)
                        if corr > self.corr_threshold:
                            candidates.append((
                                log_xi,
                                f"log({var_names[i]})",
                                sympy.log(sym_i),
                                corr,
                            ))

            # 1/xi: nonzero and reasonable range
            if self.enable_reciprocal:
                if np.all(np.abs(xi) > 1e-10):
                    ratio = np.max(np.abs(xi)) / np.min(np.abs(xi))
                    if ratio > 10:
                        recip_xi = 1.0 / xi
                        corr = _abs_correlation(recip_xi, y)
                        if corr > self.corr_threshold:
                            candidates.append((
                                recip_xi,
                                f"1/({var_names[i]})",
                                1 / sym_i,
                                corr,
                            ))

            # sqrt(xi): non-negative
            if self.enable_sqrt:
                if np.all(xi >= 0):
                    sqrt_xi = np.sqrt(xi)
                    corr = _abs_correlation(sqrt_xi, y)
                    if corr > self.corr_threshold:
                        candidates.append((
                            sqrt_xi,
                            f"sqrt({var_names[i]})",
                            sympy.sqrt(sym_i),
                            corr,
                        ))

            # exp(xi): only if safe (max(|xi|) <= exp_max_input)
            if self.enable_exp:
                if np.max(np.abs(xi)) <= self.exp_max_input:
                    exp_xi = np.exp(xi)
                    if np.all(np.isfinite(exp_xi)):
                        corr = _abs_correlation(exp_xi, y)
                        if corr > self.corr_threshold:
                            candidates.append((
                                exp_xi,
                                f"exp({var_names[i]})",
                                sympy.exp(sym_i),
                                corr,
                            ))

        # Sort by correlation (descending) and cap total features
        candidates.sort(key=lambda c: c[3], reverse=True)
        max_new = self.max_expansion_factor * k - k
        candidates = candidates[:max(max_new, 0)]

        # Build expanded matrix
        columns = [X_raw[:, i] for i in range(k)]
        name_map = {i: var_names[i] for i in range(k)}
        expr_map = {i: orig_syms[i] for i in range(k)}

        for data, name, expr, corr in candidates:
            col_idx = len(columns)
            columns.append(data)
            name_map[col_idx] = name
            expr_map[col_idx] = expr
            logger.info(f"BasisExpander: added {name} (|corr|={corr:.3f})")

        X_expanded = np.column_stack(columns) if columns else X_raw

        return BasisExpansionResult(
            X_expanded=X_expanded,
            var_name_map=name_map,
            var_expr_map=expr_map,
            log_target=log_target,
            n_original=k,
            original_var_names=var_names,
        )

    def _should_log_target(self, y):
        """Check if y should be log-transformed."""
        if not np.all(y > 0):
            return False
        ratio = np.max(y) / (np.min(y) + 1e-30)
        return ratio > self.log_target_dynamic_range
