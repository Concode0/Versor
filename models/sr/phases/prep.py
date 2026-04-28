# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Phase 0: Data preparation mixin.

Contains data preparation, linearity checking, basis expansion,
and single-group fallback logic.
"""

import logging

import numpy as np
import sympy
import torch

from core.algebra import CliffordAlgebra
from models.sr.translator import RotorTerm
from models.sr.utils import (
    safe_sympy_solve,
    standardize,
    subsample,
    safe_svd,
    make_lambdify_fn,
)

logger = logging.getLogger(__name__)


class PrepMixin:
    """Phase 0 methods: data preparation, linearity check, basis expansion."""

    def _prepare_data(self, X_orig, y_orig, X_norm, y_norm, var_names):
        """SVD alignment, variable grouping (with relationship graph),
        implicit mode probe with geometric criteria."""
        from models.sr.pipeline import _PrepResult

        # SVD align
        X_c = X_orig - X_orig.mean(axis=0)
        S, Vt = safe_svd(X_c)
        if S is None:
            S, Vt = np.ones(X_orig.shape[1]), np.eye(X_orig.shape[1])

        # Variable grouping — now returns (groups, relationship_graph)
        rel_graph = None
        if self.grouping_enabled:
            from models.sr.grouper import VariableGrouper

            gcfg = self.grouping_config
            grouper = VariableGrouper(
                max_groups=self.max_groups,
                device=self.device,
                sample_size=gcfg.get("sample_size", 500),
                commutator_weight=gcfg.get("commutator_weight", 0.4),
                coherence_weight=gcfg.get("coherence_weight", 0.3),
                spectral_weight=gcfg.get("spectral_weight", 0.3),
            )
            groups, rel_graph = grouper.group(X_orig, y_orig, var_names)
        else:
            groups = [self._single_group_fallback(X_orig, y_orig, var_names, Vt)]

        # Build geometric report from relationship graph for implicit solver
        geo_report = None
        if rel_graph is not None:
            geo_report = rel_graph.geometric_report()
            geo_report["ambient_dim"] = X_orig.shape[1] + 1  # +1 for target y

        # Implicit mode probe (now with geometric criteria)
        implicit_form = None
        if self.implicit_mode != 'explicit' and len(groups) == 1:
            try:
                from models.sr.implicit import ImplicitSolver

                solver = ImplicitSolver(
                    device=self.device,
                    probe_epochs=self.probe_config.get("probe_epochs", 30),
                    jacobian_weight=self.probe_config.get("jacobian_weight", 0.1),
                )
                algebra = groups[0].algebra
                probe_data = torch.cat([X_norm, y_norm], dim=-1)
                probe_data = subsample(probe_data, 500)
                X_probe = probe_data[:, :-1]
                y_probe = probe_data[:, -1:]
                solver_result = solver.probe_best_mode(
                    algebra,
                    X_probe,
                    y_probe,
                    geometric_report=geo_report,
                )
                if self.implicit_mode == 'auto':
                    implicit_form = solver_result
                elif self.implicit_mode == 'implicit':
                    solver_result.mode = 'implicit'
                    implicit_form = solver_result
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Implicit probe failed: {e}")

        return _PrepResult(
            groups=groups,
            relationship_graph=rel_graph,
            implicit_form=implicit_form,
            svd_Vt=Vt,
            svd_S=S,
        )

    def _single_group_fallback(self, X_orig, y_orig, var_names, Vt):
        """Create a single VariableGroup without importing VariableGrouper."""
        from models.sr.grouper import VariableGroup

        n_vars = X_orig.shape[1]
        # Quick MetricSearch
        combined = np.column_stack([X_orig, y_orig.reshape(-1, 1)])
        data = torch.tensor(combined, dtype=torch.float32, device=self.device)
        data = standardize(data)
        data = subsample(data, 500)
        if data.shape[1] > 6:
            data_c = data - data.mean(0)
            _, _, V = torch.linalg.svd(data_c, full_matrices=False)
            data = data_c @ V[:6].T

        from models.sr.utils import safe_metric_search

        p, q, r = safe_metric_search(
            data,
            self.device,
            n_vars,
            max_p=max(n_vars, 2),
        )

        algebra = CliffordAlgebra(p, q, r, device=self.device)
        return VariableGroup(
            var_indices=list(range(n_vars)),
            var_names=var_names or [f"x{i + 1}" for i in range(n_vars)],
            signature=(p, q, r),
            algebra=algebra,
            svd_Vt=Vt,
        )

    def _check_linearity(self, X_raw, y_raw, r2_threshold=0.90):
        """Multi-branch linearity check with BIC parsimony and power-law action.

        Three branches:
          1. Standard linear fit
          2. Log-log power law (builds explicit power-law term if R2 >= 0.90)
          3. BIC comparison: linear vs quadratic -- prefer simpler form

        Returns (is_linear_or_powerlaw, terms, r2) where terms are list[RotorTerm].
        """
        N, d = X_raw.shape
        symbols = [sympy.Symbol(f"x{i + 1}") for i in range(d)]

        # Branch 1: Standard linear fit
        A_lin = np.column_stack([X_raw, np.ones(N)])
        coeffs_lin = np.linalg.lstsq(A_lin, y_raw, rcond=None)[0]
        y_hat_lin = A_lin @ coeffs_lin
        ss_res_lin = np.sum((y_raw - y_hat_lin) ** 2)
        ss_tot = np.sum((y_raw - y_raw.mean()) ** 2) + 1e-12
        r2_lin = 1.0 - ss_res_lin / ss_tot

        # Branch 2: Log-log fit (power law detection + action)
        from models.sr.numerics import safe_log, safe_exp

        eps = 1e-8
        pos_mask = np.all(np.abs(X_raw) > eps, axis=1) & (np.abs(y_raw) > eps)
        r2_log = -1.0
        powerlaw_term = None

        if pos_mask.sum() > max(10, int(N * 0.8)):
            X_pos = np.abs(X_raw[pos_mask])
            y_pos = np.abs(y_raw[pos_mask])
            log_X = safe_log(X_pos)
            log_y = safe_log(y_pos)
            A_log = np.column_stack([log_X, np.ones(pos_mask.sum())])
            coeffs_log = np.linalg.lstsq(A_log, log_y, rcond=None)[0]
            log_y_hat = A_log @ coeffs_log
            ss_res_log = np.sum((log_y - log_y_hat) ** 2)
            ss_tot_log = np.sum((log_y - log_y.mean()) ** 2) + 1e-12
            r2_log = 1.0 - ss_res_log / ss_tot_log

            exponents = coeffs_log[:d]
            log_intercept = coeffs_log[d]
            has_nonunit = any(abs(e - 1.0) > 0.15 and abs(e) > 0.15 for e in exponents)

            # Action: if power law fits well, build explicit term
            if r2_log >= 0.90 and has_nonunit:
                # Round exponents to nearest simple fraction
                rounded_exp = [self._round_exponent(e) for e in exponents]
                # y = exp(intercept) * prod(xi^alpha_i)
                # Determine sign of y from original data
                y_sign = np.sign(np.median(y_raw[pos_mask]))
                scale = float(safe_exp(log_intercept))
                if y_sign < 0:
                    scale = -scale

                expr = sympy.Float(scale)
                for i in range(d):
                    if abs(rounded_exp[i]) > 1e-3:
                        expr = expr * symbols[i] ** sympy.Rational(rounded_exp[i]).limit_denominator(6)

                # Evaluate on full data to compute R2
                fn = make_lambdify_fn(symbols, expr)
                from models.sr.utils import safe_evaluate_term

                y_hat_pl = safe_evaluate_term(fn, X_raw)
                if y_hat_pl is not None:
                    ss_res_pl = np.sum((y_raw - y_hat_pl) ** 2)
                    r2_pl = 1.0 - ss_res_pl / ss_tot
                    powerlaw_term = RotorTerm(
                        planes=[],
                        weight=1.0,
                        expr=expr,
                        fn=fn,
                    )
                    logger.info(
                        f"Power law short-circuit: exponents={rounded_exp}, R2={r2_pl:.4f} (log-space R2={r2_log:.4f})"
                    )
                    # If power law is great, return immediately
                    if r2_pl >= r2_threshold:
                        return True, [powerlaw_term], r2_pl

            # If power law fits better in log-space but exponents differ from 1,
            # reject linear classification
            if has_nonunit and r2_log > r2_lin:
                logger.debug(
                    f"Power law detected (exponents={exponents.tolist()}, "
                    f"r2_log={r2_log:.4f} > r2_lin={r2_lin:.4f}), "
                    f"rejecting linear classification"
                )
                # Return power law term if we built one, else continue to unbending
                if powerlaw_term is not None:
                    return True, [powerlaw_term], r2_log
                return False, [], r2_lin

        # Branch 3: BIC parsimony -- linear vs quadratic
        if r2_lin >= r2_threshold:
            # Build quadratic features for BIC comparison
            if d <= 6:
                bic_prefers_linear = self._bic_prefers_simpler(
                    X_raw,
                    y_raw,
                    coeffs_lin,
                    ss_res_lin,
                )
            else:
                bic_prefers_linear = True  # skip BIC for high-d

            if bic_prefers_linear:
                # Build linear terms
                lin_terms = []
                for i in range(d):
                    w = float(coeffs_lin[i])
                    if abs(w) > 1e-12:
                        expr_i = sympy.Float(w) * symbols[i]
                        lin_terms.append(
                            RotorTerm(
                                planes=[],
                                weight=1.0,
                                expr=expr_i,
                                fn=make_lambdify_fn(symbols, expr_i),
                            )
                        )
                intercept = float(coeffs_lin[d])
                if abs(intercept) > 1e-8:
                    expr_c = sympy.Float(intercept)
                    lin_terms.append(
                        RotorTerm(
                            planes=[],
                            weight=1.0,
                            expr=expr_c,
                            fn=make_lambdify_fn(symbols, expr_c),
                        )
                    )
                return True, lin_terms, r2_lin

        return False, [], r2_lin

    @staticmethod
    def _round_exponent(e, tol=0.15):
        """Round an exponent to the nearest 'nice' value."""
        nice_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, -0.5, -1, -1.5, -2, -3]
        for nv in nice_values:
            if abs(e - nv) < tol:
                return nv
        return round(e * 2) / 2  # nearest half-integer

    def _bic_prefers_simpler(self, X_raw, y_raw, coeffs_lin, ss_res_lin):
        """BIC comparison: linear vs quadratic model.

        Returns True if the linear model is preferred (simpler).
        """
        N, d = X_raw.shape
        k_lin = d + 1  # d slopes + intercept

        bic_lin = N * np.log(ss_res_lin / N + 1e-30) + k_lin * np.log(N)

        quad_features = []
        for i in range(d):
            for j in range(i, d):
                quad_features.append(X_raw[:, i] * X_raw[:, j])
        A_quad = np.column_stack([X_raw] + quad_features + [np.ones(N)])
        k_quad = A_quad.shape[1]

        try:
            coeffs_quad = np.linalg.lstsq(A_quad, y_raw, rcond=None)[0]
            y_hat_quad = A_quad @ coeffs_quad
            ss_res_quad = np.sum((y_raw - y_hat_quad) ** 2)
            bic_quad = N * np.log(ss_res_quad / N + 1e-30) + k_quad * np.log(N)
        except (np.linalg.LinAlgError, ValueError):
            return True  # Can't fit quadratic, prefer linear

        # Prefer simpler (linear) unless quadratic BIC is substantially better
        # DELTA_BIC < 6 means "not strong evidence" for the complex model
        return (bic_lin - bic_quad) < 6

    def _apply_basis_expansion(self, X_orig, y_orig, var_names):
        """Run BasisExpander on raw data, updating X_orig and var_names.

        Returns:
            (X_expanded, y_transformed, expanded_var_names)
        """
        from models.sr.basis import BasisExpander

        cfg = self.basis_config
        expander = BasisExpander(
            enable_log=cfg.get("log", True),
            enable_reciprocal=cfg.get("reciprocal", True),
            enable_sqrt=cfg.get("sqrt", True),
            enable_exp=cfg.get("exp", True),
            log_target_auto=cfg.get("log_target_auto", True),
            corr_threshold=cfg.get("corr_threshold", 0.05),
            max_expansion_factor=cfg.get("max_expansion_factor", 3),
            dynamic_range_threshold=cfg.get("dynamic_range_threshold", 100.0),
            exp_max_input=cfg.get("exp_max_input", 700.0),
        )
        result = expander.analyze_and_expand(X_orig, y_orig, var_names)
        self._basis_result = result

        y_out = y_orig
        if result.log_target:
            y_out = np.log(np.abs(y_orig) + 1e-30)

        # Build expanded var_names from name_map
        n_expanded = result.X_expanded.shape[1]
        expanded_names = [result.var_name_map.get(i, f"z{i + 1}") for i in range(n_expanded)]

        logger.info(f"BasisExpander: {result.n_original} -> {n_expanded} features, log_target={result.log_target}")
        return result.X_expanded, y_out, expanded_names

    @staticmethod
    def _wrap_log_target(formula):
        """Wrap formula in exp() when target was log-transformed."""
        if formula.startswith("y = "):
            inner = formula[4:]
            return f"y = exp({inner})"
        return formula
