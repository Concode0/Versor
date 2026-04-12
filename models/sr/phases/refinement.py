# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Phase 3: SymPy refinement mixin.

Contains joint lstsq reweighting, formula assembly, and rotor
plane inspection.
"""

import logging

import numpy as np
import sympy

from models.sr.translator import RotorTerm
from models.sr.utils import safe_sympy_solve, make_lambdify_fn

logger = logging.getLogger(__name__)


class RefinementMixin:
    """Phase 3 methods: refinement, formula assembly, rotor planes."""

    def _refine_all_terms(self, all_terms, X_orig, y_orig, all_ops=None):
        """Phase 3: Joint lstsq reweighting of all extracted terms.

        For pure additive composition (all ops are "sub"), performs joint
        lstsq to find optimal weights + intercept.

        For mixed composition (has "div" ops), evaluates the compositional
        formula and fits a global scale + offset via lstsq.

        Returns:
            (refined_terms, ops)
        """
        if all_ops is None:
            all_ops = ["sub"] * len(all_terms)

        if not all_terms:
            return all_terms, all_ops

        # Joint lstsq reweighting (all additive)
        return self._refine_additive(all_terms, X_orig, y_orig, all_ops)

    def _refine_additive(self, all_terms, X_orig, y_orig, all_ops):
        """Joint lstsq reweighting for pure additive terms."""
        n_vars = X_orig.shape[1]

        term_preds = []
        for t in all_terms:
            if t.fn is not None:
                n_expected = t.fn.__code__.co_argcount
                args = [X_orig[:, i] for i in range(min(n_vars, n_expected))]
                if len(args) < n_expected:
                    args.extend([np.zeros(X_orig.shape[0])
                                 for _ in range(n_expected - len(args))])
                pred = t.fn(*args)
                pred = np.broadcast_to(np.asarray(pred, dtype=np.float64),
                                       (X_orig.shape[0],)).copy()
                if np.all(np.isfinite(pred)):
                    term_preds.append(pred)
                else:
                    term_preds.append(np.zeros(X_orig.shape[0]))
            else:
                term_preds.append(np.zeros(X_orig.shape[0]))

        if not term_preds:
            return all_terms, all_ops

        # Include intercept column for constant offset fitting
        from models.sr.numerics import safe_lstsq
        A = np.column_stack(term_preds + [np.ones(X_orig.shape[0])])
        new_weights = safe_lstsq(A, y_orig)

        refined = []
        for i, t in enumerate(all_terms):
            refined.append(RotorTerm(
                planes=t.planes,
                weight=float(new_weights[i]),
                expr=t.expr,
                fn=t.fn,
            ))

        # Add intercept term if significant
        intercept = float(new_weights[-1])
        if abs(intercept) > 1e-8:
            n_vars = X_orig.shape[1]
            intercept_syms = [sympy.Symbol(f"x{i+1}") for i in range(n_vars)]
            intercept_expr = sympy.Float(intercept)
            refined.append(RotorTerm(
                planes=[],
                weight=1.0,
                expr=intercept_expr,
                fn=make_lambdify_fn(intercept_syms, intercept_expr),
            ))
            all_ops = list(all_ops) + ["sub"]

        return refined, all_ops

    def _assemble_formula(self, all_terms, var_names, all_ops=None,
                          implicit_form=None):
        """Build simplified formula string from extracted terms."""
        if not all_terms:
            return "y = 0"
        if all_ops is None:
            all_ops = ["sub"] * len(all_terms)

        n_vars = len(var_names)
        symbols = [sympy.Symbol(f"x{i+1}") for i in range(n_vars)]
        subs = {symbols[i]: sympy.Symbol(var_names[i]) for i in range(n_vars)}

        result_expr = sympy.Integer(0)
        for t in all_terms:
            if t.expr is not None:
                result_expr += t.weight * t.expr

        result_expr = result_expr.subs(subs)

        # Implicit mode: try sympy.solve for explicit form
        if implicit_form is not None and implicit_form.mode == "implicit":
            y_sym = sympy.Symbol("y")
            if y_sym in result_expr.free_symbols:
                sol = safe_sympy_solve(result_expr, y_sym)
                if sol is not None:
                    return f"y = {sympy.simplify(sol)}"
                return f"F({', '.join(var_names)}, y) = {sympy.simplify(result_expr)}"

        return f"y = {sympy.simplify(result_expr)}"

    def _get_active_rotor_planes(self, model):
        """Get list of dominant plane names from active rotors."""
        analysis = model.get_rotor_analysis()
        return [info["dominant_plane"] for info in analysis]
