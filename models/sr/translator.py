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

    def translate(self, model) -> List[RotorTerm]:
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

            _numpy_mod = {"log": np.log, "sqrt": np.sqrt, "Abs": np.abs,
                         "sign": np.sign, "exp": np.exp}
            fn = sympy.lambdify(self.symbols, expr,
                                modules=[_numpy_mod, "numpy"])

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

    def translate_implicit(self, model, target_var_idx: int) -> List[RotorTerm]:
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

    # ------------------------------------------------------------------
    # Direct symbolic expansion (translate_direct)
    # ------------------------------------------------------------------

    def _symbolic_gp(self, mv_a: dict, mv_b: dict) -> dict:
        """Compute symbolic geometric product using the Cayley table.

        Args:
            mv_a: {blade_idx: sympy_expr} for left operand.
            mv_b: {blade_idx: sympy_expr} for right operand.

        Returns:
            dict: {blade_idx: sympy_expr} result.
        """
        cayley = self.algebra.cayley_indices.cpu().numpy()  # [D, D]
        signs = self.algebra.cayley_signs.cpu().numpy()      # [D, D]

        result = {}
        for i, expr_a in mv_a.items():
            if expr_a == 0:
                continue
            for j, expr_b in mv_b.items():
                if expr_b == 0:
                    continue
                k = int(cayley[i, j])
                s = float(signs[i, j])
                if s == 0:
                    continue
                term = s * expr_a * expr_b
                result[k] = result.get(k, sympy.Integer(0)) + term
        return result

    def _symbolic_sandwich(self, sym_mv: dict, B_weights: torch.Tensor) -> dict:
        """Compute R * M * R~ symbolically. R = exp(-B/2).

        R is numeric (trained constants), M is symbolic.

        Args:
            sym_mv: {blade_idx: sympy_expr} symbolic multivector.
            B_weights: 1-D tensor of bivector component weights.

        Returns:
            dict: {blade_idx: sympy_expr} after sandwich product.
        """
        algebra = self.algebra
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(1, 1, algebra.dim)
        for idx, bv_idx in enumerate(bv_indices):
            if idx < len(B_weights):
                B[0, 0, bv_idx] = -B_weights[idx] / 2.0

        R = algebra.exp(B)            # [1, 1, dim]
        R_rev = algebra.reverse(R)    # [1, 1, dim]

        r_np = R[0, 0].cpu().numpy()
        r_rev_np = R_rev[0, 0].cpu().numpy()

        R_sym = {i: sympy.Float(float(r_np[i]))
                 for i in range(len(r_np)) if abs(r_np[i]) > 1e-10}
        R_rev_sym = {i: sympy.Float(float(r_rev_np[i]))
                     for i in range(len(r_rev_np)) if abs(r_rev_np[i]) > 1e-10}

        temp = self._symbolic_gp(R_sym, sym_mv)
        return self._symbolic_gp(temp, R_rev_sym)

    def _build_symbolic_input(self, model) -> list:
        """Build per-channel symbolic multivectors from embedding weights.

        Returns:
            list of dicts: [{blade_idx: sympy_expr}, ...] one per channel.
        """
        emb = model.embedding
        k = emb.in_features
        n = self.algebra.n
        dim = self.algebra.dim
        C = emb.channels

        x_syms = [sympy.Symbol(f"x{i+1}") for i in range(k)]

        # Grade-1 projection weights: [C*n_g1, k]
        W1 = emb.grade1_proj.weight.detach().cpu().numpy()
        n_g1 = emb.n_g1
        g1_idx = emb.g1_idx.cpu().numpy()

        # Grade-2 LUT projection weights: [C*n_g2, n_lut_inputs]
        has_lut = hasattr(emb, 'lut_proj')
        if has_lut:
            W2 = emb.lut_proj.weight.detach().cpu().numpy()
            n_g2 = emb.n_g2
            g2_idx = emb.g2_idx.cpu().numpy()
            n_lut_inputs = emb.n_lut_inputs

            # Build symbolic LUT inputs using auxiliary symbols for clean lambdify.
            # log_xi = log(|xi|+eps), inv_xi = sign(xi)/(|xi|+eps)
            # These are treated as named functions in the expression.
            lut_syms = []
            for i in range(k):
                lut_syms.append(sympy.Symbol(f"log_x{i+1}"))
                lut_syms.append(sympy.Symbol(f"inv_x{i+1}"))
            lut_syms = lut_syms[:n_lut_inputs]

        # Grade-0 bias
        g0_bias = emb.grade0_bias.detach().cpu().numpy()  # [C]

        channels = []
        for c in range(C):
            mv = {}

            # Grade-0: bias
            mv[0] = sympy.Float(float(g0_bias[c]))

            # Grade-1: W1[c*n_g1:(c+1)*n_g1, :] @ x_syms
            for g_pos in range(n_g1):
                row = c * n_g1 + g_pos
                blade = int(g1_idx[g_pos])
                expr = sympy.Integer(0)
                for i in range(k):
                    w = float(W1[row, i])
                    if abs(w) > 1e-10:
                        expr += w * x_syms[i]
                if expr != 0:
                    mv[blade] = expr

            # Grade-2: LUT
            if has_lut:
                for g_pos in range(n_g2):
                    row = c * n_g2 + g_pos
                    blade = int(g2_idx[g_pos])
                    expr = sympy.Integer(0)
                    for i in range(n_lut_inputs):
                        w = float(W2[row, i])
                        if abs(w) > 1e-10:
                            expr += w * lut_syms[i]
                    if expr != 0:
                        mv[blade] = mv.get(blade, sympy.Integer(0)) + expr

            channels.append(mv)

        return channels

    def _prune_expr(self, expr: sympy.Expr, tol: float = 1e-4) -> sympy.Expr:
        """Zero out small numeric coefficients in a SymPy expression."""
        if expr.is_Number:
            return expr if abs(float(expr)) > tol else sympy.Integer(0)

        # Collect terms and prune
        terms = sympy.Add.make_args(expr)
        max_coeff = 0
        for t in terms:
            c = abs(complex(t.as_coeff_Mul()[0]))
            if c > max_coeff:
                max_coeff = c

        if max_coeff == 0:
            return sympy.Integer(0)

        threshold = tol * max_coeff
        pruned = sympy.Integer(0)
        for t in terms:
            c = abs(complex(t.as_coeff_Mul()[0]))
            if c >= threshold:
                pruned += t
        return pruned

    def _prune_mv(self, mv: dict, tol: float = 1e-4) -> dict:
        """Prune small numeric coefficients from symbolic multivector.

        After each symbolic operation (GP, sandwich, linear, blade),
        this bounds expression tree growth by removing terms whose
        coefficients are below tol * max_coefficient.
        """
        if not mv:
            return {}

        # Find max coefficient magnitude across all blades
        max_coeff = 0
        for expr in mv.values():
            if expr == 0:
                continue
            if expr.is_Number:
                max_coeff = max(max_coeff, abs(float(expr)))
            else:
                for term in sympy.Add.make_args(expr):
                    try:
                        c = abs(complex(term.as_coeff_Mul()[0]))
                        max_coeff = max(max_coeff, c)
                    except (TypeError, ValueError):
                        max_coeff = max(max_coeff, 1.0)

        if max_coeff == 0:
            return {}

        threshold = tol * max_coeff
        pruned = {}
        for blade_idx, expr in mv.items():
            pruned_expr = self._prune_expr(expr, tol=threshold / max(max_coeff, 1e-30))
            if pruned_expr != 0:
                pruned[blade_idx] = pruned_expr
        return pruned

    def _symbolic_linear(self, channel_mvs: list, linear) -> list:
        """Apply CliffordLinear channel mixing symbolically.

        For each output channel, combines input channels weighted
        by the linear weight matrix, plus bias.
        """
        W = linear.weight.detach().cpu().numpy()  # [out, in]
        bias = linear.bias.detach().cpu().numpy()  # [out, dim]
        out_channels = W.shape[0]
        in_channels = W.shape[1]

        new_mvs = []
        for c_out in range(out_channels):
            mv = {}
            for c_in in range(min(in_channels, len(channel_mvs))):
                w = float(W[c_out, c_in])
                if abs(w) < 1e-10:
                    continue
                for blade_idx, expr in channel_mvs[c_in].items():
                    mv[blade_idx] = mv.get(blade_idx, sympy.Integer(0)) + w * expr
            # Add bias
            for blade_idx in range(self.algebra.dim):
                b = float(bias[c_out, blade_idx])
                if abs(b) > 1e-10:
                    mv[blade_idx] = mv.get(blade_idx, sympy.Integer(0)) + b
            new_mvs.append(mv)
        return new_mvs

    def _symbolic_blade(self, channel_mvs: list, blade_selector) -> list:
        """Apply BladeSelector sigmoid gating symbolically.

        Each blade component is multiplied by sigmoid(weight) gate.
        """
        raw_w = blade_selector.weights.detach().cpu()  # [C, dim]
        sig_w = torch.sigmoid(raw_w).numpy()
        new_mvs = []
        for c, mv in enumerate(channel_mvs):
            new_mv = {}
            for blade_idx, expr in mv.items():
                w = float(sig_w[min(c, sig_w.shape[0] - 1), blade_idx])
                if abs(w) > 1e-10:
                    new_mv[blade_idx] = w * expr
            new_mvs.append(new_mv)
        return new_mvs

    def translate_direct(self, model, X_sample=None) -> List[RotorTerm]:
        """Direct symbolic extraction through the full model structure.

        Traces the full block flow per channel:
          embedding → [norm → linear → activation → rotor → blade → (+skip)] × N
          → output_blade → output_linear → grade-0

        Norm (recover=False) is pure direction normalization — a constant scaling
        factor absorbed by downstream linear weights. Skipped in symbolic trace.

        Intermediate pruning after each operation prevents memory explosion
        from cascading geometric products.

        Args:
            model: Trained SRGBN (typically single-rotor from unbender).
            X_sample: Optional data sample (unused, kept for API compat).

        Returns:
            List[RotorTerm] with symbolic expressions.
        """
        from functional.activation import GeometricSquare

        k = model.in_features
        n = self.algebra.n
        dim = self.algebra.dim
        C = model.channels
        x_syms = [sympy.Symbol(f"x{i+1}") for i in range(k)]

        # 1. Build symbolic embedding per channel
        channel_mvs = self._build_symbolic_input(model)

        # 2. For each block, apply full flow: linear → activation → rotor → blade → skip
        for block_idx, block in enumerate(model.blocks):
            # Save pre-block state for skip connection
            pre_block_mvs = channel_mvs

            # 2a. Norm: skipped in symbolic trace (recover=False makes it
            #     pure direction normalization = constant scaling absorbed
            #     by the downstream linear weights)

            # 2b. Linear: channel mixing
            channel_mvs = self._symbolic_linear(channel_mvs, block.linear)
            # Prune after linear to prevent blowup
            for c in range(len(channel_mvs)):
                channel_mvs[c] = self._prune_mv(channel_mvs[c])

            # 2c. Activation (GeometricSquare GP)
            new_channel_mvs = []
            for c in range(C):
                sym_mv = channel_mvs[c]

                if isinstance(block.activation, GeometricSquare):
                    gp_sq = self._symbolic_gp(sym_mv, sym_mv)
                    # Prune GP result immediately
                    gp_sq = self._prune_mv(gp_sq)
                    gate_val = float(torch.sigmoid(
                        block.activation.gate_logit[c]
                    ).item())
                    activated = {}
                    all_keys = set(sym_mv.keys()) | set(gp_sq.keys())
                    for idx in all_keys:
                        orig = sym_mv.get(idx, sympy.Integer(0))
                        sq = gp_sq.get(idx, sympy.Integer(0))
                        activated[idx] = orig + gate_val * sq
                    sym_mv = activated

                new_channel_mvs.append(sym_mv)
            channel_mvs = new_channel_mvs
            # Prune after activation
            for c in range(len(channel_mvs)):
                channel_mvs[c] = self._prune_mv(channel_mvs[c])

            # 2d. Rotor sandwich R * M * R~
            new_channel_mvs = []
            for c in range(C):
                bv_weights = block.rotor.bivector_weights.detach().cpu()
                B_c = bv_weights[min(c, bv_weights.shape[0] - 1)]
                sym_mv = self._symbolic_sandwich(channel_mvs[c], B_c)
                new_channel_mvs.append(sym_mv)
            channel_mvs = new_channel_mvs
            # Prune after rotor
            for c in range(len(channel_mvs)):
                channel_mvs[c] = self._prune_mv(channel_mvs[c])

            # 2e. Blade selector
            channel_mvs = self._symbolic_blade(channel_mvs, block.blade)

            # 2f. Skip connection (for use_skip=True blocks)
            if block.use_skip:
                for c in range(C):
                    for blade_idx in set(channel_mvs[c].keys()) | set(pre_block_mvs[c].keys()):
                        post = channel_mvs[c].get(blade_idx, sympy.Integer(0))
                        pre = pre_block_mvs[c].get(blade_idx, sympy.Integer(0))
                        channel_mvs[c][blade_idx] = post + pre

            # Final prune per block
            for c in range(len(channel_mvs)):
                channel_mvs[c] = self._prune_mv(channel_mvs[c])

        # 3. Output head: output_blade → output_linear
        #    (output_norm skipped — same reasoning as block norm with recover=False)
        channel_mvs = self._symbolic_blade(channel_mvs, model.output_blade)

        # output_linear: weight [1, C], bias [1, dim]
        out_w = model.output_linear.weight.detach().cpu().numpy()  # [1, C]
        out_b = model.output_linear.bias.detach().cpu().numpy()    # [1, dim]

        # Combine channels: sum_c weight[0, c] * channel_mvs[c] + bias
        combined_mv = {}
        for c in range(C):
            w_c = float(out_w[0, c])
            if abs(w_c) < 1e-10:
                continue
            for blade_idx, expr in channel_mvs[c].items():
                scaled = w_c * expr
                combined_mv[blade_idx] = combined_mv.get(
                    blade_idx, sympy.Integer(0)
                ) + scaled

        # Add bias
        for blade_idx in range(dim):
            b_val = float(out_b[0, blade_idx])
            if abs(b_val) > 1e-10:
                combined_mv[blade_idx] = combined_mv.get(
                    blade_idx, sympy.Integer(0)
                ) + b_val

        # 4. Extract grade-0 (scalar) component
        expr = combined_mv.get(0, sympy.Integer(0))

        # 5. Zero out phantom variables (beyond in_features) and their LUT symbols
        for i in range(k, n):
            expr = expr.subs(sympy.Symbol(f"x{i+1}"), 0)
            expr = expr.subs(sympy.Symbol(f"log_x{i+1}"), 0)
            expr = expr.subs(sympy.Symbol(f"inv_x{i+1}"), 0)

        # 6. Prune and simplify (use expand+collect, not simplify which can hang)
        expr = self._prune_expr(expr)
        n_terms = len(sympy.Add.make_args(expr))
        if n_terms <= 20:
            try:
                expr = sympy.nsimplify(expr, tolerance=1e-3)
            except Exception:
                pass
            expr = sympy.simplify(expr)
        else:
            # For large expressions, just collect terms
            expr = sympy.expand(expr)
            expr = self._prune_expr(expr)

        if expr == sympy.Integer(0):
            return []

        # Substitute LUT auxiliary symbols back to x_i-based expressions.
        # Use forms that lambdify handles cleanly with numpy:
        #   log_xi -> log(xi**2 + eps)/2  ≈ log(|xi|)
        #   inv_xi -> xi/(xi**2 + eps)    ≈ 1/xi
        _eps = sympy.Float(1e-12)
        for i in range(k):
            log_sym = sympy.Symbol(f"log_x{i+1}")
            inv_sym = sympy.Symbol(f"inv_x{i+1}")
            xi = sympy.Symbol(f"x{i+1}")
            if log_sym in expr.free_symbols:
                expr = expr.subs(log_sym, sympy.log(xi**2 + _eps) / 2)
            if inv_sym in expr.free_symbols:
                expr = expr.subs(inv_sym, xi / (xi**2 + _eps))

        # Lambdify with self.symbols so evaluate_terms/refine can call fn(*args)
        # Use explicit numpy module dict to ensure ufuncs work on compound exprs
        _numpy_mod = {"log": np.log, "sqrt": np.sqrt, "Abs": np.abs,
                      "sign": np.sign, "exp": np.exp, "sin": np.sin,
                      "cos": np.cos}
        fn = sympy.lambdify(self.symbols, expr, modules=[_numpy_mod, "numpy"])

        return [RotorTerm(
            planes=[],
            weight=1.0,
            expr=expr,
            fn=fn,
        )]

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

            result = t.fn(*args)
            result = np.broadcast_to(
                np.asarray(result, dtype=np.float64),
                (X_np.shape[0],),
            )
            y_hat += t.weight * result
        return y_hat
