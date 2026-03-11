# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Iterative Geometric Unbending v2 for Symbolic Regression.

4-phase pipeline:
  Phase 0: Data Preparation -- SVD alignment, variable grouping, implicit probe.
  Phase 1: Per-Group Iterative Extraction -- single-rotor-per-stage with GA
           orthogonal elimination (blade rejection, NOT numerical subtraction).
  Phase 2: Mother Algebra Cross-Term Discovery -- GPCA in Cl(P,Q,R) for
           cross-group interactions.
  Phase 3: SymPy Refinement -- lstsq reweight, implicit solve, simplify.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import sympy
import torch
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from core.search import GeodesicFlow, MetricSearch
from models.sr.translator import RotorTranslator, RotorTerm
from models.sr.net import SRGBN
from models.sr.utils import (
    LAMBDIFY_MODULES, safe_sympy_solve, safe_float, standardize,
    subsample, safe_svd, make_lambdify_fn, evaluate_terms,
)
from optimizers.riemannian import RiemannianAdam

logger = logging.getLogger(__name__)


@dataclass
class OrthogonalEliminationResult:
    """Result of GA blade rejection (orthogonal elimination)."""
    projection_energy: float
    rejection_energy: float
    soft_threshold: float
    preserved_fraction: float  # fraction of borderline terms kept


@dataclass
class StageResult:
    """Result of a single unbending stage."""

    stage_idx: int
    signature: tuple  # (p, q, r) for this stage's residual
    terms: list  # list[RotorTerm]
    fitted_values: np.ndarray  # sum of all term predictions
    residual_before: np.ndarray  # input to this stage
    residual_after: np.ndarray  # after elimination
    curvature_before: float
    curvature_after: float
    coherence_before: float  # on residual entering this stage
    coherence_after: float  # on residual after elimination
    rotor_planes: list  # dominant plane per layer
    accepted: bool  # False if backtracked
    elimination: OrthogonalEliminationResult = None
    group_idx: int = 0
    composition_ops: list = field(default_factory=list)  # kept for compat


@dataclass
class UnbendingResult:
    """Result of the full iterative unbending pipeline."""

    stages: list  # list[StageResult]
    formula: str
    r2_final: float
    all_terms: list  # list[RotorTerm]
    signature_history: list  # list[(p, q, r)] per stage
    groups: list = field(default_factory=list)  # list[VariableGroup]
    implicit_mode: str = "explicit"
    mother_cross_energy: float = 0.0
    all_ops: list = field(default_factory=list)  # "sub"|"div" per term


@dataclass
class _PrepResult:
    """Internal result of Phase 0 data preparation."""
    groups: list
    implicit_form: object  # ImplicitFormulation or None
    svd_Vt: np.ndarray
    svd_S: np.ndarray


class IterativeUnbender:
    """Iterative geometric unbending v2 for symbolic regression.

    4-phase pipeline with GA orthogonal elimination, variable grouping,
    implicit mode, SVD warm-start, and mother algebra cross-terms.
    """

    def __init__(
        self,
        in_features,
        device="cpu",
        max_stages=5,
        max_retries=2,
        curvature_weight=1.0,
        coherence_weight=0.3,
        mse_weight=0.1,
        sparsity_weight=0.01,
        stage_epochs=30,
        stage_lr=0.003,
        geodesic_k=8,
        r2_target=0.999,
        curvature_threshold=0.05,
        coherence_degradation_threshold=0.15,
        probe_config=None,
        implicit_mode='auto',
        grouping_enabled=True,
        max_groups=4,
        svd_warmstart=True,
        soft_rejection_alpha=10.0,
        soft_rejection_threshold=0.01,
        mother_cross_threshold=0.01,
        basis_config=None,
    ):
        self.in_features = in_features
        self.device = device
        self.max_stages = max_stages
        self.max_retries = max_retries
        self.curvature_weight = curvature_weight
        self.coherence_weight = coherence_weight
        self.mse_weight = mse_weight
        self.sparsity_weight = sparsity_weight
        self.stage_epochs = stage_epochs
        self.stage_lr = stage_lr
        self.geodesic_k = geodesic_k
        self.r2_target = r2_target
        self.curvature_threshold = curvature_threshold
        self.coherence_degradation_threshold = coherence_degradation_threshold
        self.probe_config = probe_config or {}
        self.implicit_mode = implicit_mode
        self.grouping_enabled = grouping_enabled
        self.max_groups = max_groups
        self.svd_warmstart = svd_warmstart
        self.soft_rejection_alpha = soft_rejection_alpha
        self.soft_rejection_threshold = soft_rejection_threshold
        self.mother_cross_threshold = mother_cross_threshold
        self.basis_config = basis_config or {}
        # Populated by basis expansion (if any) for formula assembly
        self._basis_result = None

    def run(self, X_norm, y_norm, x_mean, x_std, y_mean, y_std, var_names):
        """Run the full 4-phase iterative unbending pipeline.

        Args:
            X_norm: [N, k] normalized inputs (torch tensor).
            y_norm: [N, 1] normalized targets (torch tensor).
            x_mean, x_std: Input normalization stats (torch tensors).
            y_mean, y_std: Output normalization stats (torch tensors).
            var_names: List of variable name strings.

        Returns:
            UnbendingResult with all stages and assembled formula.
        """
        X_norm = X_norm.to(self.device)
        y_norm = y_norm.to(self.device)

        X_orig = self._to_numpy(X_norm * x_std.to(self.device) + x_mean.to(self.device))
        y_orig = self._to_numpy(
            y_norm.squeeze(-1) * y_std.to(self.device) + y_mean.to(self.device)
        )

        # Phase 0: Basis expansion (before linearity check)
        self._basis_result = None
        if self.basis_config.get("enabled", False):
            X_orig, y_orig, var_names = self._apply_basis_expansion(
                X_orig, y_orig, var_names,
            )
            # Update in_features for downstream model building
            self.in_features = X_orig.shape[1]
            # Re-normalize expanded data for GBN training
            X_expanded_t = torch.tensor(X_orig, dtype=torch.float32, device=self.device)
            x_mean = X_expanded_t.mean(0)
            x_std = X_expanded_t.std(0).clamp(min=1e-6)
            X_norm = (X_expanded_t - x_mean) / x_std
            y_expanded_t = torch.tensor(y_orig, dtype=torch.float32, device=self.device)
            y_mean = y_expanded_t.mean()
            y_std = y_expanded_t.std().clamp(min=1e-6)
            y_norm = ((y_expanded_t - y_mean) / y_std).unsqueeze(-1)

        # Phase 0a: Linearity short-circuit
        is_linear, lin_terms, lin_r2 = self._check_linearity(X_orig, y_orig)
        if is_linear:
            logger.info(f"Linear model sufficient (R2={lin_r2:.4f}), skipping unbending")
            vnames = var_names if var_names else [f"x{i+1}" for i in range(self.in_features)]
            lin_ops = ["sub"] * len(lin_terms)
            formula = self._assemble_formula(lin_terms, vnames, lin_ops)
            # Wrap in exp() if target was log-transformed
            if self._basis_result is not None and self._basis_result.log_target:
                formula = self._wrap_log_target(formula)
            return UnbendingResult(
                stages=[],
                formula=formula,
                r2_final=lin_r2,
                all_terms=lin_terms,
                signature_history=[],
                all_ops=lin_ops,
            )

        # Phase 0b: Data preparation
        logger.info("Phase 0: Data preparation")
        prep = self._prepare_data(X_orig, y_orig, X_norm, y_norm, var_names)

        # Phase 1: Per-group extraction
        logger.info("Phase 1: Per-group iterative extraction")
        all_terms = []
        all_ops = []
        all_stages = []
        ss_tot = np.sum((y_orig - y_orig.mean()) ** 2) + 1e-12

        for group_idx, group in enumerate(prep.groups):
            terms, stages = self._process_group(
                group, group_idx, prep, X_orig, y_orig, X_norm, y_norm,
            )
            all_terms.extend(terms)
            all_ops.extend(["sub"] * len(terms))
            all_stages.extend(stages)

        # Phase 2: Mother algebra cross-terms
        mother_cross_energy = 0.0
        if len(prep.groups) > 1:
            logger.info("Phase 2: Mother algebra cross-term discovery")
            cross_terms, cross_energy = self._mother_algebra_joint(
                prep.groups, X_orig, y_orig, X_norm,
            )
            all_terms.extend(cross_terms)
            all_ops.extend(["sub"] * len(cross_terms))
            mother_cross_energy = cross_energy
        else:
            logger.info("Phase 2: Skipped (single group)")

        # Phase 3: SymPy refinement
        logger.info("Phase 3: Joint refinement")
        if all_terms:
            all_terms, all_ops = self._refine_all_terms(
                all_terms, X_orig, y_orig, all_ops,
            )

        # Final R2
        y_hat = evaluate_terms(all_terms, X_orig)
        ss_res = np.sum((y_orig - y_hat) ** 2)
        r2_final = 1.0 - ss_res / ss_tot

        # If explicit R2 is poor and implicit wasn't tried, try implicit fallback
        implicit_form = prep.implicit_form
        if (r2_final < 0.5
                and self.implicit_mode != 'explicit'
                and len(prep.groups) == 1
                and (implicit_form is None or implicit_form.mode != "implicit")):
            logger.info(f"Explicit R2={r2_final:.4f} < 0.5, trying implicit fallback")
            from models.sr.implicit import ImplicitFormulation
            fallback_form = ImplicitFormulation(
                target_var_idx=X_norm.shape[1], mode="implicit",
            )
            fallback_prep = _PrepResult(
                groups=prep.groups, implicit_form=fallback_form,
                svd_Vt=prep.svd_Vt, svd_S=prep.svd_S,
            )
            impl_terms, impl_stages = self._process_group_implicit(
                prep.groups[0], 0, fallback_prep, X_orig, y_orig, X_norm, y_norm,
            )
            if impl_terms:
                impl_y_hat = evaluate_terms(impl_terms, X_orig)
                impl_ss_res = np.sum((y_orig - impl_y_hat) ** 2)
                impl_r2 = 1.0 - impl_ss_res / ss_tot
                if impl_r2 > r2_final:
                    logger.info(f"Implicit fallback improved R2: "
                                f"{r2_final:.4f} -> {impl_r2:.4f}")
                    all_terms = impl_terms
                    all_ops = ["sub"] * len(impl_terms)
                    all_stages = impl_stages
                    r2_final = impl_r2
                    implicit_form = fallback_form

        # Assemble formula
        vnames = var_names if var_names else [f"x{i+1}" for i in range(self.in_features)]
        formula = self._assemble_formula(
            all_terms, vnames, all_ops, implicit_form=implicit_form,
        )
        # Wrap in exp() if target was log-transformed
        if self._basis_result is not None and self._basis_result.log_target:
            formula = self._wrap_log_target(formula)

        return UnbendingResult(
            stages=all_stages,
            formula=formula,
            r2_final=r2_final,
            all_terms=all_terms,
            signature_history=[s.signature for s in all_stages],
            groups=prep.groups,
            implicit_mode=implicit_form.mode if implicit_form else "explicit",
            mother_cross_energy=mother_cross_energy,
            all_ops=all_ops,
        )

    # =======================================================================
    # Phase 0: Data Preparation
    # =======================================================================

    def _prepare_data(self, X_orig, y_orig, X_norm, y_norm, var_names):
        """SVD alignment, variable grouping, implicit mode probe."""
        # SVD align
        X_c = X_orig - X_orig.mean(axis=0)
        S, Vt = safe_svd(X_c)
        if S is None:
            S, Vt = np.ones(X_orig.shape[1]), np.eye(X_orig.shape[1])

        # Variable grouping
        if self.grouping_enabled:
            from models.sr.grouper import VariableGrouper
            grouper = VariableGrouper(
                max_groups=self.max_groups, device=self.device,
            )
            groups = grouper.group(X_orig, y_orig, var_names)
        else:
            groups = [self._single_group_fallback(X_orig, y_orig, var_names, Vt)]

        # Implicit mode probe
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
                # Subsample for faster/more responsive probing
                probe_data = torch.cat([X_norm, y_norm], dim=-1)
                probe_data = subsample(probe_data, 500)
                X_probe = probe_data[:, :-1]
                y_probe = probe_data[:, -1:]
                solver_result = solver.probe_best_mode(algebra, X_probe, y_probe)
                if self.implicit_mode == 'auto':
                    implicit_form = solver_result
                elif self.implicit_mode == 'implicit':
                    solver_result.mode = 'implicit'
                    implicit_form = solver_result
            except Exception as e:
                logger.warning(f"Implicit probe failed: {e}")

        return _PrepResult(groups=groups, implicit_form=implicit_form,
                          svd_Vt=Vt, svd_S=S)

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

        try:
            searcher = MetricSearch(device=self.device, num_probes=4,
                                    probe_epochs=40, micro_batch_size=64)
            p, q, r = searcher.search(data)
            if p + q + r < 2:
                p = max(p, 2)
            # Clamp p to prevent over-dimensioning for low-variable problems
            max_p = max(n_vars, 2)
            if p > max_p:
                logger.info(f"MetricSearch clamped p: {p} -> {max_p} (n_vars={n_vars})")
                p = max_p
        except Exception:
            p, q, r = min(n_vars, 4), 0, 0

        algebra = CliffordAlgebra(p, q, r, device=self.device)
        return VariableGroup(
            var_indices=list(range(n_vars)),
            var_names=var_names or [f"x{i+1}" for i in range(n_vars)],
            signature=(p, q, r),
            algebra=algebra,
            svd_Vt=Vt,
        )

    # =======================================================================
    # Phase 1: Per-Group Iterative Extraction
    # =======================================================================

    def _process_group(self, group, group_idx, prep, X_orig, y_orig,
                       X_norm, y_norm):
        """Single-rotor iterative extraction for one variable group.

        Dispatches to implicit or explicit extraction based on prep result.

        Returns:
            (list[RotorTerm], list[StageResult])
        """
        # If implicit mode selected, use implicit extraction for this group
        if prep.implicit_form is not None and prep.implicit_form.mode == "implicit":
            terms, stages = self._process_group_implicit(
                group, group_idx, prep, X_orig, y_orig, X_norm, y_norm,
            )
            return terms, stages

        return self._process_group_explicit(
            group, group_idx, prep, X_orig, y_orig, X_norm, y_norm,
        )

    def _process_group_implicit(self, group, group_idx, prep, X_orig, y_orig,
                                 X_norm, y_norm):
        """Implicit extraction: train F(x,y)=0, extract via sympy.solve.

        Returns:
            (list[RotorTerm], list[StageResult])
        """
        from models.sr.implicit import ImplicitSolver

        algebra = group.algebra
        var_indices = group.var_indices
        n_group_vars = len(var_indices)

        # Build augmented algebra (k+1 variables)
        p, q, r = group.signature
        impl_algebra = CliffordAlgebra(p + 1, q, r, device=self.device)

        # Augmented data Z = [X_group_norm, y_norm]
        X_group_norm = standardize(
            torch.tensor(X_orig[:, var_indices], dtype=torch.float32, device=self.device)
        )
        Z = torch.cat([X_group_norm, y_norm], dim=-1)  # [N, k+1]

        # Train implicit model
        solver = ImplicitSolver(device=self.device)

        auto = SRGBN.auto_config(Z.shape[0], n_group_vars + 1, impl_algebra.dim)
        model = SRGBN.single_rotor(
            impl_algebra, n_group_vars + 1, channels=max(auto["channels"], 16),
        )
        model = model.to(self.device)

        # Break F==0 dead gradient (see implicit_solver._probe_implicit)
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'grade0_bias'):
                    torch.nn.init.normal_(m.grade0_bias, std=1.0)

        model = solver.train_implicit(
            model, Z, impl_algebra,
            epochs=max(self.stage_epochs * 3, 60),
            lr=self.stage_lr,
        )

        # Extract via direct symbolic expansion, fallback to legacy
        translator = RotorTranslator(impl_algebra)
        impl_terms = translator.translate_direct(model)
        if impl_terms:
            # Rename target variable x_{k+1} -> y for implicit solve
            target_sym = sympy.Symbol(f"x{n_group_vars + 1}")
            y_sym_sub = sympy.Symbol("y")
            for t in impl_terms:
                if t.expr is not None:
                    t.expr = t.expr.subs(target_sym, y_sym_sub)
        else:
            impl_terms = translator.translate_implicit(model, target_var_idx=n_group_vars)

        if not impl_terms:
            logger.info(f"  Group {group_idx}: implicit extraction found no terms, "
                         f"falling back to explicit")
            return self._process_group_explicit(
                group, group_idx, prep, X_orig, y_orig, X_norm, y_norm,
            )

        # Build F expression and solve for y
        F_expr = sympy.Integer(0)
        for t in impl_terms:
            F_expr += t.weight * t.expr

        y_sym = sympy.Symbol("y")
        var_syms = [sympy.Symbol(f"x{i+1}") for i in range(n_group_vars)]

        explicit_expr = None
        try:
            solutions = sympy.solve(F_expr, y_sym)
            if solutions:
                explicit_expr = solutions[0]
        except Exception:
            pass

        if explicit_expr is None:
            # Keep implicit form F=0
            explicit_expr = F_expr

        # Build callable if we have an explicit (y-free) expression
        has_y = y_sym in explicit_expr.free_symbols
        result_fn = None
        if not has_y:
            # Zero out phantom variables beyond the group
            for s in list(explicit_expr.free_symbols):
                if s not in var_syms and s != y_sym:
                    explicit_expr = explicit_expr.subs(s, 0)
            n_terms = len(sympy.Add.make_args(explicit_expr))
            if n_terms <= 20:
                explicit_expr = sympy.simplify(explicit_expr)
            else:
                explicit_expr = sympy.expand(explicit_expr)
            result_fn = make_lambdify_fn(var_syms, explicit_expr)

        result_term = RotorTerm(
            planes=[p for t in impl_terms for p in t.planes],
            weight=1.0,
            expr=explicit_expr,
            fn=result_fn,
        )

        # Evaluate R2
        r2 = 0.0
        if result_term.fn is not None:
            args = [X_orig[:, var_indices[i]] for i in range(n_group_vars)]
            try:
                y_hat = result_term.fn(*args)
                if np.all(np.isfinite(y_hat)):
                    ss_tot = np.sum((y_orig - y_orig.mean()) ** 2) + 1e-12
                    ss_res = np.sum((y_orig - y_hat) ** 2)
                    r2 = 1.0 - ss_res / ss_tot
            except Exception:
                pass
            logger.info(f"  Group {group_idx}: implicit extraction R2={r2:.4f}")

        stage = StageResult(
            stage_idx=0, signature=group.signature,
            terms=[result_term], fitted_values=np.zeros(len(y_orig)),
            residual_before=y_orig, residual_after=np.zeros(len(y_orig)),
            curvature_before=0.5, curvature_after=0.0,
            coherence_before=0.5, coherence_after=0.5,
            rotor_planes=[], accepted=True,
            group_idx=group_idx,
        )

        return [result_term], [stage]

    def _process_group_explicit(self, group, group_idx, prep, X_orig, y_orig,
                                 X_norm, y_norm):
        """Explicit single-rotor iterative extraction for one variable group.

        Returns:
            (list[RotorTerm], list[StageResult])
        """
        algebra = group.algebra
        var_indices = group.var_indices
        n_group_vars = len(var_indices)

        # Data for this group
        X_group = X_orig[:, var_indices]
        X_group_norm = standardize(
            torch.tensor(X_group, dtype=torch.float32, device=self.device)
        )

        # Build residual multivector for GA elimination
        residual = y_orig.copy()
        residual_mv = algebra.embed_vector(
            torch.tensor(
                np.column_stack([X_group, residual.reshape(-1, 1)]),
                dtype=torch.float32, device=self.device,
            )
        ) if algebra.n >= n_group_vars + 1 else None

        prev_coherence = safe_float(
            self._measure_coherence(X_group, residual, algebra), 0.5,
        )

        terms = []
        stages = []
        ss_tot = np.sum((y_orig - y_orig.mean()) ** 2) + 1e-12

        for stage_idx in range(self.max_stages):
            r2_current = 1.0 - np.sum(residual ** 2) / ss_tot
            if r2_current >= self.r2_target:
                logger.info(f"  Group {group_idx} R2={r2_current:.6f} >= target")
                break

            # Build single-rotor SRGBN with adequate capacity
            N_group = X_group.shape[0]
            auto = SRGBN.auto_config(N_group, n_group_vars, algebra.dim)
            channels = max(auto["channels"], 8)
            model = SRGBN.single_rotor(
                algebra, n_group_vars, channels=channels,
            )
            model = model.to(self.device)

            # SVD warm-start
            if self.svd_warmstart and group.svd_Vt is not None:
                model.svd_warmstart(group.svd_Vt, algebra)

            # Normalize residual for training
            residual_t = torch.tensor(
                residual, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
            res_mean = residual_t.mean()
            res_std = residual_t.std().clamp(min=1e-8)
            residual_norm = (residual_t - res_mean) / res_std

            # Probe curvature
            probe_curv = safe_float(
                self._measure_curvature(X_group, residual, algebra), 0.5,
            )

            # Train with MSE-primary objective for single-rotor extraction
            model, curv_after, coh_after = self._train_single_rotor(
                model, X_group_norm, residual_norm, algebra,
            )
            curv_after = safe_float(curv_after, 0.5)
            coh_after = safe_float(coh_after, 0.5)

            # Extract terms via direct symbolic expansion
            translator = RotorTranslator(algebra)
            stage_terms = translator.translate_direct(model)
            if not stage_terms:
                # Fallback to legacy plane-by-plane translation
                stage_terms = translator.translate(model)

            if not stage_terms:
                logger.info(f"  Group {group_idx} stage {stage_idx}: no terms, stopping")
                break

            # GA orthogonal elimination
            elim_result = None
            if residual_mv is not None:
                blade = self._extract_dominant_blade(model, algebra)
                residual_mv, elim_result = self._orthogonal_eliminate(
                    residual_mv, blade, algebra,
                )

            # Numerical residual update (for R2 tracking)
            fitted = translator.evaluate_terms(stage_terms, X_group)
            new_residual = residual.copy()
            comp_ops = []

            for t in stage_terms:
                term_val = translator.evaluate_terms([t], X_group)
                new_residual = new_residual - term_val
                comp_ops.append("sub")

            fitted = residual - new_residual

            # Coherence check
            new_coh = safe_float(
                self._measure_coherence(X_group, new_residual, algebra), 0.5,
            )
            degradation = prev_coherence - new_coh

            ss_tot_local = np.sum((residual - residual.mean()) ** 2) + 1e-12
            ss_res_local = np.sum(new_residual ** 2)
            r2_extraction = 1.0 - ss_res_local / ss_tot_local

            n_vars = X_group.shape[1]
            skip_coherence = (r2_extraction > 0.9) or (n_vars <= 1)

            accepted = True
            if not skip_coherence and degradation > self.coherence_degradation_threshold:
                logger.warning(
                    f"  Group {group_idx} stage {stage_idx}: coherence degraded "
                    f"{prev_coherence:.3f} -> {new_coh:.3f}, rejecting stage"
                )
                accepted = False

            rotor_planes = self._get_active_rotor_planes(model)

            stage = StageResult(
                stage_idx=stage_idx,
                signature=group.signature,
                terms=stage_terms,
                fitted_values=fitted,
                residual_before=residual,
                residual_after=new_residual,
                curvature_before=probe_curv,
                curvature_after=curv_after,
                coherence_before=prev_coherence,
                coherence_after=new_coh,
                rotor_planes=rotor_planes,
                accepted=accepted,
                elimination=elim_result,
                group_idx=group_idx,
                composition_ops=comp_ops,
            )
            stages.append(stage)

            if accepted:
                residual = new_residual
                prev_coherence = new_coh
                terms.extend(stage_terms)

            # Stopping conditions
            if elim_result and elim_result.rejection_energy < self.curvature_threshold:
                logger.info(
                    f"  Group {group_idx} stage {stage_idx}: "
                    f"rejection energy {elim_result.rejection_energy:.3f} < threshold"
                )
                break

            if curv_after < self.curvature_threshold:
                break

        return terms, stages

    def _orthogonal_eliminate(self, data_mv, blade, algebra):
        """Soft GA rejection: preserve subtle terms near threshold.

        Instead of hard rejection (data - proj), uses sigmoid gating
        so components >> threshold are fully eliminated while components
        << threshold are fully preserved.
        """
        proj = algebra.blade_project(data_mv, blade)
        proj_energy = proj.pow(2).sum(dim=-1)

        # Soft sigmoid mask
        soft_mask = torch.sigmoid(
            self.soft_rejection_alpha * (proj_energy - self.soft_rejection_threshold)
        )
        rejected = data_mv - soft_mask.unsqueeze(-1) * proj

        proj_energy_total = proj_energy.sum().item()
        rej_energy = rejected.pow(2).sum().item()

        # Fraction preserved (borderline terms kept)
        near_threshold = (proj_energy > self.soft_rejection_threshold * 0.5) & \
                         (proj_energy < self.soft_rejection_threshold * 2.0)
        preserved = (1.0 - soft_mask[near_threshold]).mean().item() \
            if near_threshold.any() else 1.0

        return rejected, OrthogonalEliminationResult(
            projection_energy=proj_energy_total,
            rejection_energy=rej_energy,
            soft_threshold=self.soft_rejection_threshold,
            preserved_fraction=preserved,
        )

    def _extract_dominant_blade(self, model, algebra):
        """Read the trained rotor's dominant bivector as a blade tensor."""
        bv_weights = model.blocks[0].rotor.bivector_weights.detach()
        # Average across channels
        bv_mean = bv_weights.mean(dim=0)  # [n_bv]
        # Build full multivector with only grade-2 components
        blade = torch.zeros(algebra.dim, device=bv_mean.device)
        bv_mask = algebra.grade_masks[2]
        if bv_mask.device != bv_mean.device:
            bv_mask = bv_mask.to(bv_mean.device)
        blade[bv_mask] = bv_mean
        return blade

    # =======================================================================
    # Phase 2: Mother Algebra Cross-Terms
    # =======================================================================

    def _mother_algebra_joint(self, groups, X_orig, y_orig, X_norm):
        """Check cross-group interactions via geometric product in mother algebra.

        Returns:
            (list[RotorTerm], float): Cross-terms and cross-energy.
        """
        if len(groups) < 2:
            return [], 0.0

        from models.sr.grouper import VariableGrouper
        grouper = VariableGrouper(max_groups=self.max_groups, device=self.device)

        try:
            mother_alg, offsets = grouper.build_mother_algebra(groups)
        except Exception as e:
            logger.warning(f"Mother algebra construction failed: {e}")
            return [], 0.0

        # Embed each group's data into mother algebra
        N = X_orig.shape[0]
        mother_mvs = []
        for g in groups:
            X_g = X_orig[:, g.var_indices]
            X_g_std = standardize(X_g)
            # Pad to group algebra dim
            n_g = g.algebra.n
            if X_g_std.shape[1] < n_g:
                X_g_std = np.column_stack([
                    X_g_std, np.zeros((N, n_g - X_g_std.shape[1]))
                ])
            elif X_g_std.shape[1] > n_g:
                X_g_std = X_g_std[:, :n_g]

            local_mv = g.algebra.embed_vector(
                torch.tensor(X_g_std, dtype=torch.float32, device=self.device)
            )
            mother_mv = grouper.inject_to_mother(local_mv, g, mother_alg)
            mother_mvs.append(mother_mv)

        # Compute GP between groups -> check grade-2 energy
        cross_energy = 0.0
        if len(mother_mvs) >= 2:
            gp = mother_alg.geometric_product(mother_mvs[0], mother_mvs[1])
            # Grade-2 energy of cross product
            bv_mask = mother_alg.grade_masks[2]
            if bv_mask.device != gp.device:
                bv_mask = bv_mask.to(gp.device)
            cross_bv = gp[..., bv_mask]
            cross_energy = cross_bv.pow(2).mean().item()

        if cross_energy < self.mother_cross_threshold:
            logger.info(f"  Cross-energy {cross_energy:.4f} < threshold, no cross-terms")
            return [], cross_energy

        # Train joint single-rotor in mother algebra
        logger.info(f"  Cross-energy {cross_energy:.4f} detected, training joint rotor")
        try:
            auto = SRGBN.auto_config(X_orig.shape[0], self.in_features, mother_alg.dim)
            model = SRGBN.single_rotor(mother_alg, self.in_features,
                                        channels=max(auto["channels"], 8))
            model = model.to(self.device)

            residual_t = torch.tensor(
                y_orig, dtype=torch.float32, device=self.device,
            ).unsqueeze(-1)
            res_mean = residual_t.mean()
            res_std = residual_t.std().clamp(min=1e-8)
            residual_norm = (residual_t - res_mean) / res_std

            X_norm_std = standardize(X_norm) if isinstance(X_norm, torch.Tensor) else X_norm
            model, _, _ = self._train_stage(
                model, X_norm_std, residual_norm, mother_alg,
            )

            translator = RotorTranslator(mother_alg)
            cross_terms = translator.translate_direct(model)
            if not cross_terms:
                cross_terms = translator.translate(model)
            return cross_terms, cross_energy

        except Exception as e:
            logger.warning(f"Joint rotor training failed: {e}")
            return [], cross_energy

    # =======================================================================
    # Phase 3: SymPy Refinement
    # =======================================================================

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
        A = np.column_stack(term_preds + [np.ones(X_orig.shape[0])])
        new_weights = np.linalg.lstsq(A, y_orig, rcond=None)[0]

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

    # =======================================================================
    # Shared Helpers
    # =======================================================================

    def _check_linearity(self, X_raw, y_raw, r2_threshold=0.90):
        """Multi-branch linearity check with BIC parsimony and power-law action.

        Three branches:
          1. Standard linear fit
          2. Log-log power law (builds explicit power-law term if R2 >= 0.90)
          3. BIC comparison: linear vs quadratic -- prefer simpler form

        Returns (is_linear_or_powerlaw, terms, r2) where terms are list[RotorTerm].
        """
        N, d = X_raw.shape
        symbols = [sympy.Symbol(f"x{i+1}") for i in range(d)]

        # Branch 1: Standard linear fit
        A_lin = np.column_stack([X_raw, np.ones(N)])
        coeffs_lin = np.linalg.lstsq(A_lin, y_raw, rcond=None)[0]
        y_hat_lin = A_lin @ coeffs_lin
        ss_res_lin = np.sum((y_raw - y_hat_lin) ** 2)
        ss_tot = np.sum((y_raw - y_raw.mean()) ** 2) + 1e-12
        r2_lin = 1.0 - ss_res_lin / ss_tot

        # Branch 2: Log-log fit (power law detection + action)
        eps = 1e-8
        pos_mask = np.all(np.abs(X_raw) > eps, axis=1) & (np.abs(y_raw) > eps)
        r2_log = -1.0
        powerlaw_term = None

        if pos_mask.sum() > max(10, int(N * 0.8)):
            X_pos = np.abs(X_raw[pos_mask])
            y_pos = np.abs(y_raw[pos_mask])
            log_X = np.log(X_pos)
            log_y = np.log(y_pos)
            A_log = np.column_stack([log_X, np.ones(pos_mask.sum())])
            coeffs_log = np.linalg.lstsq(A_log, log_y, rcond=None)[0]
            log_y_hat = A_log @ coeffs_log
            ss_res_log = np.sum((log_y - log_y_hat) ** 2)
            ss_tot_log = np.sum((log_y - log_y.mean()) ** 2) + 1e-12
            r2_log = 1.0 - ss_res_log / ss_tot_log

            exponents = coeffs_log[:d]
            log_intercept = coeffs_log[d]
            has_nonunit = any(
                abs(e - 1.0) > 0.15 and abs(e) > 0.15 for e in exponents
            )

            # Action: if power law fits well, build explicit term
            if r2_log >= 0.90 and has_nonunit:
                # Round exponents to nearest simple fraction
                rounded_exp = [self._round_exponent(e) for e in exponents]
                # y = exp(intercept) * prod(xi^alpha_i)
                # Determine sign of y from original data
                y_sign = np.sign(np.median(y_raw[pos_mask]))
                scale = float(np.exp(log_intercept))
                if y_sign < 0:
                    scale = -scale

                expr = sympy.Float(scale)
                for i in range(d):
                    if abs(rounded_exp[i]) > 1e-3:
                        expr = expr * symbols[i] ** sympy.Rational(rounded_exp[i]).limit_denominator(6)

                # Evaluate on full data to compute R2
                fn = make_lambdify_fn(symbols, expr)
                try:
                    args = [X_raw[:, i] for i in range(d)]
                    y_hat_pl = fn(*args)
                    y_hat_pl = np.broadcast_to(
                        np.asarray(y_hat_pl, dtype=np.float64), (N,)
                    ).copy()
                    if np.all(np.isfinite(y_hat_pl)):
                        ss_res_pl = np.sum((y_raw - y_hat_pl) ** 2)
                        r2_pl = 1.0 - ss_res_pl / ss_tot
                        powerlaw_term = RotorTerm(
                            planes=[], weight=1.0, expr=expr, fn=fn,
                        )
                        logger.info(
                            f"Power law short-circuit: exponents={rounded_exp}, "
                            f"R2={r2_pl:.4f} (log-space R2={r2_log:.4f})"
                        )
                        # If power law is great, return immediately
                        if r2_pl >= r2_threshold:
                            return True, [powerlaw_term], r2_pl
                except Exception:
                    powerlaw_term = None

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
                    X_raw, y_raw, coeffs_lin, ss_res_lin,
                )
            else:
                bic_prefers_linear = True  # skip BIC for high-d

            if bic_prefers_linear or r2_lin >= 0.995:
                intercept = float(coeffs_lin[d])
                expr = sympy.Float(intercept)
                for i in range(d):
                    if abs(coeffs_lin[i]) > 1e-8:
                        expr = expr + float(coeffs_lin[i]) * symbols[i]
                fn = make_lambdify_fn(symbols, expr)
                term = RotorTerm(planes=[], weight=1.0, expr=expr, fn=fn)
                logger.info(
                    f"Linear model accepted (R2={r2_lin:.4f}, BIC_prefers_linear={bic_prefers_linear})"
                )
                return True, [term], r2_lin

        return False, [], r2_lin

    @staticmethod
    def _round_exponent(e):
        """Round exponent to nearest simple fraction: 0, +-1/3, +-1/2, +-1, +-2, +-3."""
        candidates = [0, 1/3, 1/2, 1, 2, 3, -1/3, -1/2, -1, -2, -3]
        best = min(candidates, key=lambda c: abs(e - c))
        return best

    @staticmethod
    def _bic_prefers_simpler(X_raw, y_raw, coeffs_lin, ss_res_lin):
        """BIC comparison: linear vs quadratic. Returns True if linear wins."""
        N, d = X_raw.shape

        # Linear BIC: k_lin = d + 1 (coefficients + intercept)
        k_lin = d + 1
        bic_lin = N * np.log(ss_res_lin / N + 1e-30) + k_lin * np.log(N)

        # Quadratic features: x_i*x_j for all i<=j, plus linear
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
        except Exception:
            return True  # Can't fit quadratic, prefer linear

        # Prefer simpler (linear) unless quadratic BIC is substantially better
        # DELTA_BIC < 6 means "not strong evidence" for the complex model
        delta_bic = bic_quad - bic_lin
        logger.debug(f"BIC comparison: linear={bic_lin:.1f}, quad={bic_quad:.1f}, delta={delta_bic:.1f}")
        return delta_bic > -6.0

    def _select_active_vars(self, X_raw, y_raw, max_vars=6):
        """Select top-k variables by |correlation| with target."""
        n_vars = X_raw.shape[1]
        if n_vars <= max_vars:
            return list(range(n_vars))

        correlations = []
        for i in range(n_vars):
            if np.std(X_raw[:, i]) < 1e-12:
                correlations.append(0.0)
            else:
                corr = abs(np.corrcoef(X_raw[:, i], y_raw)[0, 1])
                correlations.append(corr if np.isfinite(corr) else 0.0)

        ranked = sorted(range(n_vars), key=lambda i: correlations[i], reverse=True)
        return ranked[:max_vars]

    def _probe_residual(self, X_raw, residual_raw, n_probes=4):
        """Probe the metric signature of [X, residual] data manifold."""
        active_vars = self._select_active_vars(X_raw, residual_raw, max_vars=5)
        self._active_vars = active_vars

        X_selected = X_raw[:, active_vars]
        combined = np.column_stack([X_selected, residual_raw.reshape(-1, 1)])
        data = torch.tensor(combined, dtype=torch.float32, device=self.device)
        data = standardize(data)
        data = subsample(data, 500)

        searcher = MetricSearch(
            device=self.device,
            num_probes=n_probes,
            probe_epochs=self.probe_config.get("probe_epochs", 40),
            micro_batch_size=self.probe_config.get("micro_batch_size", 64),
            early_stop_patience=self.probe_config.get("early_stop_patience", 8),
        )
        result = searcher.search_detailed(data)
        return result

    def _build_stage_model(self, algebra, n_train, stage_idx=0):
        """Build a stage-specific SRGBN with capacity decay."""
        auto = SRGBN.auto_config(n_train, self.in_features, algebra.dim)
        decay = 0.7 ** stage_idx
        channels = max(4, int(auto["channels"] * decay))
        num_layers = auto["num_layers"]

        model = SRGBN(
            algebra=algebra,
            in_features=self.in_features,
            channels=channels,
            num_layers=num_layers,
            use_decomposition=True,
        )
        return model.to(self.device)

    def _train_stage(self, model, X_norm, residual_norm, algebra):
        """Train a stage model with curvature-primary objective.

        Returns:
            (model, final_curvature, final_coherence)
        """
        gf = GeodesicFlow(algebra, k=self.geodesic_k)
        optimizer = RiemannianAdam(
            model.parameters(), lr=self.stage_lr, algebra=algebra
        )
        N = X_norm.shape[0]
        micro_bs = min(256, N)

        model.train()
        for epoch in range(self.stage_epochs):
            optimizer.zero_grad()

            if N > micro_bs:
                idx = torch.randperm(N, device=self.device)[:micro_bs]
                x_batch = X_norm[idx]
                r_batch = residual_norm[idx]
            else:
                x_batch = X_norm
                r_batch = residual_norm

            pred = model(x_batch)
            hidden = model._hidden_for_curvature.mean(dim=1)

            curv = gf._curvature_tensor(hidden)
            coh = gf._coherence_tensor(hidden)
            mse = F.mse_loss(pred, r_batch)
            sparsity = model.total_sparsity_loss()

            if not torch.isfinite(curv):
                curv = torch.tensor(0.0, device=self.device)
            if not torch.isfinite(coh):
                coh = torch.tensor(0.0, device=self.device)

            loss = (
                self.curvature_weight * curv
                - self.coherence_weight * coh
                + self.mse_weight * mse
                + self.sparsity_weight * sparsity
            )

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            model(X_norm)
            h = model._last_hidden.mean(dim=1)
            final_curv = gf.curvature(h)
            final_coh = gf.coherence(h)
        return model, final_curv, final_coh

    def _train_single_rotor(self, model, X_norm, residual_norm, algebra):
        """Train a single-rotor model with MSE-primary objective.

        Unlike _train_stage (curvature-primary), this focuses on fitting the
        residual so the rotor's bivector actually captures the data structure.
        Light sparsity keeps the extraction interpretable.

        Returns:
            (model, final_curvature, final_coherence)
        """
        gf = GeodesicFlow(algebra, k=self.geodesic_k)
        optimizer = RiemannianAdam(
            model.parameters(), lr=self.stage_lr, algebra=algebra
        )
        N = X_norm.shape[0]
        micro_bs = min(256, N)

        # More epochs for single-rotor: needs time to learn without skip
        n_epochs = max(self.stage_epochs * 3, 60)

        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()

            if N > micro_bs:
                idx = torch.randperm(N, device=self.device)[:micro_bs]
                x_batch = X_norm[idx]
                r_batch = residual_norm[idx]
            else:
                x_batch = X_norm
                r_batch = residual_norm

            pred = model(x_batch)
            mse = F.mse_loss(pred, r_batch)
            sparsity = model.total_sparsity_loss()

            # MSE-primary: strong data fitting, light sparsity
            loss = mse + 0.001 * sparsity

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            model(X_norm)
            h = model._last_hidden.mean(dim=1)
            final_curv = gf.curvature(h)
            final_coh = gf.coherence(h)
        return model, final_curv, final_coh

    def _measure_coherence(self, X_raw, residual, algebra):
        """Measure coherence of the [X, residual] manifold."""
        data = self._prepare_manifold_data(X_raw, residual, algebra)
        mv = algebra.embed_vector(data)
        gf = GeodesicFlow(algebra, k=min(self.geodesic_k, data.shape[0] - 1))
        return gf.coherence(mv)

    def _measure_curvature(self, X_raw, residual, algebra):
        """Measure curvature of the [X, residual] manifold."""
        data = self._prepare_manifold_data(X_raw, residual, algebra)
        mv = algebra.embed_vector(data)
        gf = GeodesicFlow(algebra, k=min(self.geodesic_k, data.shape[0] - 1))
        return gf.curvature(mv)

    def _prepare_manifold_data(self, X_raw, residual, algebra):
        """Build [X, residual] data fitted to algebra dimension n.

        Always keeps the residual column (last). If d > n, truncates
        X columns (not residual) to fit. If d < n, zero-pads.
        """
        combined = np.column_stack([X_raw, residual.reshape(-1, 1)])
        data = torch.tensor(combined, dtype=torch.float32, device=self.device)
        data = standardize(data)
        data = subsample(data, 256)

        n = algebra.n
        d = data.shape[1]
        if d > n:
            # Keep residual (last column), truncate X columns to fit
            residual_col = data[:, -1:]
            x_cols = data[:, :n - 1]
            data = torch.cat([x_cols, residual_col], dim=-1)
        elif d < n:
            pad = torch.zeros(data.shape[0], n - d, device=self.device)
            data = torch.cat([data, pad], dim=-1)

        return data

    def _get_active_rotor_planes(self, model):
        """Get list of dominant plane names from active rotors."""
        analysis = model.get_rotor_analysis()
        return [info["dominant_plane"] for info in analysis]

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
        expanded_names = [result.var_name_map.get(i, f"z{i+1}") for i in range(n_expanded)]

        logger.info(
            f"BasisExpander: {result.n_original} -> {n_expanded} features, "
            f"log_target={result.log_target}"
        )
        return result.X_expanded, y_out, expanded_names

    @staticmethod
    def _wrap_log_target(formula):
        """Wrap formula in exp() when target was log-transformed."""
        if formula.startswith("y = "):
            inner = formula[4:]
            return f"y = exp({inner})"
        return formula

    @staticmethod
    def _to_numpy(t):
        """Convert torch tensor to numpy array."""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)
