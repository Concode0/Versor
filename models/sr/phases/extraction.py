# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Phase 1: Per-group iterative extraction mixin.

Contains single-rotor extraction, training, GA orthogonal elimination,
coherence/curvature measurement, and related helpers.
"""

import logging

import numpy as np
import sympy
import torch
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from core.analysis import GeodesicFlow, MetricSearch
from models.sr.translator import RotorTranslator, RotorTerm
from models.sr.net import SRGBN
from models.sr.utils import (
    safe_sympy_solve,
    safe_float,
    standardize,
    subsample,
    make_lambdify_fn,
)
from optimizers.riemannian import RiemannianAdam

logger = logging.getLogger(__name__)


class ExtractionMixin:
    """Phase 1 methods: per-group extraction, training, GA elimination."""

    def _process_group(self, group, group_idx, prep, X_orig, y_orig, X_norm, y_norm):
        """Single-rotor iterative extraction for one variable group.

        Dispatches to implicit or explicit extraction based on prep result.

        Returns:
            (list[RotorTerm], list[StageResult])
        """
        # If implicit mode selected, use implicit extraction for this group
        if prep.implicit_form is not None and prep.implicit_form.mode == "implicit":
            terms, stages = self._process_group_implicit(
                group,
                group_idx,
                prep,
                X_orig,
                y_orig,
                X_norm,
                y_norm,
            )
            return terms, stages

        return self._process_group_explicit(
            group,
            group_idx,
            prep,
            X_orig,
            y_orig,
            X_norm,
            y_norm,
        )

    def _process_group_implicit(self, group, group_idx, prep, X_orig, y_orig, X_norm, y_norm):
        """Implicit extraction: train F(x,y)=0, extract via sympy.solve.

        Returns:
            (list[RotorTerm], list[StageResult])
        """
        from models.sr.implicit import ImplicitSolver
        from models.sr.pipeline import StageResult

        algebra = group.algebra
        var_indices = group.var_indices
        n_group_vars = len(var_indices)

        # Build augmented algebra (k+1 variables)
        p, q, r = group.signature
        impl_algebra = CliffordAlgebra(p + 1, q, r, device=self.device)

        # Augmented data Z = [X_group_norm, y_norm]
        X_group_norm = standardize(torch.tensor(X_orig[:, var_indices], dtype=torch.float32, device=self.device))
        Z = torch.cat([X_group_norm, y_norm], dim=-1)  # [N, k+1]

        # Train implicit model
        solver = ImplicitSolver(device=self.device)

        auto = SRGBN.auto_config(Z.shape[0], n_group_vars + 1, impl_algebra.dim)
        model = SRGBN.single_rotor(
            impl_algebra,
            n_group_vars + 1,
            channels=max(auto["channels"], 16),
        )
        model = model.to(self.device)

        # Break F==0 dead gradient (see implicit_solver._probe_implicit)
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'grade0_bias'):
                    torch.nn.init.normal_(m.grade0_bias, std=1.0)

        model = solver.train_implicit(
            model,
            Z,
            impl_algebra,
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
            logger.info(f"  Group {group_idx}: implicit extraction found no terms, falling back to explicit")
            return self._process_group_explicit(
                group,
                group_idx,
                prep,
                X_orig,
                y_orig,
                X_norm,
                y_norm,
            )

        # Build F expression and solve for y
        F_expr = sympy.Integer(0)
        for t in impl_terms:
            F_expr += t.weight * t.expr

        y_sym = sympy.Symbol("y")
        var_syms = [sympy.Symbol(f"x{i + 1}") for i in range(n_group_vars)]

        explicit_expr = safe_sympy_solve(F_expr, y_sym)

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
            from models.sr.numerics import safe_simplify

            n_terms = len(sympy.Add.make_args(explicit_expr))
            if n_terms <= 20:
                explicit_expr = safe_simplify(explicit_expr)
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
        from models.sr.utils import safe_evaluate_term

        r2 = 0.0
        if result_term.fn is not None:
            y_hat = safe_evaluate_term(result_term.fn, X_orig, var_indices)
            if y_hat is not None:
                ss_tot = np.sum((y_orig - y_orig.mean()) ** 2) + 1e-12
                ss_res = np.sum((y_orig - y_hat) ** 2)
                r2 = 1.0 - ss_res / ss_tot
            logger.info(f"  Group {group_idx}: implicit extraction R2={r2:.4f}")

        stage = StageResult(
            stage_idx=0,
            signature=group.signature,
            terms=[result_term],
            fitted_values=np.zeros(len(y_orig)),
            residual_before=y_orig,
            residual_after=np.zeros(len(y_orig)),
            curvature_before=0.5,
            curvature_after=0.0,
            coherence_before=0.5,
            coherence_after=0.5,
            rotor_planes=[],
            accepted=True,
            group_idx=group_idx,
        )

        return [result_term], [stage]

    def _process_group_explicit(self, group, group_idx, prep, X_orig, y_orig, X_norm, y_norm):
        """Explicit single-rotor iterative extraction for one variable group.

        Uses the relationship graph (if available) to guide extraction
        ordering: strongest edges are extracted first, and rotor bivectors
        are biased toward the target plane for faster convergence.

        Returns:
            (list[RotorTerm], list[StageResult])
        """
        from models.sr.pipeline import StageResult, OrthogonalEliminationResult

        algebra = group.algebra
        var_indices = group.var_indices
        n_group_vars = len(var_indices)

        # Data for this group
        X_group = X_orig[:, var_indices]
        X_group_norm = standardize(torch.tensor(X_group, dtype=torch.float32, device=self.device))

        # Build residual multivector for GA elimination
        residual = y_orig.copy()
        residual_mv = (
            algebra.embed_vector(
                torch.tensor(
                    np.column_stack([X_group, residual.reshape(-1, 1)]),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            if algebra.n >= n_group_vars + 1
            else None
        )

        prev_coherence = safe_float(
            self._measure_coherence(X_group, residual, algebra),
            0.5,
        )

        # Graph-guided extraction ordering
        extraction_order = []
        if self.graph_guided and prep.relationship_graph is not None:
            group_edges = prep.relationship_graph.edges_for_group(group_idx)
            extraction_order = self._plan_extraction_order(
                group_edges,
                group,
            )
            if extraction_order:
                logger.info(f"  Group {group_idx}: graph-guided order with {len(extraction_order)} target planes")

        terms = []
        stages = []
        ss_tot = np.sum((y_orig - y_orig.mean()) ** 2) + 1e-12

        for stage_idx in range(self.max_stages):
            r2_current = 1.0 - np.sum(residual**2) / ss_tot
            if r2_current >= self.r2_target:
                logger.info(f"  Group {group_idx} R2={r2_current:.6f} >= target")
                break

            # Build single-rotor SRGBN with adequate capacity
            N_group = X_group.shape[0]
            auto = SRGBN.auto_config(N_group, n_group_vars, algebra.dim)
            channels = max(auto["channels"], 8)
            model = SRGBN.single_rotor(
                algebra,
                n_group_vars,
                channels=channels,
            )
            model = model.to(self.device)

            # Warm-start: prefer graph-guided plane bias, fall back to SVD
            plane_biased = False
            if self.graph_guided and stage_idx < len(extraction_order):
                target = extraction_order[stage_idx]
                plane_biased = self._bias_rotor_to_plane(
                    model,
                    target[0],
                    target[1],
                    target[2],
                    algebra,
                )
            if not plane_biased and self.svd_warmstart and group.svd_Vt is not None:
                model.svd_warmstart(group.svd_Vt, algebra)

            # Normalize residual for training
            residual_t = torch.tensor(residual, dtype=torch.float32, device=self.device).unsqueeze(-1)
            res_mean = residual_t.mean()
            res_std = residual_t.std().clamp(min=1e-8)
            residual_norm = (residual_t - res_mean) / res_std

            # Probe curvature
            probe_curv = safe_float(
                self._measure_curvature(X_group, residual, algebra),
                0.5,
            )

            # Train with MSE-primary objective for single-rotor extraction
            model, curv_after, coh_after = self._train_single_rotor(
                model,
                X_group_norm,
                residual_norm,
                algebra,
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
                    residual_mv,
                    blade,
                    algebra,
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
                self._measure_coherence(X_group, new_residual, algebra),
                0.5,
            )
            degradation = prev_coherence - new_coh

            ss_tot_local = np.sum((residual - residual.mean()) ** 2) + 1e-12
            ss_res_local = np.sum(new_residual**2)
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

    def _plan_extraction_order(self, group_edges, group):
        """Plan extraction order from strongest to weakest edges.

        Converts global variable indices in edges to group-local indices
        and returns a sorted list of (local_var_i, local_var_j, edge_type)
        tuples for biasing rotor initialization.

        Args:
            group_edges: list[VariableEdge] for this group, sorted by strength.
            group: VariableGroup with var_indices.

        Returns:
            list of (local_i, local_j, edge_type) tuples.
        """
        g2l = {gi: li for li, gi in enumerate(group.var_indices)}
        result = []
        for e in group_edges:
            li = g2l.get(e.var_i)
            lj = g2l.get(e.var_j)
            if li is not None and lj is not None:
                result.append((li, lj, e.edge_type))
        return result

    def _bias_rotor_to_plane(self, model, var_i, var_j, edge_type, algebra):
        """Initialize rotor bivector weights to favor a specific plane.

        Sets the e_{var_i} ^ e_{var_j} component higher than others,
        giving the optimizer a warm start toward the expected interaction.

        Args:
            model: SRGBN single-rotor model.
            var_i: Local variable index (within group).
            var_j: Local variable index (within group).
            edge_type: "elliptic", "hyperbolic", or "parabolic".
            algebra: CliffordAlgebra for this group.

        Returns:
            True if bias was applied, False otherwise.
        """
        if var_i >= algebra.n or var_j >= algebra.n:
            return False

        # Find the RotorLayer's bivector parameter
        rotor_layer = None
        for m in model.modules():
            if hasattr(m, 'bivector') and isinstance(getattr(m, 'bivector', None), torch.nn.Parameter):
                rotor_layer = m
                break

        if rotor_layer is None:
            return False

        # Compute the bivector basis index for e_i ^ e_j
        bv_target = (1 << var_i) | (1 << var_j)

        # Find position of this bivector in the grade-2 subset
        grade2_mask = algebra.grade_masks[2]
        grade2_indices = grade2_mask.nonzero(as_tuple=True)[0]
        target_pos = None
        for pos, idx in enumerate(grade2_indices):
            if int(idx) == bv_target:
                target_pos = pos
                break

        if target_pos is None:
            return False

        with torch.no_grad():
            bv = rotor_layer.bivector  # [C, n_bv] or [n_bv]
            # Set initial angle based on edge type
            init_angle = 0.3  # moderate rotation for elliptic
            if edge_type == "hyperbolic":
                init_angle = 0.2  # smaller for boosts (exponential growth)
            elif edge_type == "parabolic":
                init_angle = 0.5  # larger for shears (linear effect)

            if bv.ndim == 2:
                # Per-channel: set target plane for all channels
                bv[:, target_pos] = init_angle
            elif bv.ndim == 1:
                bv[target_pos] = init_angle

        logger.info(f"    Biased rotor to plane e{var_i}^e{var_j} ({edge_type}, angle={init_angle:.2f})")
        return True

    def _orthogonal_eliminate(self, data_mv, blade, algebra):
        """Soft GA rejection: preserve subtle terms near threshold.

        Instead of hard rejection (data - proj), uses sigmoid gating
        so components >> threshold are fully eliminated while components
        << threshold are fully preserved.
        """
        from models.sr.pipeline import OrthogonalEliminationResult

        proj = algebra.blade_project(data_mv, blade)
        proj_energy = proj.pow(2).sum(dim=-1)

        # Soft sigmoid mask
        soft_mask = torch.sigmoid(self.soft_rejection_alpha * (proj_energy - self.soft_rejection_threshold))
        rejected = data_mv - soft_mask.unsqueeze(-1) * proj

        proj_energy_total = proj_energy.sum().item()
        rej_energy = rejected.pow(2).sum().item()

        # Fraction preserved (borderline terms kept)
        near_threshold = (proj_energy > self.soft_rejection_threshold * 0.5) & (
            proj_energy < self.soft_rejection_threshold * 2.0
        )
        preserved = (1.0 - soft_mask[near_threshold]).mean().item() if near_threshold.any() else 1.0

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
        decay = 0.7**stage_idx
        channels = max(4, int(auto["channels"] * decay))
        num_layers = auto["num_layers"]

        model = SRGBN(
            algebra=algebra,
            in_features=self.in_features,
            channels=channels,
            num_layers=num_layers,
        )
        return model.to(self.device)

    def _train_stage(self, model, X_norm, residual_norm, algebra):
        """Train a stage model with curvature-primary objective.

        Returns:
            (model, final_curvature, final_coherence)
        """
        gf = GeodesicFlow(algebra, k=self.geodesic_k)
        optimizer = RiemannianAdam(model.parameters(), lr=self.stage_lr, algebra=algebra)
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
        optimizer = RiemannianAdam(model.parameters(), lr=self.stage_lr, algebra=algebra)
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
            x_cols = data[:, : n - 1]
            data = torch.cat([x_cols, residual_col], dim=-1)
        elif d < n:
            pad = torch.zeros(data.shape[0], n - d, device=self.device)
            data = torch.cat([data, pad], dim=-1)

        return data
