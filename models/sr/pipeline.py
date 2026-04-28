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

from models.sr.phases import CrossTermMixin, ExtractionMixin, PrepMixin, RefinementMixin
from models.sr.utils import evaluate_terms, make_lambdify_fn

logger = logging.getLogger(__name__)


# =========================================================================
# Dataclasses
# =========================================================================


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
    relationship_graph: object  # RelationshipGraph or None
    implicit_form: object  # ImplicitFormulation or None
    svd_Vt: np.ndarray
    svd_S: np.ndarray


@dataclass
class ConvergenceState:
    """Tracks convergence across feedback iterations.

    Used when feedback.enabled=True to decide whether to re-enter
    earlier phases based on per-term contribution analysis.
    """

    outer_iteration: int = 0
    r2_history: list = field(default_factory=list)
    stall_count: int = 0
    switched_to_implicit: bool = False
    terms_to_reextract: list = field(default_factory=list)


# =========================================================================
# Main class
# =========================================================================


class IterativeUnbender(PrepMixin, ExtractionMixin, CrossTermMixin, RefinementMixin):
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
        implicit_mode="auto",
        grouping_enabled=True,
        max_groups=4,
        svd_warmstart=True,
        soft_rejection_alpha=10.0,
        soft_rejection_threshold=0.01,
        mother_cross_threshold=0.01,
        basis_config=None,
        graph_guided=True,
        cross_edge_top_k=5,
        grouping_config=None,
        feedback_config=None,
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
        self.graph_guided = graph_guided
        self.cross_edge_top_k = cross_edge_top_k
        self.grouping_config = grouping_config or {}
        self.feedback_config = feedback_config or {}
        # Populated by basis expansion (if any) for formula assembly
        self._basis_result = None

    def run(self, X_norm, y_norm, x_mean, x_std, y_mean, y_std, var_names):
        """Run the full 4-phase iterative unbending pipeline.

        When feedback.enabled=True, wraps Phases 1-3 in an outer loop
        that monitors convergence and re-extracts low-contribution terms.

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
        y_orig = self._to_numpy(y_norm.squeeze(-1) * y_std.to(self.device) + y_mean.to(self.device))

        # Phase 0: Basis expansion (before linearity check)
        self._basis_result = None
        if self.basis_config.get("enabled", False):
            X_orig, y_orig, var_names = self._apply_basis_expansion(
                X_orig,
                y_orig,
                var_names,
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
            vnames = var_names if var_names else [f"x{i + 1}" for i in range(self.in_features)]
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

        ss_tot = np.sum((y_orig - y_orig.mean()) ** 2) + 1e-12

        # Feedback loop configuration
        fb = self.feedback_config
        feedback_enabled = fb.get("enabled", False)
        max_outer = fb.get("max_outer_iterations", 3) if feedback_enabled else 1
        reextract_threshold = fb.get("phase3_reextract_threshold", 0.3)
        cross_term_reeval = fb.get("cross_term_reeval", True)
        r2_stall_threshold = fb.get("r2_stall_threshold", 0.01)

        convergence = ConvergenceState()
        all_terms = []
        all_ops = []
        all_stages = []
        mother_cross_energy = 0.0
        residual_target = y_orig  # target for extraction (updated per outer iter)

        for outer_iter in range(max_outer):
            convergence.outer_iteration = outer_iter

            if outer_iter > 0:
                logger.info(
                    f"Feedback iteration {outer_iter}: "
                    f"re-extracting from residual (R2={convergence.r2_history[-1]:.4f})"
                )

            # Phase 1: Per-group extraction
            logger.info("Phase 1: Per-group iterative extraction")
            iter_terms = []
            iter_ops = []
            iter_stages = []

            # On re-entry, build residual from surviving terms
            if outer_iter > 0 and all_terms:
                y_hat_surviving = evaluate_terms(all_terms, X_orig)
                residual_target = y_orig - y_hat_surviving
                # Re-normalize for training
                res_t = torch.tensor(residual_target, dtype=torch.float32, device=self.device)
                res_mean = res_t.mean()
                res_std = res_t.std().clamp(min=1e-6)
                y_norm = ((res_t - res_mean) / res_std).unsqueeze(-1)

            for group_idx, group in enumerate(prep.groups):
                terms, stages = self._process_group(
                    group,
                    group_idx,
                    prep,
                    X_orig,
                    residual_target,
                    X_norm,
                    y_norm,
                )
                iter_terms.extend(terms)
                iter_ops.extend(["sub"] * len(terms))
                iter_stages.extend(stages)

            all_terms.extend(iter_terms)
            all_ops.extend(iter_ops)
            all_stages.extend(iter_stages)

            # Phase 2: Mother algebra cross-terms (graph-guided)
            if len(prep.groups) > 1 and outer_iter == 0:
                logger.info("Phase 2: Mother algebra cross-term discovery")
                cross_terms, cross_energy = self._mother_algebra_joint(
                    prep.groups,
                    X_orig,
                    y_orig,
                    X_norm,
                    prep,
                )
                all_terms.extend(cross_terms)
                all_ops.extend(["sub"] * len(cross_terms))
                mother_cross_energy = cross_energy

                # Feedback B: Cross-term re-evaluation of Phase 1 terms
                if feedback_enabled and cross_term_reeval and cross_terms:
                    all_terms, all_ops = self._reeval_after_cross_terms(
                        all_terms,
                        all_ops,
                        X_orig,
                        y_orig,
                        ss_tot,
                        reextract_threshold,
                    )
            elif outer_iter == 0:
                logger.info("Phase 2: Skipped (single group)")

            # Phase 3: SymPy refinement
            logger.info("Phase 3: Joint refinement")
            if all_terms:
                all_terms, all_ops = self._refine_all_terms(
                    all_terms,
                    X_orig,
                    y_orig,
                    all_ops,
                )

            # Compute R2
            y_hat = evaluate_terms(all_terms, X_orig)
            ss_res = np.sum((y_orig - y_hat) ** 2)
            r2_final = 1.0 - ss_res / ss_tot
            convergence.r2_history.append(r2_final)

            if not feedback_enabled:
                break

            # Feedback C: Refinement-driven re-extraction
            terms_before = len(all_terms)
            all_terms, all_ops = self._prune_low_contribution_terms(
                all_terms,
                all_ops,
                X_orig,
                y_orig,
                ss_tot,
                reextract_threshold,
            )
            terms_pruned = terms_before - len(all_terms)

            # Convergence check
            if terms_pruned == 0:
                logger.info(f"Feedback: no low-contribution terms, converged at R2={r2_final:.4f}")
                break

            if len(convergence.r2_history) >= 2:
                r2_delta = convergence.r2_history[-1] - convergence.r2_history[-2]
                if r2_delta < 0:
                    logger.info(f"Feedback: R2 decreased ({r2_delta:.4f}), stopping")
                    break
                if r2_delta < r2_stall_threshold:
                    logger.info(f"Feedback: R2 improvement {r2_delta:.4f} < threshold, stopping")
                    break

            logger.info(f"Feedback: pruned {terms_pruned} terms, re-entering Phase 1")

        # If explicit R2 is poor and implicit wasn't tried, try implicit fallback
        implicit_form = prep.implicit_form
        if (
            r2_final < 0.5
            and self.implicit_mode != "explicit"
            and len(prep.groups) == 1
            and (implicit_form is None or implicit_form.mode != "implicit")
        ):
            logger.info(f"Explicit R2={r2_final:.4f} < 0.5, trying implicit fallback")
            from models.sr.implicit import ImplicitFormulation

            fallback_form = ImplicitFormulation(
                target_var_idx=X_norm.shape[1],
                mode="implicit",
            )
            fallback_prep = _PrepResult(
                groups=prep.groups,
                relationship_graph=prep.relationship_graph,
                implicit_form=fallback_form,
                svd_Vt=prep.svd_Vt,
                svd_S=prep.svd_S,
            )
            impl_terms, impl_stages = self._process_group_implicit(
                prep.groups[0],
                0,
                fallback_prep,
                X_orig,
                y_orig,
                X_norm,
                y_norm,
            )
            if impl_terms:
                impl_y_hat = evaluate_terms(impl_terms, X_orig)
                impl_ss_res = np.sum((y_orig - impl_y_hat) ** 2)
                impl_r2 = 1.0 - impl_ss_res / ss_tot
                if impl_r2 > r2_final:
                    logger.info(f"Implicit fallback improved R2: {r2_final:.4f} -> {impl_r2:.4f}")
                    all_terms = impl_terms
                    all_ops = ["sub"] * len(impl_terms)
                    all_stages = impl_stages
                    r2_final = impl_r2
                    implicit_form = fallback_form

        # Assemble formula
        vnames = var_names if var_names else [f"x{i + 1}" for i in range(self.in_features)]
        formula = self._assemble_formula(
            all_terms,
            vnames,
            all_ops,
            implicit_form=implicit_form,
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

    # ------------------------------------------------------------------
    # Feedback helpers
    # ------------------------------------------------------------------

    def _prune_low_contribution_terms(self, all_terms, all_ops, X_orig, y_orig, ss_tot, threshold):
        """Remove terms whose relative contribution is below threshold.

        Contribution = |weight| * var(prediction) / total_variance.
        Returns pruned (terms, ops) lists.
        """
        if not all_terms:
            return all_terms, all_ops

        total_var = np.var(y_orig) + 1e-12
        surviving_terms = []
        surviving_ops = []

        for i, t in enumerate(all_terms):
            if t.fn is None:
                surviving_terms.append(t)
                surviving_ops.append(all_ops[i] if i < len(all_ops) else "sub")
                continue

            # Compute this term's prediction variance
            n_expected = t.fn.__code__.co_argcount
            n_vars = X_orig.shape[1]
            args = [X_orig[:, j] for j in range(min(n_vars, n_expected))]
            args.extend([np.zeros(X_orig.shape[0])] * (n_expected - len(args)))
            try:
                pred = np.asarray(t.fn(*args), dtype=np.float64)
                pred = np.broadcast_to(pred, (X_orig.shape[0],))
                term_contribution = abs(t.weight) * np.var(pred) / total_var
            except (ValueError, TypeError, OverflowError):
                term_contribution = 0.0

            if term_contribution >= threshold or not np.isfinite(term_contribution):
                surviving_terms.append(t)
                surviving_ops.append(all_ops[i] if i < len(all_ops) else "sub")
            else:
                logger.info(f"  Feedback: pruning term {i} (contribution={term_contribution:.4f} < {threshold})")

        return surviving_terms, surviving_ops

    def _reeval_after_cross_terms(self, all_terms, all_ops, X_orig, y_orig, ss_tot, threshold):
        """After Phase 2, re-evaluate Phase 1 term contributions.

        Cross-terms may explain variance that Phase 1 terms also claimed.
        Terms whose contribution dropped below threshold are removed.
        """
        if len(all_terms) <= 1:
            return all_terms, all_ops

        # Compute per-term contribution via leave-one-out R2 impact
        total_y_hat = evaluate_terms(all_terms, X_orig)
        total_var = np.var(y_orig) + 1e-12

        surviving_terms = []
        surviving_ops = []

        for i, t in enumerate(all_terms):
            # Leave this term out
            without = [tt for j, tt in enumerate(all_terms) if j != i]
            y_hat_without = evaluate_terms(without, X_orig)
            r2_without = 1.0 - np.sum((y_orig - y_hat_without) ** 2) / ss_tot
            r2_with = 1.0 - np.sum((y_orig - total_y_hat) ** 2) / ss_tot

            # Contribution = how much R2 drops without this term
            contribution = r2_with - r2_without

            if contribution >= threshold * 0.1 or contribution < 0:
                # Keep terms that contribute positively or that hurt R2 if removed
                surviving_terms.append(t)
                surviving_ops.append(all_ops[i] if i < len(all_ops) else "sub")
            else:
                logger.info(f"  Cross-term reeval: pruning term {i} (R2 contribution={contribution:.4f})")

        return surviving_terms, surviving_ops

    @staticmethod
    def _to_numpy(t):
        """Convert torch tensor to numpy array."""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)
