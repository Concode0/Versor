# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Implicit function formulation F(x,y)=0 for symbolic regression.

For rational functions and deep composites, the explicit y=f(x)
formulation is often unstable (poles, singularities). The implicit
form F(x,y)=0 treats all variables symmetrically, allowing the
rotor to discover bilinear and higher-order relationships directly.

Example: y = x/(1+x) has a pole, but F(x,y) = xy + y - x = 0
is a simple bilinear -- the rotor discovers e_x ^ e_y immediately.
"""

import logging
from dataclasses import dataclass

import numpy as np
import sympy
import torch
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from models.sr.net import SRGBN
from models.sr.utils import safe_sympy_solve
from optimizers.riemannian import RiemannianAdam

logger = logging.getLogger(__name__)


@dataclass
class ImplicitFormulation:
    """Result of implicit/explicit mode probing.

    Attributes:
        target_var_idx: Which variable is "y" (usually last augmented dim).
        mode: "implicit" or "explicit".
        probe_loss_explicit: Best explicit probe loss.
        probe_loss_implicit: Best implicit probe loss.
        curvature_score: Manifold curvature from GeodesicFlow.
        involution_score: Odd-grade energy fraction from SymmetryDetector.
        null_dim_count: Number of near-null basis directions.
        codimension: ambient_dim - intrinsic_dim.
        geometric_implicit_score: Combined geometric score for implicit mode.
    """
    target_var_idx: int
    mode: str
    probe_loss_explicit: float = float('inf')
    probe_loss_implicit: float = float('inf')
    curvature_score: float = 0.0
    involution_score: float = 0.0
    null_dim_count: int = 0
    codimension: int = 0
    geometric_implicit_score: float = 0.0


class ImplicitSolver:
    """Handles implicit F(x,y)=0 formulation for SR.

    Compares explicit y=f(x) vs implicit F(x,y)=0 via short probes,
    and provides training/extraction methods for the implicit mode.

    Args:
        device: Computation device.
        probe_epochs: Number of epochs for mode probing.
        jacobian_weight: Weight for Jacobian norm regularizer.
    """

    def __init__(self, device='cpu', probe_epochs=50, jacobian_weight=0.1):
        self.device = device
        self.probe_epochs = probe_epochs
        self.jacobian_weight = jacobian_weight

    def probe_best_mode(self, algebra, X, y, geometric_report=None):
        """Compare explicit y=f(x) vs implicit F(x,y)=0 via short probes
        combined with geometric criteria from the relationship graph.

        Geometric criteria for implicit mode:
        1. High curvature (> 0.3): data on a curved manifold that can't
           be parameterized as y=f(x).
        2. High involution symmetry (> 0.3): F(x,y) has even/odd grade
           structure suggesting bilinear or higher-order implicit form.
        3. Manifold codimension > 0: data lives on a proper submanifold,
           meaning an implicit constraint F=0 exists.
        4. Null directions present: degenerate dimensions indicate
           constraint surfaces.

        Args:
            algebra: CliffordAlgebra for the problem.
            X: torch.Tensor [N, k] normalized inputs.
            y: torch.Tensor [N, 1] normalized targets.
            geometric_report: Optional dict from RelationshipGraph with
                curvature, involution_symmetry, intrinsic_dim, ambient_dim,
                null_directions, continuous_symmetry_dim.

        Returns:
            ImplicitFormulation with the better mode.
        """
        k = X.shape[1]

        # Explicit probe
        explicit_loss = self._probe_explicit(algebra, X, y)

        # Implicit probe: augment data with y as extra variable
        Z = torch.cat([X, y], dim=-1)  # [N, k+1]

        # Build implicit algebra: add 1 to p for the y variable
        p, q, r = algebra.p, algebra.q, algebra.r
        impl_algebra = CliffordAlgebra(p + 1, q, r, device=self.device)

        implicit_loss = self._probe_implicit(impl_algebra, Z)

        # Compute geometric implicit score
        geo_score, geo_details = self._compute_geometric_implicit_score(
            geometric_report,
        )

        # Combined decision:
        # 1. Very strong geometric signal + implicit loss is at least finite → implicit
        # 2. Moderate geometric signal lowers the loss threshold → implicit
        # 3. Original loss-based criterion as fallback (no geometric info needed)
        # 4. Otherwise → explicit
        #
        # The geometric score alone is not sufficient because curvature and
        # codimension are common in normal data.  The score must be very high
        # (> 0.7) to override, or it relaxes the loss-based threshold.
        if (geo_score > 0.7
                and np.isfinite(implicit_loss)
                and implicit_loss < explicit_loss):
            mode = "implicit"
            logger.info(
                f"Implicit mode selected (strong geometric: {geo_score:.3f}): "
                f"loss {implicit_loss:.4f} vs explicit {explicit_loss:.4f}"
            )
        elif (geo_score > 0.3
              and np.isfinite(implicit_loss)
              and explicit_loss > 0.3
              and implicit_loss < explicit_loss * 0.7):
            mode = "implicit"
            logger.info(
                f"Implicit mode selected (geometric+loss: geo={geo_score:.3f}): "
                f"loss {implicit_loss:.4f} vs explicit {explicit_loss:.4f}"
            )
        elif (np.isfinite(implicit_loss)
              and explicit_loss > 0.5
              and implicit_loss < explicit_loss * 0.5):
            mode = "implicit"
            logger.info(
                f"Implicit mode selected (loss-based fallback): "
                f"loss {implicit_loss:.4f} vs explicit {explicit_loss:.4f}"
            )
        else:
            mode = "explicit"
            logger.info(
                f"Explicit mode selected (geo={geo_score:.3f}): "
                f"loss {explicit_loss:.4f} vs implicit {implicit_loss:.4f}"
            )

        return ImplicitFormulation(
            target_var_idx=k,
            mode=mode,
            probe_loss_explicit=explicit_loss,
            probe_loss_implicit=implicit_loss,
            curvature_score=geo_details.get("curvature", 0.0),
            involution_score=geo_details.get("involution", 0.0),
            null_dim_count=geo_details.get("null_count", 0),
            codimension=geo_details.get("codimension", 0),
            geometric_implicit_score=geo_score,
        )

    def _compute_geometric_implicit_score(self, geometric_report):
        """Compute geometric criteria for implicit mode.

        Returns:
            (score: float, details: dict) where score is in [0, 1].
        """
        if geometric_report is None:
            return 0.0, {}

        curvature = geometric_report.get("curvature", 0.0)
        involution = geometric_report.get("involution_symmetry", 0.0)
        intrinsic_dim = geometric_report.get("intrinsic_dim", 0)
        ambient_dim = geometric_report.get("ambient_dim", 0)
        null_dirs = geometric_report.get("null_directions", [])

        codimension = max(0, ambient_dim - intrinsic_dim)
        null_count = len(null_dirs)

        # Involution score: only meaningful when it indicates mixed
        # even/odd grade structure (0.3-0.7 range).  Pure grade-1
        # embedded data always gives involution ≈ 1.0 (all energy
        # in odd grades), which is uninformative.  Treat near-0 or
        # near-1 as no signal.
        invol_signal = 0.0
        if 0.15 < involution < 0.85:
            invol_signal = min(involution / 0.3, 1.0)

        # Weighted combination:
        # - High curvature → manifold can't be flattened to y=f(x)
        # - Mixed involution → bilinear/implicit structure
        # - Positive codimension → proper submanifold (constraint exists)
        # - Null directions → degenerate constraint surfaces
        score = (0.35 * min(curvature / 0.5, 1.0)
                 + 0.20 * invol_signal
                 + 0.25 * float(codimension > 0)
                 + 0.20 * float(null_count > 0))

        details = {
            "curvature": curvature,
            "involution": involution,
            "codimension": codimension,
            "null_count": null_count,
        }

        logger.info(
            f"Geometric implicit score: {score:.3f} "
            f"(curv={curvature:.3f}, invol={involution:.3f}, "
            f"codim={codimension}, nulls={null_count})"
        )

        return score, details

    def train_implicit(self, model, Z_data, algebra, epochs, lr,
                       geometric_report=None):
        """Train F(Z)=0 where Z=[X,y] with warmup + Eikonal loss.

        Phase 1 (warmup): maximize gradient norm to escape trivial F==0.
        Phase 2 (main): Eikonal-normalized loss F/||gradF|| -> 0.

        If geometric_report shows high curvature, the warmup phase is
        extended to give the optimizer more time to discover the manifold
        structure before switching to Eikonal convergence.

        Args:
            model: SRGBN model with in_features=k+1.
            Z_data: torch.Tensor [N, k+1] augmented data.
            algebra: CliffordAlgebra.
            epochs: Training epochs.
            lr: Learning rate.
            geometric_report: Optional dict from RelationshipGraph.

        Returns:
            Trained model.
        """
        optimizer = RiemannianAdam(model.parameters(), lr=lr, algebra=algebra)
        target = torch.zeros(Z_data.shape[0], 1, device=self.device)
        warmup_epochs = max(epochs // 4, 5)

        # Extend warmup for high-curvature manifolds
        if geometric_report is not None:
            curvature = geometric_report.get("curvature", 0.0)
            if curvature > 0.5:
                warmup_epochs = max(epochs // 3, 10)
                logger.info(
                    f"High curvature ({curvature:.3f}): "
                    f"extended implicit warmup to {warmup_epochs} epochs"
                )

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            Z_grad = Z_data.detach().requires_grad_(True)
            pred = model(Z_grad)  # [N, 1]

            # Only build higher-order graph when needed (Phase 2 Eikonal).
            # Warmup only needs grad magnitudes, not gradients-of-gradients.
            need_higher_order = epoch >= warmup_epochs
            grad_F = torch.autograd.grad(
                pred.sum(), Z_grad,
                create_graph=need_higher_order,
                retain_graph=True,
            )[0]  # [N, k+1]
            jac_norm_sq = (grad_F ** 2).sum(dim=-1).mean()

            sparsity = model.total_sparsity_loss()

            if epoch < warmup_epochs:
                # Warmup: escape trivial F==0
                output_mag = pred.pow(2).mean()
                log_eps = max(1e-8, 1e-4 * jac_norm_sq.detach().item())
                loss = (-self.jacobian_weight * torch.log(jac_norm_sq + log_eps)
                        - 0.01 * torch.log(output_mag + log_eps)
                        + 0.01 * sparsity)
            else:
                # Eikonal-normalized loss
                per_sample_grad_norm = torch.sqrt(
                    (grad_F ** 2).sum(dim=-1, keepdim=True) + 1e-8
                )
                log_eps = max(1e-8, 1e-4 * jac_norm_sq.detach().item())
                normalized_pred = pred / (per_sample_grad_norm + 1e-8)
                eikonal_loss = F.mse_loss(normalized_pred, target)
                jac_loss = -self.jacobian_weight * torch.log(jac_norm_sq + log_eps)
                loss = eikonal_loss + jac_loss + 0.01 * sparsity

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        model.eval()
        return model

    def _probe_explicit(self, algebra, X, y):
        """Short explicit probe: train SRGBN to predict y from X."""
        model = SRGBN.single_rotor(algebra, X.shape[1], channels=16)
        model = model.to(self.device)
        optimizer = RiemannianAdam(model.parameters(), lr=0.003, algebra=algebra)

        model.train()
        for _ in range(self.probe_epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, y)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X)
            final_loss = F.mse_loss(pred, y).item()

        return final_loss

    def _probe_implicit(self, algebra, Z):
        """Short implicit probe with two-phase training.

        Phase 1 (warmup): maximize gradient norm to escape the trivial F==0
        state where both F and gradF are zero (so Eikonal loss gives no signal).
        Phase 2 (main): Eikonal-normalized loss F/||gradF|| -> 0 for non-trivial
        convergence.

        Reports raw f_loss for comparison with explicit probe.
        """
        model = SRGBN.single_rotor(algebra, Z.shape[1], channels=16)
        model = model.to(self.device)

        # Break the F==0 dead gradient: SRGBN initializes grade0_bias to zeros,
        # making output == 0. With implicit target = 0, gradient = 2*pred*dpred/dtheta = 0.
        # Non-zero grade0_bias and grade1_proj give a non-trivial starting point.
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'grade0_bias'):
                    torch.nn.init.normal_(m.grade0_bias, std=1.0)
                if hasattr(m, 'grade1_proj'):
                    torch.nn.init.xavier_normal_(m.grade1_proj.weight)

        optimizer = RiemannianAdam(model.parameters(), lr=0.003, algebra=algebra)
        target = torch.zeros(Z.shape[0], 1, device=self.device)
        warmup_epochs = max(self.probe_epochs // 3, 5)

        model.train()
        for epoch in range(self.probe_epochs):
            optimizer.zero_grad()

            Z_grad = Z.detach().requires_grad_(True)
            pred = model(Z_grad)

            need_higher_order = epoch >= warmup_epochs
            grad_F = torch.autograd.grad(
                pred.sum(), Z_grad,
                create_graph=need_higher_order,
                retain_graph=True,
            )[0]
            jac_norm_sq = (grad_F ** 2).sum(dim=-1).mean()

            if epoch < warmup_epochs:
                # Phase 1: maximize gradient norm to escape F==0
                output_mag = pred.pow(2).mean()
                log_eps = max(1e-8, 1e-4 * jac_norm_sq.detach().item())
                loss = (-self.jacobian_weight * torch.log(jac_norm_sq + log_eps)
                        - 0.01 * torch.log(output_mag + log_eps))
            else:
                # Phase 2: Eikonal-normalized loss
                per_sample_grad_norm = torch.sqrt(
                    (grad_F ** 2).sum(dim=-1, keepdim=True) + 1e-8
                )
                log_eps = max(1e-8, 1e-4 * jac_norm_sq.detach().item())
                normalized_pred = pred / (per_sample_grad_norm + 1e-8)
                eikonal_loss = F.mse_loss(normalized_pred, target)
                jac_loss = -self.jacobian_weight * torch.log(jac_norm_sq + log_eps)
                loss = eikonal_loss + jac_loss

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        # Evaluate final quality
        model.eval()
        Z_grad = Z.detach().requires_grad_(True)
        pred = model(Z_grad)
        f_loss = F.mse_loss(pred, target)

        grad_F = torch.autograd.grad(pred.sum(), Z_grad, retain_graph=False)[0]
        jac_norm_sq = (grad_F ** 2).sum(dim=-1).mean()

        # If gradient is near-zero, the model learned the trivial solution
        if jac_norm_sq.item() < 0.001:
            logger.info(f"Implicit probe: trivial (jac_norm_sq={jac_norm_sq.item():.6f}, "
                        f"f_loss={f_loss.item():.6f})")
            return float('inf')

        # Report raw f_loss (comparable to explicit MSE)
        logger.info(f"Implicit probe: f_loss={f_loss.item():.4f}, "
                    f"jac_norm_sq={jac_norm_sq.item():.4f}")
        return f_loss.item()

    def extract_implicit_formula(self, model, algebra, var_names, target_var_idx):
        """Extract F(x,y)=0 via translate_implicit, then solve for y.

        Fallback chain: sympy.solve -> sympy.solveset -> return implicit form.
        """
        from models.sr.translator import RotorTranslator

        translator = RotorTranslator(algebra)
        terms = translator.translate_implicit(model, target_var_idx)

        if not terms:
            return None

        # Build F expression
        F_expr = sympy.Integer(0)
        for t in terms:
            F_expr += t.weight * t.expr

        y_sym = sympy.Symbol("y")

        # Attempt 1: sympy.solve
        sol = safe_sympy_solve(F_expr, y_sym)
        if sol is not None:
            return sol

        # Attempt 2: sympy.solveset
        try:
            sol_set = sympy.solveset(F_expr, y_sym, domain=sympy.S.Reals)
            if sol_set.is_FiniteSet and len(sol_set) > 0:
                return list(sol_set)[0]
        except (NotImplementedError, ValueError, TypeError):
            pass

        return F_expr  # Return implicit form if all solves fail
