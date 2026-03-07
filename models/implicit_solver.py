# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

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
from models.sr_net import SRGBN
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
    """
    target_var_idx: int
    mode: str
    probe_loss_explicit: float = float('inf')
    probe_loss_implicit: float = float('inf')


class ImplicitSolver:
    """Handles implicit F(x,y)=0 formulation for SR.

    Compares explicit y=f(x) vs implicit F(x,y)=0 via short probes,
    and provides training/extraction methods for the implicit mode.

    Args:
        device: Computation device.
        probe_epochs: Number of epochs for mode probing.
        jacobian_weight: Weight for Jacobian norm regularizer.
    """

    def __init__(self, device='cpu', probe_epochs=10, jacobian_weight=0.1):
        self.device = device
        self.probe_epochs = probe_epochs
        self.jacobian_weight = jacobian_weight

    def probe_best_mode(self, algebra, X, y):
        """Compare explicit y=f(x) vs implicit F(x,y)=0 via short probes.

        Explicit: SRGBN(in_features=k) -> target y
        Implicit: SRGBN(in_features=k+1) -> target 0, with Jacobian regularizer

        Args:
            algebra: CliffordAlgebra for the problem.
            X: torch.Tensor [N, k] normalized inputs.
            y: torch.Tensor [N, 1] normalized targets.

        Returns:
            ImplicitFormulation with the better mode.
        """
        k = X.shape[1]
        N = X.shape[0]

        # Explicit probe
        explicit_loss = self._probe_explicit(algebra, X, y)

        # Implicit probe: augment data with y as extra variable
        Z = torch.cat([X, y], dim=-1)  # [N, k+1]

        # Build a larger algebra for k+1 variables if needed
        p, q, r = algebra.p, algebra.q, algebra.r
        n_needed = k + 1
        if p + q + r < n_needed:
            p_impl = min(n_needed, 6)
            impl_algebra = CliffordAlgebra(p_impl, q, r, device=self.device)
        else:
            impl_algebra = algebra

        implicit_loss = self._probe_implicit(impl_algebra, Z)

        # Pick better mode (implicit must be substantially better, not just
        # trivially zero from the F=0 solution)
        if np.isfinite(implicit_loss) and implicit_loss < explicit_loss * 0.8:
            mode = "implicit"
            logger.info(f"Implicit mode selected: loss {implicit_loss:.4f} vs explicit {explicit_loss:.4f}")
        else:
            mode = "explicit"
            logger.info(f"Explicit mode selected: loss {explicit_loss:.4f} vs implicit {implicit_loss:.4f}")

        return ImplicitFormulation(
            target_var_idx=k,  # y is the last augmented variable
            mode=mode,
            probe_loss_explicit=explicit_loss,
            probe_loss_implicit=implicit_loss,
        )

    def train_implicit(self, model, Z_data, algebra, epochs, lr):
        """Train F(Z)=0 where Z=[X,y].

        Loss = F(Z)^2 + lambda_jac / (||grad_Z F||^2 + eps) + sparsity

        The Jacobian term prevents the trivial F=0 solution.

        Args:
            model: SRGBN model with in_features=k+1.
            Z_data: torch.Tensor [N, k+1] augmented data.
            algebra: CliffordAlgebra.
            epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Trained model.
        """
        optimizer = RiemannianAdam(model.parameters(), lr=lr, algebra=algebra)
        target = torch.zeros(Z_data.shape[0], 1, device=self.device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            Z_grad = Z_data.detach().requires_grad_(True)
            pred = model(Z_grad)  # [N, 1]

            # F(Z) should be 0
            f_loss = F.mse_loss(pred, target)

            # Jacobian regularizer: prevent trivial F=0
            if Z_grad.grad is not None:
                Z_grad.grad = None
            grad_F = torch.autograd.grad(
                pred.sum(), Z_grad, create_graph=True, retain_graph=True
            )[0]  # [N, k+1]
            jac_norm_sq = (grad_F ** 2).sum(dim=-1).mean()
            jac_loss = self.jacobian_weight / (jac_norm_sq + 1e-8)

            sparsity = model.total_sparsity_loss()

            loss = f_loss + jac_loss + 0.01 * sparsity

            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        model.eval()
        return model

    def _probe_explicit(self, algebra, X, y):
        """Short explicit probe: train SRGBN to predict y from X."""
        model = SRGBN.single_rotor(algebra, X.shape[1], channels=4)
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
        """Short implicit probe: train SRGBN(k+1) to output 0.

        Reports the *combined* loss (F^2 + Jacobian penalty) rather than
        just MSE, to prevent the trivial F=0 solution from winning.
        Also validates that the learned function has non-trivial gradients.
        """
        model = SRGBN.single_rotor(algebra, Z.shape[1], channels=4)
        model = model.to(self.device)
        optimizer = RiemannianAdam(model.parameters(), lr=0.003, algebra=algebra)
        target = torch.zeros(Z.shape[0], 1, device=self.device)

        model.train()
        for _ in range(self.probe_epochs):
            optimizer.zero_grad()

            Z_grad = Z.detach().requires_grad_(True)
            pred = model(Z_grad)
            f_loss = F.mse_loss(pred, target)

            # Jacobian regularizer
            grad_F = torch.autograd.grad(
                pred.sum(), Z_grad, create_graph=True, retain_graph=True
            )[0]
            jac_norm_sq = (grad_F ** 2).sum(dim=-1).mean()
            jac_loss = self.jacobian_weight / (jac_norm_sq + 1e-8)

            loss = f_loss + jac_loss
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()

        # Report combined loss including Jacobian penalty to prevent
        # trivial F=0 from always winning the comparison
        model.eval()
        Z_grad = Z.detach().requires_grad_(True)
        pred = model(Z_grad)
        f_loss = F.mse_loss(pred, target)

        grad_F = torch.autograd.grad(pred.sum(), Z_grad, retain_graph=False)[0]
        jac_norm_sq = (grad_F ** 2).sum(dim=-1).mean()

        # If gradient is near-zero, the model learned the trivial solution
        # Penalize heavily
        if jac_norm_sq.item() < 0.01:
            logger.debug("Implicit probe: trivial solution (near-zero gradient)")
            return float('inf')

        combined_loss = f_loss.item() + self.jacobian_weight / (jac_norm_sq.item() + 1e-8)
        return combined_loss
