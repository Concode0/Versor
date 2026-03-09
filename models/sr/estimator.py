# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Scikit-learn compatible wrapper for Geometric Blade Network SR.

Follows SRBench API: fit(X, y), predict(X), model(est), complexity(est).
"""

import time

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin

from core.algebra import CliffordAlgebra
from models.sr.net import SRGBN
from models.sr.utils import make_lambdify_fn
from optimizers.riemannian import RiemannianAdam


class VersorSR(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Geometric Blade Network for symbolic regression.

    Parameters follow sklearn convention: all set in __init__, stored as attributes.
    """

    def __init__(
        self,
        p=4, q=0, r=0,
        hidden_channels=16,
        num_layers=2,
        num_rotors=4,
        epochs=40,
        lr=0.001,
        batch_size=128,
        sparsity_weight=0.01,
        random_state=42,
        max_time=600,
        use_decomposition=True,
        max_bivector_norm=10.0,
        basis_config=None,
    ):
        self.p = p
        self.q = q
        self.r = r
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rotors = num_rotors
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.sparsity_weight = sparsity_weight
        self.random_state = random_state
        self.max_time = max_time
        self.use_decomposition = use_decomposition
        self.max_bivector_norm = max_bivector_norm
        self.basis_config = basis_config

    def fit(self, X, y):
        """Train GBN on (X, y).

        Args:
            X: np.ndarray [N, k] input features.
            y: np.ndarray [N] target values.

        Returns:
            self
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        # Store normalization stats
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ < 1e-6] = 1e-6
        self.y_mean_ = y.mean()
        self.y_std_ = max(y.std(), 1e-6)

        X_norm = (X - self.x_mean_) / self.x_std_
        y_norm = (y - self.y_mean_) / self.y_std_

        X_t = torch.from_numpy(X_norm)
        y_t = torch.from_numpy(y_norm).unsqueeze(-1)

        n_vars = X.shape[1]
        algebra = CliffordAlgebra(p=self.p, q=self.q, r=self.r, device="cpu")

        self.model_ = SRGBN(
            algebra=algebra,
            in_features=n_vars,
            channels=self.hidden_channels,
            num_layers=self.num_layers,
            use_decomposition=self.use_decomposition,
        )
        self.algebra_ = algebra

        optimizer = RiemannianAdam(
            self.model_.parameters(),
            lr=self.lr,
            algebra=algebra,
            max_bivector_norm=self.max_bivector_norm,
        )
        criterion = torch.nn.MSELoss()

        N = len(X_t)
        t_start = time.time()

        self.model_.train()
        for epoch in range(self.epochs):
            if time.time() - t_start > self.max_time:
                break

            perm = torch.randperm(N)
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                x_b = X_t[idx]
                y_b = y_t[idx]

                optimizer.zero_grad()
                pred = self.model_(x_b)
                loss = criterion(pred, y_b)
                loss = loss + self.sparsity_weight * self.model_.total_sparsity_loss()
                loss.backward()
                optimizer.step()

        self.model_.eval()

        # Extract formula via IterativeUnbender
        self.formula_ = ""
        self._formula_fn = None
        try:
            from models.sr.unbender import IterativeUnbender
            import sympy

            x_mean_t = torch.from_numpy(self.x_mean_)
            x_std_t = torch.from_numpy(self.x_std_)
            y_mean_t = torch.tensor(self.y_mean_)
            y_std_t = torch.tensor(self.y_std_)

            unbender = IterativeUnbender(
                in_features=n_vars, device="cpu",
                max_stages=3, stage_epochs=self.epochs // 3,
                implicit_mode='auto', svd_warmstart=True,
                basis_config=self.basis_config or {},
            )
            var_names = [f"x{i + 1}" for i in range(n_vars)]
            result = unbender.run(
                X_t, y_t, x_mean_t, x_std_t, y_mean_t, y_std_t,
                var_names=var_names,
            )
            self.formula_ = result.formula

            # Build callable from extracted terms for predict()
            if result.all_terms:
                syms = [sympy.Symbol(f"x{i+1}") for i in range(n_vars)]
                combined = sympy.Integer(0)
                for t in result.all_terms:
                    if t.expr is not None:
                        combined += t.weight * t.expr
                self._formula_fn = make_lambdify_fn(syms, combined)
        except Exception:
            self.formula_ = "extraction_failed"

        return self

    def predict(self, X):
        """Predict y from X using the extracted formula.

        Falls back to model forward pass if formula extraction failed.

        Args:
            X: np.ndarray [N, k] input features.

        Returns:
            np.ndarray [N] predictions.
        """
        X = np.asarray(X, dtype=np.float32)

        # Try formula-based prediction first (reflects actual SR quality)
        if hasattr(self, '_formula_fn') and self._formula_fn is not None:
            try:
                X_norm = (X - self.x_mean_) / self.x_std_
                n_vars = X.shape[1]
                args = [X_norm[:, i] for i in range(n_vars)]
                pred_norm = self._formula_fn(*args)
                pred_norm = np.broadcast_to(
                    np.asarray(pred_norm, dtype=np.float64),
                    (X.shape[0],),
                ).copy()
                if np.all(np.isfinite(pred_norm)):
                    return pred_norm * self.y_std_ + self.y_mean_
            except Exception:
                pass

        # Fallback: model forward pass
        X_norm = (X - self.x_mean_) / self.x_std_
        X_t = torch.from_numpy(X_norm)

        self.model_.eval()
        with torch.no_grad():
            pred_norm = self.model_(X_t).squeeze(-1).numpy()

        return pred_norm * self.y_std_ + self.y_mean_


def model(est, X=None) -> str:
    """Return sympy-compatible formula string from fitted estimator.

    SRBench API function.
    """
    return getattr(est, "formula_", "")


def complexity(est) -> int:
    """Return expression complexity (number of sympy tree nodes).

    SRBench API function.
    """
    try:
        import sympy
        formula = getattr(est, "formula_", "")
        formula = formula.replace("y = ", "")
        if not formula or formula == "extraction_failed":
            return 999
        expr = sympy.parse_expr(formula)
        return sympy.count_ops(expr) + len(expr.free_symbols)
    except Exception:
        return 999
