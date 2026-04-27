# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Symbolic Regression Task (SRBench / PMLB).

Learns symbolic expressions via iterative geometric unbending:
  1. Probe residual metric signature
  2. Train stage SRGBN with curvature-primary objective
  3. Extract terms via rotor plane analysis
  4. Subtract, re-probe, repeat
  5. Joint refinement + sympy simplification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from datalib.symbolic_regression import get_sr_loaders, get_sr_raw_splits, get_dataset_ids, _fetch_pmlb_data
from models.sr import SRGBN
from models.sr.unbender import IterativeUnbender
from log import get_logger

logger = get_logger(__name__)


class SRTask(BaseTask):
    """Symbolic Regression via Iterative Geometric Unbending.

    Uses the IterativeUnbender pipeline: each iteration re-probes the
    residual's metric signature, trains a stage SRGBN, extracts terms
    via MultiRotorLayer rotor plane analysis, and checks coherence for
    backtracking.

    Config keys (under dataset / model / training / iterative):
        dataset.dataset_name    : PMLB dataset name (e.g. "first_principles_newton")
        dataset.category        : "first_principles" | "blackbox" | etc.
        dataset.n_samples       : number of samples to use
        dataset.noise           : Gaussian noise std fraction
        dataset.cache_dir       : where to cache PMLB downloads
        model.hidden_channels   : channel count C
        model.num_layers        : residual block count
        model.num_rotors        : K rotors per MultiRotorLayer
        model.exp_policy        : exp policy ('balanced', 'precise')
        iterative.max_stages    : maximum unbending iterations
        iterative.stage_epochs  : epochs per stage
        iterative.r2_target     : R2 threshold to stop
    """

    def __init__(self, cfg: DictConfig):
        self.dataset_name    = cfg.dataset.get("dataset_name", "feynman_II_37_1")
        self.sparsity_weight = cfg.training.get("sparsity_weight", 0.01)
        self.category        = cfg.dataset.get("category", "feynman")

        # Placeholders - filled in get_data()
        self.n_vars = None
        self.var_names = None
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

        # Epoch counter (used by BaseTask internals)
        self._epoch = 0

        # Probe n_vars from PMLB before model construction
        self._probe_n_vars(cfg)

        # Optional automatic signature discovery
        self._searched_signature = None
        if cfg.algebra.get("auto", False):
            self._searched_signature = self._run_metric_search(cfg)

        # Iterative unbending config
        self.iterative_cfg = dict(cfg.get("iterative", {}))
        # Remove 'enabled' key if present (legacy compat)
        self.iterative_cfg.pop("enabled", None)
        # Merge new pipeline config sections
        self.iterative_cfg.update({
            'implicit_mode': cfg.get("implicit", {}).get("mode", "auto"),
            'probe_config': dict(cfg.get("implicit", {})),
            'grouping_enabled': cfg.get("grouping", {}).get("enabled", True),
            'max_groups': cfg.get("grouping", {}).get("max_groups", 4),
            'svd_warmstart': cfg.get("svd", {}).get("warmstart", True),
            'soft_rejection_alpha': cfg.get("rejection", {}).get("soft_alpha", 10.0),
            'soft_rejection_threshold': cfg.get("rejection", {}).get("soft_threshold", 0.01),
            'mother_cross_threshold': cfg.get("mother_algebra", {}).get("cross_term_threshold", 0.01),
            'basis_config': dict(cfg.get("basis", {})),
            'grouping_config': dict(cfg.get("grouping", {})),
            'feedback_config': dict(cfg.get("feedback", {})),
        })

        super().__init__(cfg)

        # Override scheduler patience (BaseTask defaults to 3)
        sched_patience = cfg.training.get("scheduler_patience", 10)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=sched_patience,
        )

    def _probe_n_vars(self, cfg):
        """Fetch PMLB dataset to determine n_vars and n_train before model setup."""
        cache_dir = cfg.dataset.get("cache_dir", "./data/pmlb_cache")
        df = _fetch_pmlb_data(self.dataset_name, cache_dir)
        self.n_vars = df.shape[1] - 1  # subtract target column
        self.var_names = [c for c in df.columns if c != "target"]
        # Estimate training set size for auto-capacity
        n_total = len(df)
        n_samples_cap = cfg.dataset.get("n_samples", 10000)
        n_use = min(n_total, n_samples_cap) if n_samples_cap > 0 else n_total
        self._n_train_est = cfg.dataset.get("_n_train_est", int(0.75 * n_use))

    def _run_metric_search(self, cfg):
        """Auto-discover optimal (p,q,r) via MetricSearch."""
        from core.analysis import MetricSearch
        import numpy as np

        cache_dir = cfg.dataset.get("cache_dir", "./data/pmlb_cache")
        df = _fetch_pmlb_data(self.dataset_name, cache_dir)
        X = df.drop("target", axis=1).values.astype(np.float32)

        # Use at most 500 samples for search
        if len(X) > 500:
            rng = np.random.default_rng(42)
            X = X[rng.permutation(len(X))[:500]]

        x = torch.from_numpy(X)
        if x.shape[1] > 6:
            x_c = x - x.mean(0)
            _, _, V = torch.linalg.svd(x_c, full_matrices=False)
            x = x_c @ V[:6].T

        device = cfg.algebra.get("device", "cpu")
        searcher = MetricSearch(
            device=device,
            num_probes=cfg.algebra.get("search_probes", 4),
            probe_epochs=cfg.algebra.get("search_epochs", 40),
            micro_batch_size=cfg.algebra.get("search_batch_size", 64),
            early_stop_patience=cfg.algebra.get("search_patience", 8),
        )
        p, q, r = searcher.search(x)
        n = p + q + r
        if n < 2:
            p += (2 - n)
            logger.info(f"MetricSearch: clamped to Cl({p},{q},{r}) (n>=2 required)")
        logger.info(f"MetricSearch: Cl({p},{q},{r}) for {self.dataset_name}")
        return (p, q, r)

    def setup_algebra(self) -> CliffordAlgebra:
        """Use searched signature or configured Cl(p,q,r)."""
        if self._searched_signature is not None:
            p, q, r = self._searched_signature
        else:
            p = self.cfg.algebra.p
            q = self.cfg.algebra.get("q", 0)
            r = self.cfg.algebra.get("r", 0)
        exp_policy = self.cfg.model.get("exp_policy", "balanced")
        return CliffordAlgebra(p=p, q=q, r=r, device=self.device,
                               exp_policy=exp_policy)

    def setup_model(self) -> SRGBN:
        """Build SRGBN with config parameters, optionally auto-sizing."""
        auto = {}
        if self.cfg.model.get("auto_capacity", False):
            auto = SRGBN.auto_config(
                self._n_train_est, self.n_vars, self.algebra.dim
            )
        return SRGBN(
            algebra=self.algebra,
            in_features=self.n_vars,
            channels=self.cfg.model.get("hidden_channels", auto.get("channels", 16)),
            num_layers=self.cfg.model.get("num_layers", auto.get("num_layers", 2)),
        )

    def setup_criterion(self) -> nn.Module:
        return nn.MSELoss()

    def get_data(self):
        """Load PMLB dataset and store normalisation stats."""
        # Augmentation config
        aug_cfg = self.cfg.dataset.get("augmentation", {})
        aug_enabled = aug_cfg.get("enabled", True) if aug_cfg else True
        aug_sigma = aug_cfg.get("sigma", 0.0) if aug_cfg else 0.0
        if not aug_enabled:
            aug_sigma = -1.0  # negative disables auto-sigma in get_sr_loaders

        train_loader, test_loader, x_mean, x_std, y_mean, y_std, var_names = \
            get_sr_loaders(
                dataset_name=self.dataset_name,
                n_samples=self.cfg.dataset.get("n_samples", 10000),
                batch_size=self.cfg.training.batch_size,
                noise=self.cfg.dataset.get("noise", 0.0),
                cache_dir=self.cfg.dataset.get("cache_dir", "./data/pmlb_cache"),
                seed=self.cfg.training.get("seed", 42),
                num_workers=self.cfg.dataset.get("num_workers", 2),
                aug_sigma=aug_sigma,
            )

        self.x_mean = x_mean.to(self.device)
        self.x_std  = x_std.to(self.device)
        self.y_mean = y_mean.to(self.device)
        self.y_std  = y_std.to(self.device)
        self.var_names = var_names

        return train_loader, test_loader

    def train_step(self, batch) -> tuple:
        """One optimisation step (used by BaseTask internals)."""
        x_norm, y_norm = batch
        x_norm = x_norm.to(self.device)
        y_norm = y_norm.to(self.device)

        self.optimizer.zero_grad()
        pred_norm = self.model(x_norm)
        mse_loss = self.criterion(pred_norm, y_norm)
        sparsity = self.sparsity_weight * self.model.total_sparsity_loss()
        loss = mse_loss + sparsity
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            pred_orig = pred_norm.detach() * self.y_std + self.y_mean
            y_orig    = y_norm.detach()    * self.y_std + self.y_mean
            mae_orig  = torch.abs(pred_orig - y_orig).mean().item()

        return loss.item(), {
            "MSE":      mse_loss.item(),
            "Sparsity": sparsity.item(),
            "MAE":      mae_orig,
        }

    def evaluate(self, loader) -> tuple:
        """Compute MAE (original units) and R**2 on a loader."""
        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for x_norm, y_norm in loader:
                x_norm = x_norm.to(self.device)
                y_norm = y_norm.to(self.device)
                pred_norm = self.model(x_norm)

                pred_orig   = pred_norm * self.y_std + self.y_mean
                target_orig = y_norm    * self.y_std + self.y_mean
                preds.append(pred_orig)
                targets.append(target_orig)

        preds   = torch.cat(preds,   dim=0)
        targets = torch.cat(targets, dim=0)

        mae = torch.abs(preds - targets).mean().item()

        ss_res = ((preds - targets) ** 2).sum().item()
        ss_tot = ((targets - targets.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        logger.info(f"MAE (orig units): {mae:.6f}  |  R**2: {r2:.4f}")
        return mae, r2
    
    def extract_formula(self, loader):
        """Extract formula via IterativeUnbender on the given data.

        Returns:
            UnbendingResult with formula and metadata.
        """
        X_all, y_all = [], []
        for x_b, y_b in loader:
            X_all.append(x_b)
            y_all.append(y_b)
        X_norm = torch.cat(X_all).to(self.device)
        y_norm = torch.cat(y_all).to(self.device)

        unbender = IterativeUnbender(
            in_features=self.n_vars,
            device=self.device,
            **self.iterative_cfg,
        )
        return unbender.run(
            X_norm, y_norm,
            self.x_mean, self.x_std, self.y_mean, self.y_std,
            var_names=self.var_names,
        )

    def variable_importance(self, x_batch):
        """Compute per-variable importance via gradient magnitude.

        Args:
            x_batch: torch.Tensor [B, k] input batch.

        Returns:
            torch.Tensor [k] importance scores.
        """
        x_grad = x_batch.detach().clone().requires_grad_(True)
        self.model.eval()
        with torch.enable_grad():
            self.model(x_grad).sum().backward()
        imp = x_grad.grad.abs().mean(0)
        return imp

    def visualize(self, loader):
        """Just Pass we dont need viz."""
        pass

    def run(self):
        """Run the iterative geometric unbending pipeline."""
        logger.info(f"Starting SRTask: {self.dataset_name}")

        train_loader, test_loader = self.get_data()

        # Collect all training data
        X_all, y_all = [], []
        for x_b, y_b in train_loader:
            X_all.append(x_b)
            y_all.append(y_b)
        X_norm = torch.cat(X_all).to(self.device)
        y_norm = torch.cat(y_all).to(self.device)

        unbender = IterativeUnbender(
            in_features=self.n_vars,
            device=self.device,
            **self.iterative_cfg,
        )
        result = unbender.run(
            X_norm, y_norm,
            self.x_mean, self.x_std, self.y_mean, self.y_std,
            var_names=self.var_names,
        )

        # Log per-stage results
        for s in result.stages:
            logger.info(
                f"Stage {s.stage_idx}: Cl{s.signature} "
                f"curv={s.curvature_before:.3f}->{s.curvature_after:.3f} "
                f"coh={s.coherence_before:.3f}->{s.coherence_after:.3f} "
                f"terms={len(s.terms)} accepted={s.accepted}"
            )
        logger.info(f"Formula: {result.formula}")
        logger.info(f"R2: {result.r2_final:.6f}")
        return result
