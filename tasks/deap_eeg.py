# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""DEAP EEG Emotion Regression Task.

Predicts Valence, Arousal, Dominance, and Liking (VADL) from 32-channel EEG
using Geometric Algebra. Cross-subject (LOSO) validation by default.

Key: emotional states are pushed into Grade-0 (rotor-invariant scalars).
"""

import torch
import torch.nn as nn

from core.config import make_algebra_from_config
from datalib.deap import get_deap_loaders, get_group_sizes
from log import get_logger
from models.deap import EEGNet
from tasks.base import BaseTask

logger = get_logger(__name__)

VADL_NAMES = ["Valence", "Arousal", "Dominance", "Liking"]


class DEAPEEGTask(BaseTask):
    """DEAP EEG Emotion Regression Task.

    Predicts VADL ratings from 32-channel EEG using a Geometric Algebra
    transformer with Mother embedding and Neutral artifact removal.

    Default evaluation: cross-subject LOSO (leave-one-subject-out).
    """

    def __init__(self, cfg):
        self.data_root = cfg.dataset.get("data_root", "data/deap/data_preprocessed_python")
        self.subject_id = cfg.dataset.get("subject_id", 1)
        self.eval_mode = cfg.dataset.get("eval_mode", "cross_subject")
        self.window_size = cfg.dataset.get("window_size", 512)
        self.stride = cfg.dataset.get("stride", None)
        super().__init__(cfg)

    def setup_algebra(self):
        return make_algebra_from_config(
            self.cfg.algebra,
            p=self.cfg.algebra.get("p", 3),
            q=self.cfg.algebra.get("q", 1),
            r=self.cfg.algebra.get("r", 0),
            device=self.device,
        )

    def setup_model(self):
        group_sizes = get_group_sizes()

        profiles = None
        if self.cfg.model.get("use_profiler", False):
            profiles = self._compute_profiles(group_sizes)

        return EEGNet(
            group_sizes,
            profiles=profiles,
            device=self.device,
            config=self.cfg,
        )

    def _compute_profiles(self, group_sizes):
        """Compute uncertainty (U) and Procrustes alignment (V) per region."""
        try:
            from core.analysis import compute_uncertainty_and_alignment
            from datalib.deap import DEAPDataset
        except ImportError:
            logger.warning("Profiler unavailable, skipping alignment computation.")
            return None

        ds = DEAPDataset(self.data_root, [self.subject_id], self.window_size, self.stride, normalize=False)
        if len(ds) == 0:
            return None

        profiles = {}
        for name in sorted(group_sizes.keys()):
            feats = torch.stack([ds[i][0][name] for i in range(len(ds))])  # [N, dim]
            U, V = compute_uncertainty_and_alignment(self.algebra, feats.to(self.device))
            profiles[name] = {"U": U, "V": V}
            logger.info("Profile %s: U=%.4f, V shape=%s", name, U, list(V.shape))

        return profiles

    def setup_criterion(self):
        return nn.MSELoss()

    def get_data(self):
        train_loader, val_loader = get_deap_loaders(
            self.data_root,
            subject_id=self.subject_id,
            mode=self.eval_mode,
            batch_size=self.cfg.training.batch_size,
            window_size=self.window_size,
            stride=self.stride,
        )
        return train_loader, val_loader

    def train_step(self, batch):
        self.optimizer.zero_grad()
        group_data, labels = batch

        group_data = {k: v.to(self.device) for k, v in group_data.items()}
        labels = labels.to(self.device)

        preds = self.model(group_data)  # [B, 4]
        loss = self.criterion(preds, labels)

        self._backward(loss)
        self._optimizer_step()

        return loss.item(), {"Loss": loss.item()}

    def evaluate(self, val_loader):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                group_data, labels = batch
                group_data = {k: v.to(self.device) for k, v in group_data.items()}
                preds = self.model(group_data)
                all_preds.append(preds.cpu())
                all_labels.append(labels)

        preds_tensor = torch.cat(all_preds)  # [N, 4]
        labels_tensor = torch.cat(all_labels)  # [N, 4]

        metrics = {}

        # RMSE per VADL dimension
        rmse = ((preds_tensor - labels_tensor) ** 2).mean(dim=0).sqrt()
        for i, name in enumerate(VADL_NAMES):
            metrics[f"{name}_RMSE"] = rmse[i].item()

        # Binary F1 -- fixed threshold 0.5 (Koelstra 2012: midpoint of 1-9 scale = (5-1)/8)
        try:
            from sklearn.metrics import f1_score

            preds_np = preds_tensor.numpy()
            labels_np = labels_tensor.numpy()
            for i, name in enumerate(VADL_NAMES):
                pred_bin = (preds_np[:, i] > 0.5).astype(int)
                label_bin = (labels_np[:, i] > 0.5).astype(int)
                metrics[f"{name}_F1"] = f1_score(label_bin, pred_bin, average="binary", zero_division=0)
        except ImportError:
            logger.warning("scikit-learn not available, skipping F1 metrics.")

        return metrics

    def _log_results_table(self, metrics: dict) -> None:
        """Log VADL results as a compact ASCII table."""
        header = f"{'Dimension':<13} {'RMSE':>8} {'F1':>8}"
        sep = "-" * len(header)
        logger.info("Final Results (mode=regression, threshold=0.5)")
        logger.info(sep)
        logger.info(header)
        logger.info(sep)
        for name in VADL_NAMES:
            rmse = metrics.get(f"{name}_RMSE", float("nan"))
            f1 = metrics.get(f"{name}_F1", float("nan"))
            logger.info(f"{name:<13} {rmse:>8.4f} {f1:>8.4f}")
        logger.info(sep)
        mean_rmse = sum(metrics.get(f"{n}_RMSE", 0) for n in VADL_NAMES) / len(VADL_NAMES)
        mean_f1 = sum(metrics.get(f"{n}_F1", 0) for n in VADL_NAMES) / len(VADL_NAMES)
        logger.info(f"{'Mean':<13} {mean_rmse:>8.4f} {mean_f1:>8.4f}")
        logger.info(sep)

    def visualize(self, val_loader):
        pass

    def run(self):
        logger.info("Starting Task: DEAP EEG (subject=%d, mode=%s)", self.subject_id, self.eval_mode)
        train_loader, val_loader = self.get_data()

        from tqdm import tqdm

        pbar = tqdm(range(self.epochs))
        best_val_rmse = float("inf")

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            val_metrics = self.evaluate(val_loader)
            val_rmse_mean = sum(v for k, v in val_metrics.items() if k.endswith("_RMSE")) / 4

            self.scheduler.step(val_rmse_mean)

            if val_rmse_mean < best_val_rmse:
                best_val_rmse = val_rmse_mean
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            display = {
                "Loss": avg_loss,
                "Val_RMSE": val_rmse_mean,
                "LR": self.optimizer.param_groups[0]["lr"],
            }
            desc = " | ".join(f"{k}: {v:.4f}" for k, v in display.items())
            pbar.set_description(desc)

        logger.info("Training Complete. Best RMSE: %.4f", best_val_rmse)

        self.load_checkpoint(f"{self.cfg.name}_best.pt")
        final_metrics = self.evaluate(val_loader)
        self._log_results_table(final_metrics)

        return final_metrics
