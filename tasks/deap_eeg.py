# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""DEAP EEG Emotion Classification Task.

Predicts Valence, Arousal, Dominance, and Liking (VADL) from 32-channel EEG
using Geometric Algebra. Cross-subject (LOSO) validation by default.

Key: emotional states are pushed into Grade-0 (rotor-invariant scalars).
"""

import numpy as np
import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from datalib.deap import get_deap_loaders, get_group_sizes
from models.deap import EEGNet
from log import get_logger

logger = get_logger(__name__)

VADL_NAMES = ['Valence', 'Arousal', 'Dominance', 'Liking']


class DEAPEEGTask(BaseTask):
    """DEAP EEG Emotion Classification Task.

    Predicts VADL ratings from 32-channel EEG using a Geometric Algebra
    transformer with Mother embedding and Neutral artifact removal.

    Default evaluation: cross-subject LOSO (leave-one-subject-out).
    """

    def __init__(self, cfg):
        self.data_root = cfg.dataset.get('data_root', 'data/deap/data_preprocessed_python')
        self.subject_id = cfg.dataset.get('subject_id', 1)
        self.eval_mode = cfg.dataset.get('eval_mode', 'cross_subject')
        self.window_size = cfg.dataset.get('window_size', 512)
        self.stride = cfg.dataset.get('stride', None)
        self.task_mode = cfg.dataset.get('task_mode', 'regression')
        super().__init__(cfg)

    def setup_algebra(self):
        return CliffordAlgebra(
            p=self.cfg.algebra.get('p', 3),
            q=self.cfg.algebra.get('q', 1),
            r=self.cfg.algebra.get('r', 0),
            device=self.device,
        )

    def setup_model(self):
        group_sizes = get_group_sizes()

        # Optionally compute profiler-based alignment per region
        profiles = None
        if self.cfg.model.get('use_profiler', False):
            profiles = self._compute_profiles(group_sizes)

        return EEGNet(
            group_sizes,
            profiles=profiles,
            device=self.device,
            config=self.cfg,
        )

    def _compute_profiles(self, group_sizes):
        """Compute uncertainty (U) and Procrustes alignment (V) per region.

        Loads a small sample of the target subject's data and runs the
        geometric profiler on each brain region group.
        """
        try:
            from core.search import compute_uncertainty_and_alignment
            from datalib.deap import DEAPDataset, REGION_GROUPS
        except ImportError:
            logger.warning("Profiler unavailable, skipping alignment computation.")
            return None

        # Load one subject raw (unnormalized) for profiling
        ds = DEAPDataset(self.data_root, [self.subject_id], self.window_size, self.stride, normalize=False)
        if len(ds) == 0:
            return None

        profiles = {}
        for name in sorted(group_sizes.keys()):
            # Collect all features for this region
            feats = torch.stack([ds[i][0][name] for i in range(len(ds))])  # [N, dim]
            U, V = compute_uncertainty_and_alignment(self.algebra, feats.to(self.device))
            profiles[name] = {'U': U, 'V': V}
            logger.info("Profile %s: U=%.4f, V shape=%s", name, U, list(V.shape))

        return profiles

    def setup_criterion(self):
        if self.task_mode == 'regression':
            return nn.MSELoss()
        return nn.BCEWithLogitsLoss()

    def get_data(self):
        return get_deap_loaders(
            self.data_root,
            subject_id=self.subject_id,
            mode=self.eval_mode,
            batch_size=self.cfg.training.batch_size,
            window_size=self.window_size,
            stride=self.stride,
        )

    def train_step(self, batch):
        self.optimizer.zero_grad()
        group_data, labels = batch

        group_data = {k: v.to(self.device) for k, v in group_data.items()}
        labels = labels.to(self.device)

        if self.task_mode == 'classification':
            medians = labels.median(dim=0).values
            labels = (labels > medians).float()

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

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        # RMSE per VADL dimension
        rmse = ((preds - labels) ** 2).mean(dim=0).sqrt()

        metrics = {}
        for i, name in enumerate(VADL_NAMES):
            metrics[f'{name}_RMSE'] = rmse[i].item()

        # Binary F1 — per-dimension median threshold (Koelstra et al., 2012)
        try:
            from sklearn.metrics import f1_score
            preds_np = preds.numpy()
            labels_np = labels.numpy()
            for i, name in enumerate(VADL_NAMES):
                threshold_i = float(np.median(labels_np[:, i]))
                pred_bin = (preds_np[:, i] > threshold_i).astype(int)
                label_bin = (labels_np[:, i] > threshold_i).astype(int)
                f1 = f1_score(label_bin, pred_bin, average='binary', zero_division=0)
                metrics[f'{name}_F1'] = f1
                metrics[f'{name}_threshold'] = threshold_i
        except ImportError:
            logger.warning("scikit-learn not available, skipping F1 metrics.")

        return metrics

    def visualize(self, val_loader):
        pass

    def run(self):
        logger.info("Starting Task: DEAP EEG (subject=%d, mode=%s)", self.subject_id, self.eval_mode)
        train_loader, val_loader = self.get_data()

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))
        best_val_loss = float('inf')

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_rmse_mean = sum(v for k, v in val_metrics.items() if k.endswith('_RMSE')) / 4

            self.scheduler.step(val_rmse_mean)

            if val_rmse_mean < best_val_loss:
                best_val_loss = val_rmse_mean
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            display = {
                'Loss': avg_loss,
                'Val_RMSE': val_rmse_mean,
                'LR': self.optimizer.param_groups[0]['lr'],
            }
            desc = " | ".join(f"{k}: {v:.4f}" for k, v in display.items())
            pbar.set_description(desc)

        logger.info("Training Complete. Best Val RMSE: %.4f", best_val_loss)

        # Load best and final evaluation
        self.load_checkpoint(f"{self.cfg.name}_best.pt")
        final_metrics = self.evaluate(val_loader)
        for k, v in final_metrics.items():
            logger.info("FINAL %s: %.4f", k, v)

        return final_metrics
