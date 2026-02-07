# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.normalization import CliffordLayerNorm
from functional.activation import GeometricGELU
from functional.loss import GeometricMSELoss
from datasets.newsgroups import NewsgroupsCrossModalDataset
from tasks.base import BaseTask
from torch.utils.data import DataLoader


class ModalityEncoder(nn.Module):
    """Two-stage rotor modality encoder.

    Stage 1: CliffordLinear → GeometricGELU → CliffordLinear → RotorLayer
    Stage 2: CliffordLinear → GeometricGELU → CliffordLinear → RotorLayer
    Output:  CliffordLayerNorm

    Two rotations give enough geometric expressivity to map
    one modality's manifold into the shared space.
    """

    def __init__(self, algebra, in_channels=8, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            # Stage 1: initial rotation
            CliffordLinear(algebra, in_channels, hidden_channels),
            GeometricGELU(algebra, channels=hidden_channels),
            CliffordLinear(algebra, hidden_channels, in_channels),
            RotorLayer(algebra, channels=in_channels),
            # Stage 2: refinement rotation
            CliffordLinear(algebra, in_channels, hidden_channels),
            GeometricGELU(algebra, channels=hidden_channels),
            CliffordLinear(algebra, hidden_channels, in_channels),
            RotorLayer(algebra, channels=in_channels),
            # Normalize
            CliffordLayerNorm(algebra, channels=in_channels),
        )

    def forward(self, x):
        return self.net(x)


class CrossModalBinder(nn.Module):
    """Dual single-rotor encoders. Finding the relative rotation.

    Each modality gets its own rotor that learns to align
    into a shared geometric space. The rotation IS the alignment.
    """

    def __init__(self, algebra, in_channels=8, hidden_channels=16):
        super().__init__()
        self.algebra = algebra
        self.encoder_A = ModalityEncoder(algebra, in_channels, hidden_channels)
        self.encoder_B = ModalityEncoder(algebra, in_channels, hidden_channels)

    def forward(self, x_a, x_b):
        z_a = self.encoder_A(x_a)
        z_b = self.encoder_B(x_b)
        return z_a, z_b


class CrossModalTask(BaseTask):
    """Cross-Modal Alignment on 20 Newsgroups.

    Aligns BERT (contextual semantics) with TF-IDF (word frequency statistics)
    into a shared geometric space via dual single-rotor encoders.

    Each encoder learns a rotation that maps its modality into
    the shared space. Contrastive + alignment loss pulls matched
    pairs together and pushes non-pairs apart.
    """

    def __init__(self, cfg):
        self.in_channels = cfg.model.get('in_channels', 8)
        super().__init__(cfg)
        # Override scheduler: cosine annealing with warm restarts
        # keeps LR high enough for contrastive learning throughout training
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-4
        )

    def setup_algebra(self):
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        return CrossModalBinder(
            self.algebra,
            in_channels=self.in_channels,
            hidden_channels=self.cfg.model.hidden_channels,
        )

    def setup_criterion(self):
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        cache_dir = self.cfg.dataset.get('cache_dir', './data/newsgroups')
        max_samples = self.cfg.dataset.get('max_samples', 2000) or 2000
        embedding_dim = self.cfg.dataset.embedding_dim

        train_dataset = NewsgroupsCrossModalDataset(
            self.algebra, embedding_dim=embedding_dim,
            in_channels=self.in_channels,
            split='train', max_samples=max_samples, cache_dir=cache_dir
        )
        val_dataset = NewsgroupsCrossModalDataset(
            self.algebra, embedding_dim=embedding_dim,
            in_channels=self.in_channels,
            split='test', max_samples=max_samples, cache_dir=cache_dir
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.training.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.cfg.training.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def _contrastive_loss(self, z_a, z_b, temperature=0.5):
        """InfoNCE contrastive loss. Pulls pairs together, pushes non-pairs apart."""
        # Flatten channels: [B, C, dim] → [B, C*dim]
        a = z_a.reshape(z_a.size(0), -1)
        b = z_b.reshape(z_b.size(0), -1)

        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        logits = a @ b.t() / temperature
        targets = torch.arange(logits.size(0), device=logits.device)

        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)

        return (loss_ab + loss_ba) / 2

    def train_step(self, batch):
        data_bert, data_tfidf, _ = batch
        data_bert = data_bert.to(self.device)
        data_tfidf = data_tfidf.to(self.device)

        self.optimizer.zero_grad()

        z_a, z_b = self.model(data_bert, data_tfidf)

        # Contrastive loss (primary)
        loss_contrast = self._contrastive_loss(z_a, z_b)

        # Norm regularization (keep embeddings from collapsing or exploding)
        norm_a = (z_a ** 2).sum(dim=-1).mean()
        norm_b = (z_b ** 2).sum(dim=-1).mean()
        loss_norm = (norm_a - 1.0).pow(2) + (norm_b - 1.0).pow(2)

        loss = loss_contrast + 0.1 * loss_norm

        loss.backward()
        self.optimizer.step()

        return loss.item(), {
            "Contr": loss_contrast.item(),
            "Norm": loss_norm.item()
        }

    def evaluate(self, val_loader):
        self.model.eval()

        all_za = []
        all_zb = []
        all_da = []
        all_db = []

        with torch.no_grad():
            for data_bert, data_tfidf, _ in val_loader:
                data_bert = data_bert.to(self.device)
                data_tfidf = data_tfidf.to(self.device)
                z_a, z_b = self.model(data_bert, data_tfidf)
                all_za.append(z_a)
                all_zb.append(z_b)
                all_da.append(data_bert)
                all_db.append(data_tfidf)

        z_a = torch.cat(all_za, dim=0)
        z_b = torch.cat(all_zb, dim=0)
        data_A = torch.cat(all_da, dim=0)
        data_B = torch.cat(all_db, dim=0)

        # Alignment distance
        final_dist = torch.norm(z_a - z_b, dim=-1).mean().item()
        print(f"Final Alignment Distance: {final_dist:.6f}")

        # Alignment ratio
        initial_dist = torch.norm(data_A - data_B, dim=-1).mean().item()
        ratio = final_dist / (initial_dist + 1e-9)
        print(f"Alignment Ratio: {ratio:.4f}")

        # Cosine similarity analysis (flatten channels)
        a = F.normalize(z_a.reshape(z_a.size(0), -1), dim=-1)
        b = F.normalize(z_b.reshape(z_b.size(0), -1), dim=-1)
        sim_matrix = a @ b.t()

        n = sim_matrix.size(0)
        diag_mask = torch.eye(n, dtype=torch.bool, device=self.device)

        matched_sim = sim_matrix[diag_mask].mean().item()
        unmatched_sim = sim_matrix[~diag_mask].mean().item()
        sim_gap = matched_sim - unmatched_sim

        print(f"Matched Cosine Sim: {matched_sim:.4f}")
        print(f"Unmatched Cosine Sim: {unmatched_sim:.4f}")
        print(f"Similarity Gap: {sim_gap:.4f}")

        # Retrieval accuracy
        preds_ab = sim_matrix.argmax(dim=1)
        preds_ba = sim_matrix.argmax(dim=0)
        targets = torch.arange(n, device=self.device)
        acc_ab = (preds_ab == targets).float().mean().item()
        acc_ba = (preds_ba == targets).float().mean().item()
        print(f"Retrieval Accuracy (A→B): {acc_ab * 100:.2f}%")
        print(f"Retrieval Accuracy (B→A): {acc_ba * 100:.2f}%")

        return sim_gap

    def visualize(self, val_loader):
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print(">>> matplotlib/sklearn not found, skipping visualization.")
            return

        self.model.eval()
        all_za = []
        all_zb = []

        with torch.no_grad():
            for data_bert, data_tfidf, _ in val_loader:
                data_bert = data_bert.to(self.device)
                data_tfidf = data_tfidf.to(self.device)
                z_a, z_b = self.model(data_bert, data_tfidf)
                all_za.append(z_a)
                all_zb.append(z_b)

        z_a = torch.cat(all_za, dim=0)
        z_b = torch.cat(all_zb, dim=0)

        A_flat = z_a.reshape(-1, self.algebra.dim).detach().cpu().numpy()
        B_flat = z_b.reshape(-1, self.algebra.dim).detach().cpu().numpy()

        combined = np.concatenate([A_flat, B_flat], axis=0)

        pca = PCA(n_components=2)
        emb = pca.fit_transform(combined)
        n = len(A_flat)

        plt.figure(figsize=(10, 6))
        plt.scatter(emb[:n, 0], emb[:n, 1], c='blue', label='BERT', alpha=0.4, s=10)
        plt.scatter(emb[n:, 0], emb[n:, 1], c='red', label='TF-IDF', alpha=0.4, s=10)
        plt.legend()
        plt.title('Cross-Modal Alignment (BERT vs TF-IDF)')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.savefig('crossmodal_alignment.png')
        plt.close()
        print(">>> Saved visualization to crossmodal_alignment.png")

    def run(self):
        """Train/val loop."""
        print(f">>> Starting Task: {self.cfg.name}")
        train_loader, val_loader = self.get_data()

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))

        best_sim_gap = -float('inf')

        for epoch in pbar:
            self.model.train()
            total_loss = 0
            total_contr = 0

            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                total_contr += logs['Contr']

            n_batches = len(train_loader)
            avg_loss = total_loss / n_batches
            avg_contr = total_contr / n_batches

            sim_gap = self.evaluate(val_loader)
            self.scheduler.step(epoch)

            if sim_gap > best_sim_gap:
                best_sim_gap = sim_gap
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            logs = {
                'Loss': avg_loss,
                'Contr': avg_contr,
                'Gap': sim_gap,
                'LR': self.optimizer.param_groups[0]['lr']
            }
            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

        print(f">>> Training Complete. Best Similarity Gap: {best_sim_gap:.4f}")
        self.save_checkpoint(f"{self.cfg.name}_final.pt")

        self.model.eval()
        with torch.no_grad():
            self.visualize(val_loader)
