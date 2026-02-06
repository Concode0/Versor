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
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer
from datasets.crossmodal import CrossModalDataset
from torch.utils.data import DataLoader

class ModalityEncoder(nn.Module):
    """Encodes a modality.

    Projects to high-dim space and rotates it to align with the truth.
    """

    def __init__(self, algebra):
        """Sets up the encoder."""
        super().__init__()
        self.net = nn.Sequential(
            CliffordLinear(algebra, 1, 4),
            CliffordLayerNorm(algebra, channels=4),
            GeometricGELU(algebra, channels=4),
            CliffordLinear(algebra, 4, 1),
            CliffordLayerNorm(algebra, channels=1),
            RotorLayer(algebra, channels=1) 
        )
    def forward(self, x):
        """Forward pass."""
        return self.net(x)

class CrossModalBinder(nn.Module):
    """Dual-encoder. Making apples look like oranges.

    Two encoders, one shared space.
    """

    def __init__(self, algebra):
        """Sets up the binder."""
        super().__init__()
        self.algebra = algebra
        self.encoder_A = ModalityEncoder(algebra) # e.g. Text
        self.encoder_B = ModalityEncoder(algebra) # e.g. Image
        
    def forward(self, x_a, x_b):
        """Encodes both."""
        z_a = self.encoder_A(x_a)
        z_b = self.encoder_B(x_b)
        return z_a, z_b

class CrossModalTask(BaseTask):
    """Cross-Modal Alignment. Fixing the tower of Babel.

    Aligns disparate modalities by finding the relative rotation.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """High dimensional Euclidean."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        """The Binder."""
        return CrossModalBinder(self.algebra)

    def setup_criterion(self):
        """Geometric MSE."""
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        """Synthetic multi-modal data."""
        dataset = CrossModalDataset(self.algebra, embedding_dim=self.cfg.dataset.embedding_dim)
        return DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=True)

    def _contrastive_loss(self, z_a, z_b, temperature=0.1):
        """InfoNCE contrastive loss. Pulls pairs together, pushes non-pairs apart."""
        # Flatten to [N, D]
        a = z_a.squeeze(1)
        b = z_b.squeeze(1)

        # L2 normalize
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        # Cosine similarity matrix [N, N]
        logits = a @ b.t() / temperature

        # Targets: diagonal (i-th row should match i-th column)
        targets = torch.arange(logits.size(0), device=logits.device)

        # Symmetric cross-entropy: A->B and B->A
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)

        return (loss_ab + loss_ba) / 2

    def train_step(self, data):
        """Align them."""
        data_A, data_B = data
        data_A, data_B = data_A.to(self.device), data_B.to(self.device)

        self.optimizer.zero_grad()

        z_a, z_b = self.model(data_A, data_B)

        # Alignment Loss (pull matched pairs together)
        loss_align = self.criterion(z_a, z_b)

        # Contrastive Loss (push non-pairs apart)
        loss_contrast = self._contrastive_loss(z_a, z_b)

        # Norm regularization to prevent collapse
        norm_a = (z_a**2).sum(dim=-1).mean()
        norm_input = (data_A**2).sum(dim=-1).mean()
        loss_norm = (norm_a - norm_input).pow(2)

        loss = loss_align + 0.5 * loss_contrast + 0.1 * loss_norm

        loss.backward()
        self.optimizer.step()

        return loss.item(), {"Align": loss_align.item(), "Contr": loss_contrast.item(), "Norm": loss_norm.item()}

    def evaluate(self, data):
        """Did we solve translation?"""
        data_A, data_B = data
        data_A, data_B = data_A.to(self.device), data_B.to(self.device)

        z_a, z_b = self.model(data_A, data_B)

        # 1. Alignment Distance
        final_dist = torch.norm(z_a - z_b, dim=-1).mean().item()
        print(f"Final Alignment Distance: {final_dist:.6f}")

        # 2. Alignment Ratio
        initial_dist = torch.norm(data_A - data_B, dim=-1).mean().item()
        ratio = final_dist / (initial_dist + 1e-9)
        print(f"Alignment Ratio: {ratio:.4f}")

        # 3. Cosine Similarity Analysis
        a = F.normalize(z_a.squeeze(1), dim=-1)
        b = F.normalize(z_b.squeeze(1), dim=-1)
        sim_matrix = a @ b.t()

        n = sim_matrix.size(0)
        diag_mask = torch.eye(n, dtype=torch.bool, device=self.device)

        matched_sim = sim_matrix[diag_mask].mean().item()
        unmatched_sim = sim_matrix[~diag_mask].mean().item()
        sim_gap = matched_sim - unmatched_sim

        print(f"Matched Cosine Sim: {matched_sim:.4f}")
        print(f"Unmatched Cosine Sim: {unmatched_sim:.4f}")
        print(f"Similarity Gap: {sim_gap:.4f}")

        # 4. Retrieval Accuracy (bidirectional)
        preds_ab = sim_matrix.argmax(dim=1)
        preds_ba = sim_matrix.argmax(dim=0)
        targets = torch.arange(n, device=self.device)
        acc_ab = (preds_ab == targets).float().mean().item()
        acc_ba = (preds_ba == targets).float().mean().item()
        print(f"Retrieval Accuracy (A→B): {acc_ab*100:.2f}%")
        print(f"Retrieval Accuracy (B→A): {acc_ba*100:.2f}%")

    def visualize(self, data):
        """Plots the alignment."""
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        data_A, data_B = data
        data_A, data_B = data_A.to(self.device), data_B.to(self.device)
        z_a, z_b = self.model(data_A, data_B)

        viz = GeneralVisualizer(self.algebra)

        def plot_embedding(A, B, title, fname):
            A_flat = A.reshape(-1, self.algebra.dim).detach().cpu().numpy()
            B_flat = B.reshape(-1, self.algebra.dim).detach().cpu().numpy()
            combined = torch.cat([torch.tensor(A_flat), torch.tensor(B_flat)], dim=0)

            pca = PCA(n_components=2)
            emb = pca.fit_transform(combined.numpy())
            n = len(A_flat)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(emb[:n, 0], emb[:n, 1], c='blue', label='Source A', alpha=0.6)
            plt.scatter(emb[n:, 0], emb[n:, 1], c='red', label='Source B', alpha=0.6)
            plt.legend()
            plt.title(title)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
            plt.savefig(fname)
            plt.close()

        plot_embedding(data_A, data_B, "Before Unification", "crossmodal_before.png")
        plot_embedding(z_a, z_b, "After Unification", "crossmodal_after.png")