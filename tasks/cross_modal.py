# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.normalization import CliffordLayerNorm
from functional.activation import GeometricGELU
from functional.loss import GeometricMSELoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer
from datasets.synthetic import CrossModalDataset
from torch.utils.data import DataLoader

class ModalityEncoder(nn.Module):
    """Encoder network for a single modality.

    Projects data into a high-dimensional Clifford space and learns a
    rotor-based transformation to align with a shared semantic space.
    """

    def __init__(self, algebra):
        """Initializes the encoder.

        Args:
            algebra: The algebra instance.
        """
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
    """Dual-encoder network for cross-modal alignment.

    Consists of two independent encoders (A and B) that map different modalities
    to a shared geometric embedding space.
    """

    def __init__(self, algebra):
        """Initializes the binder."""
        super().__init__()
        self.algebra = algebra
        
        # Two independent encoders
        self.encoder_A = ModalityEncoder(algebra) # e.g. Text
        self.encoder_B = ModalityEncoder(algebra) # e.g. Image
        
    def forward(self, x_a, x_b):
        """Encodes both modalities."""
        z_a = self.encoder_A(x_a)
        z_b = self.encoder_B(x_b)
        return z_a, z_b

class CrossModalTask(BaseTask):
    """Task for aligning two disparate modalities (Text and Distorted Image).

    Demonstrates how RotorLayers can find the optimal relative rotation between
    misaligned semantic manifolds.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up the algebra (p=embedding_dim, q=0)."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        """Sets up the CrossModalBinder."""
        return CrossModalBinder(self.algebra)

    def setup_criterion(self):
        """Sets up Geometric MSE Loss."""
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        """Generates synthetic multi-modal data."""
        dataset = CrossModalDataset(self.algebra, embedding_dim=self.cfg.task.dataset.embedding_dim)
        return DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=True)

    def train_step(self, data):
        """Executes one training step."""
        data_A, data_B = data
        data_A, data_B = data_A.to(self.device), data_B.to(self.device)
        
        self.optimizer.zero_grad()
        
        z_a, z_b = self.model(data_A, data_B)
        
        # Contrastive/Alignment Loss
        loss_align = self.criterion(z_a, z_b)
        
        # Regularization to prevent collapse
        norm_a = (z_a**2).sum(dim=-1).mean()
        norm_input = (data_A**2).sum(dim=-1).mean()
        loss_norm = (norm_a - norm_input).pow(2)
        
        loss = loss_align + 0.1 * loss_norm
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Align": loss_align.item(), "Norm": loss_norm.item()}

    def evaluate(self, data):
        """Evaluates alignment quality and retrieval accuracy."""
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
        
        # 3. Retrieval Accuracy
        z_a_norm = z_a / (z_a.norm(dim=-1, keepdim=True) + 1e-9)
        z_b_norm = z_b / (z_b.norm(dim=-1, keepdim=True) + 1e-9)
        sim_matrix = torch.mm(z_a_norm.squeeze(1), z_b_norm.squeeze(1).t())
        preds = sim_matrix.argmax(dim=1)
        targets = torch.arange(len(preds), device=self.device)
        accuracy = (preds == targets).float().mean().item()
        print(f"Retrieval Accuracy: {accuracy*100:.2f}%")

    def visualize(self, data):
        """Visualizes the alignment before and after training."""
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