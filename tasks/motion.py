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
from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from datasets.amass import AMASSDataset
from models.motion import MotionManifoldNetwork
from torch.utils.data import DataLoader
from core.visualizer import GeneralVisualizer

class MotionAlignmentTask(BaseTask):
    """Task for aligning complex motion data into linearly separable latent space."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up the algebra. 
        We use a lower dimension (e.g. p=4) for the latent space, even if input is high-dim.
        """
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        return MotionManifoldNetwork(self.algebra, input_dim=45, latent_dim=self.cfg.algebra.p)

    def setup_criterion(self):
        return nn.CrossEntropyLoss()

    def get_data(self):
        dataset = AMASSDataset(self.algebra, num_samples=self.cfg.dataset.samples)
        return DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=True)

    def train_step(self, batch):
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward
        logits, vectors, aligned = self.model(data)
        
        # Loss: Classification accuracy implies linear separability
        loss = self.criterion(logits, labels)
        
        # Optional: Add sparsity loss to the rotor
        sparsity = self.model.rotor.sparsity_loss()
        total_loss = loss + 0.01 * sparsity
        
        total_loss.backward()
        self.optimizer.step()
        
        # Calculate Acc
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        return total_loss.item(), {"Loss": loss.item(), "Acc": acc.item(), "Sparse": sparsity.item()}

    def evaluate(self, data):
        # Evaluation is implicit in training logs for this prototype
        pass

    def visualize(self, data):
        """Visualizes the latent space distribution."""
        # Get a batch
        batch = next(iter(self.get_data()))
        inputs, labels = batch
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            _, vectors, _ = self.model(inputs)
            
        # Plot 2D or 3D scatter of vectors colored by label
        # We need a custom visualizer function for labeled data, 
        # but GeneralVisualizer is mainly for multivectors.
        # We'll just print a summary for now or save a simple plot if matplotlib is available.
        try:
            import matplotlib.pyplot as plt
            vecs = vectors.cpu().numpy()
            lbls = labels.numpy()
            
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(vecs[:, 0], vecs[:, 1], c=lbls, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Action Class')
            plt.title("Latent Space Distribution (Dim 1 vs Dim 2)")
            plt.xlabel("e1")
            plt.ylabel("e2")
            plt.savefig("motion_latent_space.png")
            print(">>> Saved visualization to motion_latent_space.png")
            plt.close()
        except ImportError:
            print("Matplotlib not found, skipping plot.")
