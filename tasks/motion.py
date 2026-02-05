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
from datasets.har import HARDataset
from models.motion import MotionManifoldNetwork
from torch.utils.data import DataLoader
from core.visualizer import GeneralVisualizer

class MotionAlignmentTask(BaseTask):
    """Task for aligning complex motion data into linearly separable latent space."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up the algebra using config values."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        # HAR has 561 features
        input_dim = 561
        num_classes = 6 # Default for HAR
        
        # Check if we are using synthetic fallback
        import os
        if not os.path.exists('./data/HAR/train.csv'):
            input_dim = 45 # AMASS synthetic default
            num_classes = 3 # Walk, Run, Jump
            
        return MotionManifoldNetwork(self.algebra, input_dim=input_dim, latent_dim=self.cfg.algebra.p, num_classes=num_classes)

    def setup_criterion(self):
        return nn.CrossEntropyLoss()

    def get_data(self):
        try:
            dataset = HARDataset(self.algebra, root='./data/HAR', split='train')
        except FileNotFoundError:
            from datasets.amass import AMASSDataset
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
        
        total_loss = loss
        
        total_loss.backward()
        self.optimizer.step()
        
        # Calculate Acc
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Calculate Grade Purity (Expect Grade 1 vectors in latent space)
        # 'aligned' is [Batch, 1, Algebra_Dim]
        from core.metric import grade_purity
        purity = grade_purity(self.algebra, aligned.squeeze(1), grade=1).mean()
        
        return total_loss.item(), {
            "Loss": loss.item(), 
            "Acc": acc.item(), 
            "Purity": purity.item()
        }

    def evaluate(self, data):
        # Evaluation is implicit in training logs for this prototype
        pass

    def visualize(self, data):
        """Visualizes the latent space distribution."""
        # Get a batch
        loader = self.get_data()
        batch = next(iter(loader))
        inputs, labels = batch
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            _, vectors, _ = self.model(inputs)
            
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            vecs = vectors.cpu().numpy()
            lbls = labels.numpy()
            
            # Use PCA if dims > 2
            if vecs.shape[1] > 2:
                pca = PCA(n_components=2)
                vecs_2d = pca.fit_transform(vecs)
                x_label, y_label = "PC1", "PC2"
                explained_var = pca.explained_variance_ratio_
                title_suffix = f"(PCA: {explained_var[0]:.2f}, {explained_var[1]:.2f})"
            else:
                vecs_2d = vecs
                x_label, y_label = "e1", "e2"
                title_suffix = ""
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], c=lbls, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Activity Class')
            plt.title(f"Motion Latent Space {title_suffix}")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            plt.savefig("motion_latent_space.png")
            print(">>> Saved visualization to motion_latent_space.png")
            plt.close()
            
            # 2. Bivector Map (Rotor Weights)
            # Weights: [Channels, Num_Bivectors]
            weights = self.model.rotor.bivector_weights.detach().cpu().numpy()
            
            plt.figure(figsize=(10, 4))
            plt.imshow(weights, aspect='auto', cmap='coolwarm', interpolation='nearest')
            plt.colorbar(label='Weight Magnitude')
            plt.title("Learned Bivector Map (Rotor Planes)")
            plt.xlabel("Bivector Basis Index")
            plt.ylabel("Channel")
            plt.savefig("motion_bivector_map.png")
            print(">>> Saved visualization to motion_bivector_map.png")
            plt.close()
            
        except ImportError:
            print("Matplotlib or Sklearn not found, skipping plot.")