# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from datasets.qm9 import get_qm9_loader, VersorQM9
from models.molecule import MoleculeGNN

class QM9Task(BaseTask):
    """Molecular Property Prediction Task using Geometric GNN."""

    def __init__(self, cfg):
        self.target_name = cfg.dataset.get('target', 'U0')
        self.data_root = "./data/QM9"
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up algebra with optimized signature for molecules."""
        # 1. Sample Data (Load a small subset or 1st graph)
        # We use raw PyG dataset just to get one sample
        dataset = VersorQM9(root=self.data_root)
        sample = dataset[0].pos
        
        # 2. Metric Search
        from core.search import MetricSearch
        searcher = MetricSearch(device=self.device)
        best_p, best_q = searcher.search(sample)
        
        print(f">>> QM9Task: Optimized Signature: Cl({best_p}, {best_q})")
        
        return CliffordAlgebra(p=best_p, q=best_q, device=self.device)

    def setup_model(self):
        return MoleculeGNN(self.algebra, hidden_dim=self.cfg.model.hidden_dim, layers=self.cfg.model.layers)

    def setup_criterion(self):
        return nn.MSELoss()

    def get_data(self):
        loader, mean, std = get_qm9_loader(
            root=self.data_root, 
            target=self.target_name, 
            batch_size=self.cfg.training.batch_size,
            split='train',
            max_samples=self.cfg.dataset.samples
        )
        self.t_mean = torch.tensor(mean, device=self.device)
        self.t_std = torch.tensor(std, device=self.device)
        return loader

    def train_step(self, batch):
        batch = batch.to(self.device)
        batch_z = batch.z
        batch_pos = batch.pos
        batch_idx = batch.batch
        
        # We need to pick the target index.
        target_map = {'mu': 0, 'U0': 7}
        target_idx = target_map.get(self.target_name, 7)
        targets = batch.y[:, target_idx].unsqueeze(-1) # [Batch, 1]
        
        # Target Normalization
        t_mean = targets.mean(dim=0, keepdim=True)
        t_std = targets.std(dim=0, keepdim=True) + 1e-6
        targets_norm = (targets - t_mean) / t_std
        
        self.optimizer.zero_grad()
        
        # Output: [B, D] (Multivector coeffs)
        out_mv = self.model(batch_z, batch_pos, batch_idx)
        
        # Extract relevant part based on target
        if self.target_name == 'U0':
            # Scalar (Grade 0, Index 0)
            # Prediction is the scalar coefficient directly
            pred_raw = out_mv[:, 0:1] 
        elif self.target_name == 'mu': # Case sensitive check, usually lower 'mu' in PyG
            # Vector (Grade 1, Indices 1,2,4 for 3D)
            # Extract vector components
            vec_pred = torch.stack([out_mv[:, 1], out_mv[:, 2], out_mv[:, 4]], dim=-1)
            # Compute magnitude to match QM9 'mu' target (which is a norm)
            pred_raw = torch.norm(vec_pred, dim=-1, keepdim=True)
        else:
            # Fallback
            pred_raw = out_mv[:, 0:1]

        pred_norm = pred_raw
        
        loss = self.criterion(pred_norm, targets_norm)
        loss.backward()
        self.optimizer.step()
        
        # Un-normalize for Metrics
        pred = pred_norm * t_std + t_mean
        
        # Metrics
        mae = torch.abs(pred - targets).mean()
        rmse = torch.sqrt(torch.mean((pred - targets)**2))
        
        return loss.item(), {"MSE": loss.item(), "MAE": mae.item(), "RMSE": rmse.item()}

    def evaluate(self, data):
        pass

    def visualize(self, data):
        """Plots Predicted vs Actual values."""
        # Get a batch
        loader = self.get_data()
        batch = next(iter(loader))
        batch = batch.to(self.device)
        
        batch_z = batch.z
        batch_pos = batch.pos
        batch_idx = batch.batch
        
        target_map = {'mu': 0, 'U0': 7}
        target_idx = target_map.get(self.target_name, 7)
        targets = batch.y[:, target_idx].unsqueeze(-1)
        
        # Normalization Stats (from this batch)
        t_mean = targets.mean(dim=0, keepdim=True)
        t_std = targets.std(dim=0, keepdim=True) + 1e-6
        
        with torch.no_grad():
            out_mv = self.model(batch_z, batch_pos, batch_idx)
            
            if self.target_name == 'U0':
                pred_raw = out_mv[:, 0:1]
            elif self.target_name == 'mu':
                vec_pred = torch.stack([out_mv[:, 1], out_mv[:, 2], out_mv[:, 4]], dim=-1)
                pred_raw = torch.norm(vec_pred, dim=-1, keepdim=True)
            else:
                pred_raw = out_mv[:, 0:1]
            
            # Un-normalize (Assumption: network predicts normalized value)
            pred = pred_raw * t_std + t_mean
                
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            # 1. Prediction Plot
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            
            t = targets.cpu().numpy()
            p = pred.cpu().numpy()
            
            plt.scatter(t, p, alpha=0.5)
            plt.xlabel(f"Actual {self.target_name}")
            plt.ylabel(f"Predicted {self.target_name}")
                
            # Identity line
            min_val = min(t.min(), p.min())
            max_val = max(t.max(), p.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.title(f"QM9 Prediction ({self.target_name})")
            plt.grid(True)
            
            # 2. Embedding PCA
            embed_weights = self.model.embedding.weight.detach().cpu().numpy()
            
            plt.subplot(1, 2, 2)
            if embed_weights.shape[1] > 2:
                pca = PCA(n_components=2)
                embed_2d = pca.fit_transform(embed_weights)
            else:
                embed_2d = embed_weights
                
            atom_labels = ['?', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F']
            for i in range(len(embed_2d)):
                if i < len(atom_labels):
                    plt.scatter(embed_2d[i, 0], embed_2d[i, 1])
                    plt.text(embed_2d[i, 0], embed_2d[i, 1], atom_labels[i], fontsize=12)
            
            plt.title("Atomic Embeddings PCA")
            plt.grid(True)
            
            plt.savefig("qm9_prediction.png")
            print(">>> Saved visualization to qm9_prediction.png")
            plt.close()
            
        except ImportError:
            print("Matplotlib not found.")