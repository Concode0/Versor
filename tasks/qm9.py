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
from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from datasets.qm9 import get_qm9_loaders, VersorQM9
from models.molecule import MoleculeGNN

class QM9Task(BaseTask):
    """Predicting molecules. Standard Graph Neural Network implementation."""

    def __init__(self, cfg):
        self.target_name = cfg.dataset.get('target', 'U0')
        self.data_root = "./data/QM9"
        super().__init__(cfg)

    def setup_algebra(self):
        """Optimized algebra. Let the data decide."""
        # 1. Sample Data
        dataset = VersorQM9(root=self.data_root)
        sample = dataset[0].pos
        
        # 2. Metric Search
        from core.search import MetricSearch
        searcher = MetricSearch(device=self.device)
        best_p, best_q = searcher.search(sample)
        
        print(f">>> QM9Task: Optimized Signature: Cl({best_p}, {best_q})")
        
        return CliffordAlgebra(p=best_p, q=best_q, device=self.device)

    def setup_model(self):
        return MoleculeGNN(self.algebra, hidden_dim=self.cfg.model.hidden_dim, num_layers=self.cfg.model.layers)

    def setup_criterion(self):
        return nn.MSELoss()

    def get_data(self):
        train_loader, val_loader, test_loader, mean, std = get_qm9_loaders(
            root=self.data_root, 
            target=self.target_name, 
            batch_size=self.cfg.training.batch_size,
            max_samples=self.cfg.dataset.samples
        )
        self.t_mean = torch.tensor(mean, device=self.device)
        self.t_std = torch.tensor(std, device=self.device)
        return train_loader, val_loader, test_loader

    def train_step(self, batch):
        batch = batch.to(self.device)
        batch_z = batch.z
        batch_pos = batch.pos
        batch_idx = batch.batch
        
        target_map = {'mu': 0, 'U0': 7}
        target_idx = target_map.get(self.target_name, 7)
        targets = batch.y[:, target_idx].unsqueeze(-1)
        
        targets_norm = (targets - self.t_mean) / (self.t_std + 1e-6)
        
        self.optimizer.zero_grad()
        out_mv = self.model(batch_z, batch_pos, batch_idx, batch.edge_index).unsqueeze(-1)
        
        loss = self.criterion(out_mv, targets_norm)
        loss.backward()
        self.optimizer.step()
        
        pred = out_mv * self.t_std + self.t_mean
        mae = torch.abs(pred - targets).mean()
        
        return loss.item(), {"MSE": loss.item(), "MAE": mae.item()}

    def evaluate(self, val_loader):
        """Standard metrics."""
        self.model.eval()
        total_mae = 0
        count = 0
        target_map = {'mu': 0, 'U0': 7}
        target_idx = target_map.get(self.target_name, 7)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                targets = batch.y[:, target_idx].unsqueeze(-1)
                pred_norm = self.model(batch.z, batch.pos, batch.batch, batch.edge_index).unsqueeze(-1)
                pred = pred_norm * self.t_std + self.t_mean
                total_mae += torch.abs(pred - targets).sum().item()
                count += targets.size(0)
        
        avg_mae = total_mae / count
        # Don't print here to keep output clean during loops, or print conditionally.
        # But for now, let's keep it simple.
        return avg_mae

    def visualize(self, val_loader):
        """Scatter plot. Truth vs Prediction."""
        self.model.eval()
        batch = next(iter(val_loader))
        batch = batch.to(self.device)
        
        target_map = {'mu': 0, 'U0': 7}
        target_idx = target_map.get(self.target_name, 7)
        targets = batch.y[:, target_idx].unsqueeze(-1)
        
        with torch.no_grad():
            pred_norm = self.model(batch.z, batch.pos, batch.batch, batch.edge_index).unsqueeze(-1)
            pred = pred_norm * self.t_std + self.t_mean
                
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 5))
            t = targets.cpu().numpy()
            p = pred.cpu().numpy()
            plt.scatter(t, p, alpha=0.5, label='MoleculeGNN Pred')
            plt.xlabel(f"Actual {self.target_name}")
            plt.ylabel(f"Predicted {self.target_name}")
            min_val = min(t.min(), p.min())
            max_val = max(t.max(), p.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.title(f"QM9 Prediction ({self.target_name})")
            plt.grid(True)
            plt.legend()
            plt.savefig("qm9_prediction.png")
            print(">>> Saved visualization to qm9_prediction.png")
            plt.close()
        except ImportError:
            print("Matplotlib not found.")

    def run(self):
        """Runs train and val."""
        print(f">>> Starting Task: {self.cfg.name}")
        train_loader, val_loader, test_loader = self.get_data()
        
        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))
        
        best_val_mae = float('inf')
        
        for epoch in pbar:
            self.model.train()
            total_loss = 0
            total_mae = 0
            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                total_mae += logs['MAE']
            
            avg_loss = total_loss / len(train_loader)
            avg_mae = total_mae / len(train_loader)
            
            val_mae = self.evaluate(val_loader)
            self.scheduler.step(val_mae)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                self.save_checkpoint(f"{self.cfg.name}_best.pt")
            
            logs = {
                'Loss': avg_loss,
                'MAE': avg_mae,
                'Val_MAE': val_mae,
                'LR': self.optimizer.param_groups[0]['lr']
            }
            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

        print(f">>> Training Complete. Best Val MAE: {best_val_mae:.4f}")
        
        # Load best model for final test
        print(">>> Loading best model for Test Set evaluation...")
        self.load_checkpoint(f"{self.cfg.name}_best.pt")
        
        test_mae = self.evaluate(test_loader)
        print(f">>> FINAL TEST MAE: {test_mae:.4f}")
        
        self.visualize(test_loader)