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
from datasets.modelnet import get_modelnet_loader
from models.invariant import SO3InvariantNet
from torch_geometric.utils import to_dense_batch

class SO3InvariantTask(BaseTask):
    """SO(3) Invariance. Spin it, I don't care.

    Demonstrates robust classification regardless of orientation.
    """

    def __init__(self, cfg):
        self.data_root = "./data/ModelNet10"
        super().__init__(cfg)

    def setup_algebra(self):
        """Standard 3D."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        """Invariant Net."""
        return SO3InvariantNet(self.algebra, num_classes=10)

    def setup_criterion(self):
        return nn.CrossEntropyLoss()

    def get_data(self):
        """Loads fixed-pose training data."""
        return get_modelnet_loader(
            self.data_root, 
            batch_size=self.cfg.training.batch_size, 
            subset='train', 
            rotated=False 
        )

    def train_step(self, batch):
        """Learn from fixed poses."""
        batch = batch.to(self.device)
        pos = batch.pos
        batch_idx = batch.batch
        labels = batch.y.squeeze()
        
        dense_pos, mask = to_dense_batch(pos, batch_idx)
        
        self.optimizer.zero_grad()
        logits = self.model(dense_pos)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        return loss.item(), {"Loss": loss.item(), "Acc": acc.item()}

    def evaluate(self, data):
        """Spinning the test set. Does it break?"""
        print("\n>>> Evaluating on Arbitrarily Rotated Data (SO(3) Robustness)")
        test_loader = get_modelnet_loader(
            self.data_root, 
            batch_size=self.cfg.training.batch_size, 
            subset='test', 
            rotated=True 
        )
        self.model.eval()
        
        total_acc = 0
        batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                dense_pos, _ = to_dense_batch(batch.pos, batch.batch)
                logits = self.model(dense_pos)
                preds = logits.argmax(dim=1)
                acc = (preds == batch.y.squeeze()).float().mean()
                total_acc += acc.item()
                batches += 1
                
        avg_acc = total_acc / batches
        print(f"Test Accuracy on Rotated Data: {avg_acc*100:.2f}%")

    def visualize(self, data):
        """Shows the invariance."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get one rotated sample
            loader = get_modelnet_loader(self.data_root, batch_size=1, subset='test', rotated=True)
            batch = next(iter(loader)).to(self.device)
            dense_pos, _ = to_dense_batch(batch.pos, batch.batch)
            pcl = dense_pos[0].cpu().numpy()

            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(pcl[:,0], pcl[:,1], pcl[:,2], s=2, alpha=0.5)
            ax1.set_title("Input (Rotated ModelNet10)")
            
            ax2 = fig.add_subplot(122)
            norms = np.linalg.norm(pcl - pcl.mean(axis=0), axis=1)
            ax2.hist(norms, bins=30, color='skyblue', alpha=0.7)
            ax2.set_title("Invariant Shape Signature (Point Norms)")
            
            plt.savefig("so3_alignment.png")
            plt.close()
            print(">>> Saved visualization to so3_alignment.png")
            
        except ImportError:
            pass
