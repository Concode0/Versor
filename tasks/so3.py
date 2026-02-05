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
from datasets.modelnet import get_modelnet_loader
from models.invariant import SO3InvariantNet
from torch_geometric.utils import to_dense_batch

class SO3InvariantTask(BaseTask):
    """Task for SO(3)-invariant classification on ModelNet10 data.
    
    Demonstrates generalization from fixed-pose training to arbitrary-pose testing.
    """

    def __init__(self, cfg):
        self.data_root = "./data/ModelNet10"
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up algebra using config values."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        # ModelNet10 has 10 classes
        return SO3InvariantNet(self.algebra, num_classes=10)

    def setup_criterion(self):
        return nn.CrossEntropyLoss()

    def get_data(self):
        # Return TRAIN loader for training loop
        return get_modelnet_loader(
            self.data_root, 
            batch_size=self.cfg.training.batch_size, 
            subset='train', 
            rotated=False # Fixed Pose
        )

    def train_step(self, batch):
        batch = batch.to(self.device)
        pos = batch.pos
        batch_idx = batch.batch
        labels = batch.y.squeeze()
        
        # Convert to Dense [B, N, 3]
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
        pass

    def run(self):
        # Override run to include proper testing on Rotated set
        super().run()
        
        print("\n>>> Testing on Arbitrarily Rotated Data (SO(3))")
        test_loader = get_modelnet_loader(
            self.data_root, 
            batch_size=self.cfg.training.batch_size, 
            subset='test', 
            rotated=True # Arbitrary Rotation
        )
        self.model.eval()
        
        total_acc = 0
        batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                pos = batch.pos
                batch_idx = batch.batch
                labels = batch.y.squeeze()
                
                dense_pos, _ = to_dense_batch(pos, batch_idx)
                
                logits = self.model(dense_pos)
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                total_acc += acc.item()
                batches += 1
                
        avg_acc = total_acc / batches
        print(f"Test Accuracy on Rotated Data: {avg_acc*100:.2f}%")
        
        if avg_acc > 0.8:
            print("SUCCESS: Model is robust to SO(3) rotations.")
        else:
            print("FAILURE: Model struggled with rotations.")

    def visualize(self, data):
        """Visualizes Input and Invariant Features."""
        # Get one sample from Test set (Rotated)
        loader = get_modelnet_loader(self.data_root, batch_size=4, subset='test', rotated=True)
        batch = next(iter(loader))
        batch = batch.to(self.device)
        
        dense_pos, _ = to_dense_batch(batch.pos, batch.batch)
        
        try:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(12, 5))
            
            # 1. Input (Rotated)
            ax1 = fig.add_subplot(121, projection='3d')
            pcl = dense_pos[0].cpu().numpy()
            ax1.scatter(pcl[:,0], pcl[:,1], pcl[:,2], s=1)
            ax1.set_title("Input (Rotated)")
            
            # 2. Invariant Norm Distribution
            ax2 = fig.add_subplot(122)
            # Center and norm
            centered = pcl - pcl.mean(axis=0)
            norms = (centered**2).sum(axis=1)**0.5
            ax2.hist(norms, bins=30, color='skyblue', edgecolor='black')
            ax2.set_title("Invariant Point Norm Distribution")
            ax2.set_xlabel("Distance to Centroid")
            
            plt.savefig("so3_alignment.png")
            print(">>> Saved visualization to so3_alignment.png")
            plt.close()
            
        except ImportError:
            print("Matplotlib not found.")