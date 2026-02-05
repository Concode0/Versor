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
from datasets.qm9 import QM9Dataset
from models.molecule import MoleculeGNN
from torch.utils.data import DataLoader

class QM9Task(BaseTask):
    """Molecular Property Prediction Task using Geometric GNN."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.target_name = cfg.dataset.get('target', 'U0')

    def setup_algebra(self):
        # 3D Euclidean Algebra
        return CliffordAlgebra(p=3, q=0, device=self.device)

    def setup_model(self):
        return MoleculeGNN(self.algebra, hidden_dim=self.cfg.model.hidden_dim, layers=self.cfg.model.layers)

    def setup_criterion(self):
        return nn.MSELoss()

    def get_data(self):
        dataset = QM9Dataset(self.algebra, num_samples=self.cfg.dataset.samples, target=self.target_name)
        return DataLoader(
            dataset, 
            batch_size=self.cfg.training.batch_size, 
            shuffle=True, 
            collate_fn=dataset.collate_fn
        )

    def train_step(self, batch):
        batch_z = batch['z'].to(self.device)
        batch_pos = batch['pos'].to(self.device)
        batch_idx = batch['batch'].to(self.device)
        targets = batch['y'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Output: [B, D] (Multivector coeffs)
        out_mv = self.model(batch_z, batch_pos, batch_idx)
        
        # Extract relevant part based on target
        if self.target_name == 'U0':
            # Scalar (Grade 0, Index 0)
            pred = out_mv[:, 0:1] # [B, 1]
        elif self.target_name == 'Mu':
            # Vector (Grade 1, Indices 1,2,4 for 3D)
            # We need to extract these specifically
            pred = torch.zeros_like(targets)
            pred[:, 0] = out_mv[:, 1]
            pred[:, 1] = out_mv[:, 2]
            pred[:, 2] = out_mv[:, 4]
        
        loss = self.criterion(pred, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"MSE": loss.item()}

    def evaluate(self, data):
        pass

    def visualize(self, data):
        pass
