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
from datasets.modelnet import ModelNetSynthetic
from models.invariant import SO3InvariantNet
from torch.utils.data import DataLoader

class SO3InvariantTask(BaseTask):
    """Task for SO(3)-invariant classification on ModelNet40-like data.
    
    Demonstrates generalization from fixed-pose training to arbitrary-pose testing.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        return CliffordAlgebra(p=3, q=0, device=self.device)

    def setup_model(self):
        return SO3InvariantNet(self.algebra, num_classes=3)

    def setup_criterion(self):
        return nn.CrossEntropyLoss()

    def get_data(self):
        # We need separate train and test loaders
        # But BaseTask only supports one 'get_data'.
        # We'll return the TRAIN loader here, and create the TEST loader manually in run() 
        # or overload run() or evaluate().
        # Let's override run() logic implicitly by creating attributes.
        
        self.train_dataset = ModelNetSynthetic(
            self.algebra, 
            num_samples=self.cfg.dataset.samples, 
            subset='train' # Fixed Pose
        )
        self.test_dataset = ModelNetSynthetic(
            self.algebra, 
            num_samples=200, 
            subset='test' # Random Rotation
        )
        
        return DataLoader(self.train_dataset, batch_size=self.cfg.training.batch_size, shuffle=True)

    def train_step(self, batch):
        data, labels = batch
        data, labels = data.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        self.optimizer.step()
        
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        return loss.item(), {"Loss": loss.item(), "Acc": acc.item()}

    def evaluate(self, data):
        # This default evaluate is called with a sample from train loader in BaseTask.run()
        # We want to run a full test set evaluation.
        pass

    def run(self):
        # Override run to include proper testing on Rotated set
        super().run()
        
        print("\n>>> Testing on Arbitrarily Rotated Data (SO(3))")
        test_loader = DataLoader(self.test_dataset, batch_size=self.cfg.training.batch_size)
        self.model.eval()
        
        total_acc = 0
        batches = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                logits = self.model(data)
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
        pass
