# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers import RotorLayer, BladeSelector, CliffordModule
from functional.loss import SubspaceLoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer
from examples.datasets.synthetic import Figure8Dataset
from torch.utils.data import DataLoader

class ManifoldNetwork(CliffordModule):
    """The Unbender.

    Aligns the manifold and filters the noise.
    """

    def __init__(self, algebra):
        """Sets up the network."""
        super().__init__(algebra)
        self.rotor = RotorLayer(algebra, channels=1)
        self.selector = BladeSelector(algebra, channels=1)

    def forward(self, x):
        """Forward pass."""
        x_rot = self.rotor(x)
        return self.selector(x_rot)

class ManifoldTask(BaseTask):
    """Manifold Unbending. Flattening the manifold.

    Restores a distorted 3D manifold to its planar truth.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """3D Euclidean."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        """The Unbender."""
        return ManifoldNetwork(self.algebra)

    def setup_criterion(self):
        """Subspace Loss. Only Grade 1 allowed."""
        grade_1_indices = []
        for i in range(self.algebra.dim):
            if bin(i).count('1') == 1:
                grade_1_indices.append(i)
                
        return SubspaceLoss(self.algebra, target_indices=grade_1_indices)

    def get_data(self):
        """Figure-8 dataset."""
        dataset = Figure8Dataset(self.algebra, num_samples=self.cfg.dataset.samples)
        return DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=True)

    def inject_noise(self, data):
        """Adds Gaussian noise to the multivectors.
        
        Only injects noise if dataset.noise_std > 0.
        """
        noise_std = self.cfg.dataset.get("noise_std", 0.0)
        if noise_std > 0:
            noise = torch.randn_like(data) * noise_std
            return data + noise
        return data

    def train_step(self, data):
        """Flatten it."""
        data = data.to(self.device)
        data = self.inject_noise(data)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        
        loss = self.criterion(output)
        
        if self.algebra.dim > 4:
            z_energy = (output[..., 4]**2).mean()
            loss = loss + z_energy
        else:
            z_energy = torch.tensor(0.0)

        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Loss": loss.item(), "Z": z_energy.item()}

    def evaluate(self, data):
        """How flat is it?"""
        data = data.to(self.device)
        output = self.model(data)
        loss = self.criterion(output).item()
        print(f"Final Reconstruction Loss: {loss:.6f}")
        
        # Run noise robustness test
        self.noise_test(data)

    def noise_test(self, data):
        """Tests model robustness under increasing noise levels."""
        print("\n--- Noise Robustness Test ---")
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        for level in noise_levels:
            noise = torch.randn_like(data) * level
            noisy_data = data + noise
            with torch.no_grad():
                output = self.model(noisy_data)
                loss = self.criterion(output).item()
                print(f"Noise Std: {level:.2f} | Loss: {loss:.6f}")
        print("-----------------------------\n")

    def visualize(self, data):
        """Plots the evidence."""
        data = data.to(self.device)
        viz = GeneralVisualizer(self.algebra)
        
        viz.plot_3d(data, title="Original Distorted Manifold (Z = 0.5 * X * Y)")
        viz.save("manifold_original.png")
        
        output = self.model(data)
        viz.plot_3d(output, title="Unbent Latent Space (Z -> 0)")
        viz.save("manifold_latent.png")