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
from layers.rotor import RotorLayer
from layers.projection import BladeSelector
from functional.loss import SubspaceLoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer
from datasets.synthetic import Figure8Dataset
from torch.utils.data import DataLoader

class ManifoldNetwork(nn.Module):
    """Neural Network for Manifold Unbending.

    Consists of a learnable Rotor to align the manifold and a BladeSelector to
    filter out noise dimensions.
    """

    def __init__(self, algebra):
        """Initializes the network.

        Args:
            algebra: The algebra instance.
        """
        super().__init__()
        self.rotor = RotorLayer(algebra, channels=1)
        self.selector = BladeSelector(algebra, channels=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input multivectors.

        Returns:
            torch.Tensor: Transformed multivectors.
        """
        x_rot = self.rotor(x)
        return self.selector(x_rot)

class ManifoldTask(BaseTask):
    """Task for restoring a distorted 3D manifold (Figure-8) to a 2D plane.

    Demonstrates the ability of GA rotors to learn optimal geometric transformations
    to simplify data topology.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up 3D Euclidean algebra (p=3, q=0)."""
        return CliffordAlgebra(p=self.cfg.algebra.p, q=self.cfg.algebra.q, device=self.device)

    def setup_model(self):
        """Sets up the ManifoldNetwork."""
        return ManifoldNetwork(self.algebra)

    def setup_criterion(self):
        """Sets up SubspaceLoss to penalize non-vector components."""
        # Calculate indices for Grade 1 (vectors)
        grade_1_indices = []
        for i in range(self.algebra.dim):
            if bin(i).count('1') == 1:
                grade_1_indices.append(i)
                
        return SubspaceLoss(self.algebra, target_indices=grade_1_indices)

    def get_data(self):
        """Generates the Figure-8 dataset."""
        dataset = Figure8Dataset(self.algebra, num_samples=self.cfg.dataset.samples)
        return DataLoader(dataset, batch_size=self.cfg.training.batch_size, shuffle=True)

    def train_step(self, data):
        """Executes one training step."""
        data = data.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        
        # Loss: minimize energy in non-vector grades
        loss = self.criterion(output)
        
        # Explicitly minimize e3 energy (index 4) if applicable
        if self.algebra.dim > 4:
            z_energy = (output[..., 4]**2).mean()
            loss = loss + z_energy
        else:
            z_energy = torch.tensor(0.0)

        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {"Loss": loss.item(), "Z": z_energy.item()}

    def evaluate(self, data):
        """Evaluates reconstruction loss."""
        data = data.to(self.device)
        output = self.model(data)
        loss = self.criterion(output).item()
        print(f"Final Reconstruction Loss: {loss:.6f}")

    def visualize(self, data):
        """Visualizes the original and unbent manifolds."""
        data = data.to(self.device)
        viz = GeneralVisualizer(self.algebra)
        
        viz.plot_3d(data, title="Original Distorted Manifold (Z = 0.5 * X * Y)")
        viz.save("manifold_original.png")
        
        output = self.model(data)
        viz.plot_3d(output, title="Unbent Latent Space (Z -> 0)")
        viz.save("manifold_latent.png")