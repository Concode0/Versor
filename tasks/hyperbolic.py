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
from layers.rotor import RotorLayer
from functional.loss import GeometricMSELoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer
import matplotlib.pyplot as plt

class HyperbolicNetwork(nn.Module):
    """Network for learning hyperbolic transformations (Lorentz Boosts).

    Uses a RotorLayer in a mixed-signature algebra Cl(1, 1) or Cl(1, 3).
    """

    def __init__(self, algebra):
        """Initializes the network.

        Args:
            algebra: The algebra instance.
        """
        super().__init__()
        # Learn a single rotor (Lorentz transformation)
        self.rotor = RotorLayer(algebra, channels=1)

    def forward(self, x):
        """Forward pass."""
        return self.rotor(x)

class HyperbolicTask(BaseTask):
    """Task for reversing a Lorentz Boost in Minkowski Spacetime.

    Demonstrates the capability to handle non-Euclidean metrics (q > 0).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up 2D Spacetime algebra Cl(1, 1)."""
        return CliffordAlgebra(p=1, q=1, device=self.device)

    def setup_model(self):
        """Sets up the HyperbolicNetwork."""
        return HyperbolicNetwork(self.algebra)

    def setup_criterion(self):
        """Sets up Geometric MSE Loss."""
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        """Generates spacetime events and applies a Lorentz boost."""
        # 1. Random events in 2D spacetime (t, x)
        # e1 (time, +), e2 (space, -)
        n = 100
        x = torch.randn(n, 1, self.algebra.dim, device=self.device)
        
        # Mask only vector parts (index 1 and 2)
        # 0: scalar, 1: e1, 2: e2, 3: e12
        mask = torch.tensor([0, 1, 1, 0], dtype=torch.bool, device=self.device)
        data = torch.zeros_like(x)
        data[..., mask] = x[..., mask]
        
        # 2. Apply a known Lorentz Boost
        # Boost parameter phi (rapidity)
        phi = 1.5
        # Bivector B = phi * e1e2 (index 3)
        B = torch.zeros(1, self.algebra.dim, device=self.device)
        B[0, 3] = phi
        
        # R = exp(-B/2) -> cosh(phi/2) - sinh(phi/2) e1e2
        self.target_rotor = self.algebra.exp(-0.5 * B)
        self.target_rotor_rev = self.algebra.reverse(self.target_rotor)
        
        # Boosted data: x' = R x R~
        data_boosted = self.algebra.geometric_product(self.target_rotor.expand_as(data), data)
        data_boosted = self.algebra.geometric_product(data_boosted, self.target_rotor_rev.expand_as(data))
        
        # Task: Given boosted data, recover original (Reverse the boost)
        # Input: Boosted, Target: Original
        return data_boosted, data

    def train_step(self, data):
        """Training step to recover original frame."""
        input_data, target_data = data
        self.optimizer.zero_grad()
        output = self.model(input_data)
        loss = self.criterion(output, target_data)
        loss.backward()
        self.optimizer.step()
        return loss.item(), {}

    def evaluate(self, data):
        """Evaluates parameter recovery."""
        # Check learned rotor vs inverse of target rotor
        learned_rotor = self.model.rotor.bivector_weights
        print(f"True Phi: 1.5")
        print(f"Learned Rotor Weights: {learned_rotor.detach().cpu().numpy().flatten()}")
        
        input_data, target_data = data
        output = self.model(input_data)
        loss = self.criterion(output, target_data).item()
        print(f"Final Reconstruction Loss: {loss:.6f}")

    def visualize(self, data):
        """Visualizes spacetime diagrams."""
        input_data, target_data = data
        output = self.model(input_data)
        
        viz = GeneralVisualizer(self.algebra)
        
        # We need to plot (t, x) components
        # e1 index 1, e2 index 2
        
        def extract_tx(tensor):
            t = tensor[..., 1].detach().cpu().numpy().flatten()
            x = tensor[..., 2].detach().cpu().numpy().flatten()
            return t, x
            
        t_orig, x_orig = extract_tx(target_data)
        t_boost, x_boost = extract_tx(input_data)
        t_rec, x_rec = extract_tx(output)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x_orig, t_orig, label="Original (Rest)", alpha=0.6)
        plt.scatter(x_boost, t_boost, label="Boosted (Input)", alpha=0.6)
        plt.scatter(x_rec, t_rec, label="Recovered", marker='x', alpha=0.6)
        
        # Draw light cones
        plt.plot([-3, 3], [-3, 3], 'k--', alpha=0.3, label="Light Cone")
        plt.plot([-3, 3], [3, -3], 'k--', alpha=0.3)
        
        plt.xlabel("Space (x)")
        plt.ylabel("Time (t)")
        plt.title("Lorentz Boost Recovery in Cl(1, 1)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig("hyperbolic_viz.png")
        print("Saved hyperbolic visualization.")