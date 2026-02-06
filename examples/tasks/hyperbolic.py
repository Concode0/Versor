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
from layers.rotor import RotorLayer
from functional.loss import GeometricMSELoss
from tasks.base import BaseTask
from core.visualizer import GeneralVisualizer
import matplotlib.pyplot as plt

class HyperbolicNetwork(nn.Module):
    """Lorentz Booster. Faster than light.

    Learns hyperbolic transformations in Cl(1, 1).
    """

    def __init__(self, algebra):
        """Sets up the network."""
        super().__init__()
        self.rotor = RotorLayer(algebra, channels=1)

    def forward(self, x):
        """Forward pass."""
        return self.rotor(x)

class HyperbolicTask(BaseTask):
    """Time travel. Reversing the boost.

    Recovers the original frame from a Lorentz-boosted nightmare.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """2D Spacetime Cl(1, 1)."""
        return CliffordAlgebra(p=1, q=1, device=self.device)

    def setup_model(self):
        """The Booster."""
        return HyperbolicNetwork(self.algebra)

    def setup_criterion(self):
        """Geometric MSE."""
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        """Simulates relativity."""
        # 1. Random events (t, x)
        n = 100
        x = torch.randn(n, 1, self.algebra.dim, device=self.device)
        
        # Mask only vector parts
        mask = torch.tensor([0, 1, 1, 0], dtype=torch.bool, device=self.device)
        data = torch.zeros_like(x)
        data[..., mask] = x[..., mask]
        
        # 2. Apply Lorentz Boost
        phi = 1.5
        B = torch.zeros(1, self.algebra.dim, device=self.device)
        B[0, 3] = phi
        
        self.target_rotor = self.algebra.exp(-0.5 * B)
        self.target_rotor_rev = self.algebra.reverse(self.target_rotor)
        
        data_boosted = self.algebra.geometric_product(self.target_rotor.expand_as(data), data)
        data_boosted = self.algebra.geometric_product(data_boosted, self.target_rotor_rev.expand_as(data))
        
        return data_boosted, data

    def train_step(self, data):
        """Reverse the boost."""
        input_data, target_data = data
        self.optimizer.zero_grad()
        output = self.model(input_data)
        loss = self.criterion(output, target_data)
        loss.backward()
        self.optimizer.step()
        return loss.item(), {}

    def evaluate(self, data):
        """Did we break physics?"""
        learned_rotor = self.model.rotor.bivector_weights
        print(f"True Phi: 1.5")
        print(f"Learned Rotor Weights: {learned_rotor.detach().cpu().numpy().flatten()}")
        
        input_data, target_data = data
        output = self.model(input_data)
        loss = self.criterion(output, target_data).item()
        print(f"Final Reconstruction Loss: {loss:.6f}")

    def visualize(self, data):
        """Draws light cones."""
        input_data, target_data = data
        output = self.model(input_data)
        
        viz = GeneralVisualizer(self.algebra)
        
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