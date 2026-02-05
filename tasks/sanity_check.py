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

class IdentityNetwork(nn.Module):
    """Simple identity network with a learnable rotor.

    Used to verify that the network can learn an identity mapping or simple
    transformation on random noise.
    """

    def __init__(self, algebra):
        """Initializes the network.

        Args:
            algebra: The algebra instance.
        """
        super().__init__()
        self.rotor = RotorLayer(algebra, channels=1)

    def forward(self, x):
        """Forward pass."""
        return self.rotor(x)

class SanityCheckTask(BaseTask):
    """Sanity Check: Can the model learn the identity on random noise?

    Verifies that gradients flow and the algebra backend is numerically stable
    with unstructured input.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """Sets up Cl(3, 0)."""
        return CliffordAlgebra(p=3, q=0, device=self.device)

    def setup_model(self):
        """Sets up the IdentityNetwork."""
        return IdentityNetwork(self.algebra)

    def setup_criterion(self):
        """Sets up Geometric MSE Loss."""
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        """Generates random noise (Input) and exact same noise (Target)."""
        n = 1000
        # Random multivectors
        data = torch.randn(n, 1, self.algebra.dim, device=self.device)
        return data

    def train_step(self, data):
        """Training step (Autoencoder task)."""
        self.optimizer.zero_grad()
        output = self.model(data)
        
        # Loss: reconstruction of self
        # Rotor should learn to be Identity (or close to it)
        loss = self.criterion(output, data)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {}

    def evaluate(self, data):
        """Evaluates reconstruction."""
        output = self.model(data)
        loss = self.criterion(output, data).item()
        print(f"Final Sanity Loss: {loss:.6f}")

    def visualize(self, data):
        """Visualizes PCA of noise (should be preserved)."""
        output = self.model(data)
        viz = GeneralVisualizer(self.algebra)
        
        viz.plot_latent_projection(data, title="Input Noise")
        viz.save("sanity_noise_pca.png")