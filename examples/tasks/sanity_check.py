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
    """The identity network. Verifies pass-through capability."""

    def __init__(self, algebra):
        """Sets up the network."""
        super().__init__()
        self.rotor = RotorLayer(algebra, channels=1)

    def forward(self, x):
        """Pass through."""
        return self.rotor(x)

class SanityCheckTask(BaseTask):
    """Sanity Check. Verifies algebraic consistency.

    Tests if the model can learn the identity function f(x) = x.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_algebra(self):
        """Standard 3D Euclidean."""
        return CliffordAlgebra(p=3, q=0, device=self.device)

    def setup_model(self):
        """Identity Net."""
        return IdentityNetwork(self.algebra)

    def setup_criterion(self):
        """Geometric MSE."""
        return GeometricMSELoss(self.algebra)

    def get_data(self):
        """Random noise input."""
        n = 1000
        data = torch.randn(n, 1, self.algebra.dim, device=self.device)
        return data

    def train_step(self, data):
        """Learn identity."""
        self.optimizer.zero_grad()
        output = self.model(data)
        
        loss = self.criterion(output, data)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), {}

    def evaluate(self, data):
        """Evaluates identity learning."""
        output = self.model(data)
        loss = self.criterion(output, data).item()
        print(f"Final Sanity Loss: {loss:.6f}")

    def visualize(self, data):
        """Plots the input distribution."""
        viz = GeneralVisualizer(self.algebra)
        viz.plot_latent_projection(data, title="Input Noise")
        viz.save("sanity_noise_pca.png")