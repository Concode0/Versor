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
from layers.linear import CliffordLinear

class MotionManifoldNetwork(nn.Module):
    """Geometric Motion Network. Disentangles motion patterns.

    Projects raw features into GA space, aligns them with a rotor,
    and then classifies based on the resulting vector part.
    """

    def __init__(self, algebra: CliffordAlgebra, input_dim: int, latent_dim: int, num_classes: int = 3):
        """Sets up the motion network.

        Args:
            algebra: Algebra instance.
            input_dim: Raw dimensions.
            latent_dim: Geometric dimensions.
            num_classes: Targets.
        """
        super().__init__()
        self.algebra = algebra
        
        # Standard Linear layer to map raw inputs to multivector coefficients
        self.input_proj = nn.Linear(input_dim, algebra.dim)
        
        # Geometric Alignment Layer
        self.rotor = RotorLayer(algebra, channels=1)
        
        # Classification Head (Linear separability check)
        self.classifier = nn.Linear(algebra.n, num_classes)

    def forward(self, x):
        """Projects input, applies geometric alignment, and classifies.

        Args:
            x: [Batch, Input_Dim]
        """
        # 1. Project to Algebra
        x_mv = self.input_proj(x).unsqueeze(1)
        
        # 2. Geometric Alignment
        aligned = self.rotor(x_mv)
        
        # 3. Extract Grade 1 (Vector part) for latent representation
        vectors = torch.zeros(x.shape[0], self.algebra.n, device=x.device)
        for i in range(self.algebra.n):
            vectors[:, i] = aligned[:, 0, 1 << i]
            
        # 4. Classify
        logits = self.classifier(vectors)
        
        return logits, vectors, aligned