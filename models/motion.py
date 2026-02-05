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
from layers.linear import CliffordLinear

class MotionManifoldNetwork(nn.Module):
    """Network for aligning complex motion data in a low-dimensional geometric space.

    Architecture:
    1. Feature Projector: Linear map from raw motion features (45 dims) to GA space (e.g. 4D, 16 basis).
    2. Rotor Alignment: Learnable rotors to disentangle class manifolds.
    3. Metric Projector: Maps back to a Euclidean space for classification/visualization.
    """

    def __init__(self, algebra: CliffordAlgebra, input_dim: int, latent_dim: int, num_classes: int = 3):
        """
        Args:
            algebra: Clifford Algebra instance.
            input_dim: Raw feature dimension (e.g., 45).
            latent_dim: Dimension of the GA space (e.g., 4 means Cl(4,0)).
            num_classes: Number of target classes.
        """
        super().__init__()
        self.algebra = algebra
        
        # Standard Linear layer to map raw inputs to multivector coefficients
        # Input: [Batch, Input_Dim] -> Output: [Batch, 1, Algebra_Dim]
        # We treat the algebra as having 1 channel.
        self.input_proj = nn.Linear(input_dim, algebra.dim)
        
        # Geometric Alignment Layer
        self.rotor = RotorLayer(algebra, channels=1)
        
        # Classification Head (Linear separability check)
        # We extract vector part (grade 1) for classification
        self.classifier = nn.Linear(algebra.n, num_classes)

    def forward(self, x):
        """
        Args:
            x: [Batch, Input_Dim]
        """
        # 1. Project to Algebra
        # x_mv: [Batch, 1, Algebra_Dim]
        x_mv = self.input_proj(x).unsqueeze(1)
        
        # 2. Geometric Alignment
        # aligned: [Batch, 1, Algebra_Dim]
        aligned = self.rotor(x_mv)
        
        # 3. Extract Grade 1 (Vector part) for latent representation
        # vectors: [Batch, Algebra_N]
        # We need to gather indices corresponding to basis vectors e1...en
        vectors = torch.zeros(x.shape[0], self.algebra.n, device=x.device)
        for i in range(self.algebra.n):
            vectors[:, i] = aligned[:, 0, 1 << i]
            
        # 4. Classify
        logits = self.classifier(vectors)
        
        return logits, vectors, aligned
