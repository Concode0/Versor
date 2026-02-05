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
from layers.base import CliffordModule

class SO3InvariantNet(CliffordModule):
    """A Neural Network that is invariant to SO(3) rotations using Pure Geometric Invariants.

    Instead of aligning to a canonical frame (which suffers from axis permutation and 
    flip ambiguities), this network computes features that are rotationally invariant 
    by definition using Geometric Products and Norms.
    """

    def __init__(self, algebra: CliffordAlgebra, num_classes: int = 3):
        super().__init__(algebra)
        
        # Features: 
        # 1. Point Norm (1)
        # 2. Covariance Eigenvalues (3) - Global shape descriptors
        # 3. Projection magnitudes (3) - Invariant scalars from Geometric Product
        # Total: 7 features
        
        self.mlp1 = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, N, 3] (Cartesian coordinates)
        """
        B, N, _ = x.shape
        
        # 1. Centering (Translation Invariance)
        mean = x.mean(dim=1, keepdim=True)
        x_centered = x - mean
        
        # 2. Global Shape Descriptor: Covariance Spectrum (Rotation Invariant)
        cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / N
        evals, evecs = torch.linalg.eigh(cov) # evals: [B, 3], evecs: [B, 3, 3]
        
        # 3. Local Invariant: Point Norms
        norms = torch.norm(x_centered, dim=2, keepdim=True) # [B, N, 1]
        
        # 4. Geometric Product Invariants
        # We use the principal axes 'evecs' as a robust reference frame.
        # Although evecs have sign ambiguity, the magnitudes of dot/wedge products are invariant.
        
        # Dot Products (Magnitudes) |x_i . u_k|
        # proj: [B, N, 3]
        proj = torch.bmm(x_centered, evecs)
        abs_proj = torch.abs(proj) 
        
        # Wedge Product Magnitudes ||x_i ^ u_k||
        # In 3D, ||x ^ u|| = ||x|| ||u|| sin(theta). Since ||u||=1, it's ||x|| sin(theta).
        # This is sqrt(||x||^2 - (x.u)^2).
        # These are also rotation and flip invariant.
        
        wedge_sq = norms**2 - proj**2
        abs_wedge = torch.sqrt(torch.clamp(wedge_sq, min=1e-8)) # [B, N, 3]
        
        # 5. Assemble Pure Invariant Features
        # Using global eigenvalues (3) and local magnitudes (1 norm + 3 projections)
        # Or alternatively: Global (3 evals) + Local (1 norm + 3 abs_proj) = 7
        
        global_feats = evals.unsqueeze(1).expand(-1, N, -1) # [B, N, 3]
        
        # Combined: [B, N, 7]
        features = torch.cat([norms, global_feats, abs_proj], dim=2)
        
        # 6. PointNet Classifier
        feat_emb = self.mlp1(features)
        global_feat, _ = torch.max(feat_emb, dim=1)
        logits = self.mlp2(global_feat)
        
        return logits