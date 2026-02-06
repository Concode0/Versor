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
from layers.base import CliffordModule

class SO3InvariantNet(CliffordModule):
    """Physics doesn't care about your frame. Pure invariants.

    Computes features that are rotationally invariant by definition.
    No more data augmentation needed.
    """

    def __init__(self, algebra: CliffordAlgebra, num_classes: int = 3):
        super().__init__(algebra)
        
        # Features: 
        # 1. Point Norm (1)
        # 2. Covariance Eigenvalues (3)
        # 3. Projection magnitudes (3)
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
        """Center, Covariance, Invariants, PointNet. Standard recipe.

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
        # Dot Products (Magnitudes) |x_i . u_k|
        proj = torch.bmm(x_centered, evecs)
        abs_proj = torch.abs(proj) 
        
        # Wedge Product Magnitudes ||x_i ^ u_k||
        # sqrt(||x||^2 - (x.u)^2)
        wedge_sq = norms**2 - proj**2
        abs_wedge = torch.sqrt(torch.clamp(wedge_sq, min=1e-8)) # [B, N, 3]
        
        # 5. Assemble Pure Invariant Features
        global_feats = evals.unsqueeze(1).expand(-1, N, -1) # [B, N, 3]
        
        # Combined: [B, N, 7]
        features = torch.cat([norms, global_feats, abs_proj], dim=2)
        
        # 6. PointNet Classifier
        feat_emb = self.mlp1(features)
        global_feat, _ = torch.max(feat_emb, dim=1)
        logits = self.mlp2(global_feat)
        
        return logits