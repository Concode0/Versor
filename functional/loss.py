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
import torch.nn.functional as F

class GeometricMSELoss(nn.Module):
    """Geometric MSE. Euclidean distance in embedding space.

    Standard MSE on coefficients.
    """

    def __init__(self, algebra=None):
        """Sets up the loss."""
        super().__init__()
        self.algebra = algebra

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE."""
        return F.mse_loss(pred, target, reduction='mean')

class SubspaceLoss(nn.Module):
    """Subspace Loss. Enforces grade constraints.

    Penalizes energy in forbidden grades.
    """

    def __init__(self, algebra, target_indices: list = None, exclude_indices: list = None):
        """Sets up the penalties."""
        super().__init__()
        self.algebra = algebra
        
        if target_indices is not None:
            mask = torch.ones(algebra.dim, dtype=torch.bool)
            mask[target_indices] = False
        elif exclude_indices is not None:
            mask = torch.zeros(algebra.dim, dtype=torch.bool)
            mask[exclude_indices] = True
        else:
            raise ValueError("Must provide target_indices or exclude_indices")
            
        self.register_buffer('penalty_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Penalizes deviations."""
        penalty_components = x[..., self.penalty_mask]
        loss = (penalty_components ** 2).sum(dim=-1).mean()
        return loss

class IsometryLoss(nn.Module):
    """Isometry Loss. Don't warp the space.

    Ensures transformations preserve the metric norm.
    """

    def __init__(self, algebra):
        """Sets up the isometry check."""
        super().__init__()
        self.algebra = algebra
        self.metric_diag = self._compute_metric_diagonal()

    def _compute_metric_diagonal(self):
        """Finds the signature."""
        basis = torch.eye(self.algebra.dim, device=self.algebra.device)
        sq = self.algebra.geometric_product(basis, basis)
        diag = sq[:, 0]
        return diag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compares norms."""
        pred_sq = (pred ** 2) * self.metric_diag
        target_sq = (target ** 2) * self.metric_diag
        
        pred_norm = pred_sq.sum(dim=-1)
        target_norm = target_sq.sum(dim=-1)
        
        return F.mse_loss(pred_norm, target_norm)

class BivectorRegularization(nn.Module):
    """Bivector Regularization. Be a rotation.

    Forces multivectors to be pure bivectors (Grade 2).
    """

    def __init__(self, algebra, grade=2):
        """Sets up the reg."""
        super().__init__()
        self.algebra = algebra
        self.grade = grade

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Penalizes non-bivector parts."""
        target_part = self.algebra.grade_projection(x, self.grade)
        residual = x - target_part
        return (residual ** 2).sum(dim=-1).mean()