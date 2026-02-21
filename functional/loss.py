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
        """Initialize the geometric MSE loss."""
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
        """Initialize grade constraint penalties."""
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
    """Isometry loss enforcing metric norm preservation.

    Ensures transformations preserve the metric norm.
    """

    def __init__(self, algebra):
        """Initialize isometry loss with metric diagonal."""
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
    """Bivector regularization enforcing grade-2 purity.

    Penalizes energy outside the target grade (default: grade 2).
    """

    def __init__(self, algebra, grade=2):
        """Initialize bivector regularization."""
        super().__init__()
        self.algebra = algebra
        self.grade = grade

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Penalizes non-bivector parts."""
        target_part = self.algebra.grade_projection(x, self.grade)
        residual = x - target_part
        return (residual ** 2).sum(dim=-1).mean()


class HermitianGradeRegularization(nn.Module):
    """Regularizes grade spectrum toward a target distribution using Hermitian norm.

    Computes the KL-divergence-like loss between the actual grade energy
    distribution and a target spectrum. Encourages the model to distribute
    energy across grades in a physically meaningful way.
    """

    def __init__(self, algebra, target_spectrum=None):
        """Initialize grade regularization.

        Args:
            algebra: CliffordAlgebra instance.
            target_spectrum: [n+1] tensor of desired relative grade energies.
                If None, defaults to uniform distribution.
        """
        super().__init__()
        self.algebra = algebra
        n_grades = algebra.n + 1
        if target_spectrum is None:
            target = torch.ones(n_grades) / n_grades
        else:
            target = torch.tensor(target_spectrum, dtype=torch.float32)
            target = target / (target.sum() + 1e-8)
        self.register_buffer('target', target)

    def forward(self, features):
        """Compute grade regularization loss.

        Args:
            features: Multivector features [..., Channels, Dim].

        Returns:
            Scalar loss (MSE between actual and target grade distribution).
        """
        from core.metric import hermitian_grade_spectrum

        # Flatten all dims except last (Dim) for spectrum computation
        original_shape = features.shape
        flat = features.reshape(-1, original_shape[-1])

        spectrum = hermitian_grade_spectrum(self.algebra, flat)  # [N, n+1]
        # Normalize to distribution
        total = spectrum.sum(dim=-1, keepdim=True) + 1e-8
        dist = spectrum / total  # [N, n+1]

        # Mean distribution across batch
        mean_dist = dist.mean(dim=0)  # [n+1]

        # MSE from target spectrum
        return F.mse_loss(mean_dist, self.target)


class ChamferDistance(nn.Module):
    """Symmetric Chamfer distance between two point clouds.

    CD(P, Q) = (1/|P|) sum_p min_q ||p-q||^2 + (1/|Q|) sum_q min_p ||q-p||^2

    Standard metric for 3D point cloud reconstruction and generation.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Chamfer distance.

        Args:
            pred: Predicted point cloud [B, M, 3].
            target: Target point cloud [B, N, 3].

        Returns:
            Chamfer distance (scalar).
        """
        diff = pred.unsqueeze(2) - target.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        min_dist_pred = dist_sq.min(dim=2)[0].mean(dim=1)
        min_dist_target = dist_sq.min(dim=1)[0].mean(dim=1)
        return (min_dist_pred + min_dist_target).mean()


class ConservativeLoss(nn.Module):
    """Enforces F = -grad(E) conservative force constraint.

    Physics: forces should be the negative gradient of energy
    with respect to atomic positions. Used in molecular dynamics tasks.
    """

    def forward(self, energy: torch.Tensor, force_pred: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        """Compute conservative force loss.

        Args:
            energy: Predicted energy (scalar, requires grad graph).
            force_pred: Predicted forces [N, 3].
            pos: Atom positions [N, 3] (must have requires_grad=True).

        Returns:
            MSE between predicted forces and -grad(E).
        """
        force_from_energy = -torch.autograd.grad(
            energy.sum(), pos,
            create_graph=True, retain_graph=True
        )[0]
        return F.mse_loss(force_pred, force_from_energy)


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss combining MSE with conservation penalty.

    Enforces that global weighted mean of each variable is approximately
    conserved between forecast and target. Used in weather forecasting.
    """

    def __init__(self, physics_weight: float = 0.1):
        super().__init__()
        self.physics_weight = physics_weight

    def forward(self, forecast: torch.Tensor, target: torch.Tensor,
                lat_weights: torch.Tensor = None) -> torch.Tensor:
        """Compute physics-informed loss.

        Args:
            forecast: Predicted state [B, H, W, C].
            target: Target state [B, H, W, C].
            lat_weights: Latitude area weights [H].

        Returns:
            Combined MSE + conservation penalty.
        """
        mse_loss = F.mse_loss(forecast, target)

        if lat_weights is not None and forecast.dim() == 4:
            w = lat_weights.view(1, -1, 1, 1).to(forecast.device)
            forecast_mean = (forecast * w).sum(dim=[1, 2]) / w.sum()
            target_mean = (target * w).sum(dim=[1, 2]) / w.sum()
        else:
            forecast_mean = forecast.mean(dim=list(range(1, forecast.dim() - 1)))
            target_mean = target.mean(dim=list(range(1, target.dim() - 1)))

        conservation_loss = F.mse_loss(forecast_mean, target_mean)
        return mse_loss + self.physics_weight * conservation_loss