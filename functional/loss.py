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
    """Standard Mean Squared Error on Multivector coefficients.

    Equivalent to the squared Euclidean distance in the high-dimensional
    embedding space of the algebra.
    """

    def __init__(self, algebra=None):
        """Initializes the loss function.

        Args:
            algebra: Unused, kept for API consistency.
        """
        super().__init__()
        self.algebra = algebra

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes MSE loss.

        Args:
            pred (torch.Tensor): Predicted multivectors.
            target (torch.Tensor): Target multivectors.

        Returns:
            torch.Tensor: Scalar loss.
        """
        return F.mse_loss(pred, target, reduction='mean')

class SubspaceLoss(nn.Module):
    """Penalizes energy in specific basis components to enforce subspace constraints.

    Used for manifold regularization, e.g., forcing data to lie on a plane (Grade 1)
    by penalizing higher-grade components.

    Attributes:
        penalty_mask (torch.Tensor): Boolean mask of indices to penalize.
    """

    def __init__(self, algebra, target_indices: list = None, exclude_indices: list = None):
        """Initializes the Subspace Loss.

        Args:
            algebra: The algebra instance.
            target_indices (list[int], optional): Indices allowed (no penalty).
            exclude_indices (list[int], optional): Indices penalized.
        
        Raises:
            ValueError: If neither or both index lists are provided.
        """
        super().__init__()
        self.algebra = algebra
        
        if target_indices is not None:
            # Mask is TRUE for indices we want to PENALIZE (i.e., NOT in target)
            mask = torch.ones(algebra.dim, dtype=torch.bool)
            mask[target_indices] = False
        elif exclude_indices is not None:
            mask = torch.zeros(algebra.dim, dtype=torch.bool)
            mask[exclude_indices] = True
        else:
            raise ValueError("Must provide target_indices or exclude_indices")
            
        self.register_buffer('penalty_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy of forbidden components.

        Args:
            x (torch.Tensor): Input multivectors.

        Returns:
            torch.Tensor: Scalar loss.
        """
        penalty_components = x[..., self.penalty_mask]
        loss = (penalty_components ** 2).sum(dim=-1).mean()
        return loss

class IsometryLoss(nn.Module):
    """Ensures transformations preserve the metric norm (Quadratic Form).

    Crucial for learning valid rotors, which must be isometries.
    Loss = MSE( Q(pred), Q(target) ).

    Attributes:
        metric_diag (torch.Tensor): Diagonal of the metric signature.
    """

    def __init__(self, algebra):
        """Initializes Isometry Loss.

        Args:
            algebra: The algebra instance.
        """
        super().__init__()
        self.algebra = algebra
        self.metric_diag = self._compute_metric_diagonal()

    def _compute_metric_diagonal(self):
        """Computes the metric signature (+1 or -1) for each basis blade."""
        basis = torch.eye(self.algebra.dim, device=self.algebra.device)
        sq = self.algebra.geometric_product(basis, basis)
        # The scalar part (index 0) of e_k^2 is the signature
        diag = sq[:, 0]
        return diag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the difference in quadratic form between input and output.

        Args:
            pred (torch.Tensor): Transformed vectors.
            target (torch.Tensor): Original vectors.

        Returns:
            torch.Tensor: Scalar loss.
        """
        pred_sq = (pred ** 2) * self.metric_diag
        target_sq = (target ** 2) * self.metric_diag
        
        pred_norm = pred_sq.sum(dim=-1)
        target_norm = target_sq.sum(dim=-1)
        
        return F.mse_loss(pred_norm, target_norm)

class BivectorRegularization(nn.Module):
    """Regularizes a multivector to be a pure bivector (Grade 2).

    Useful for learning generators of rotations directly.
    """

    def __init__(self, algebra, grade=2):
        """Initializes the regularization.

        Args:
            algebra: The algebra instance.
            grade (int, optional): Target grade. Defaults to 2.
        """
        super().__init__()
        self.algebra = algebra
        self.grade = grade

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the energy of components NOT in the target grade.

        Args:
            x (torch.Tensor): Input multivectors.

        Returns:
            torch.Tensor: Scalar loss.
        """
        target_part = self.algebra.grade_projection(x, self.grade)
        residual = x - target_part
        return (residual ** 2).sum(dim=-1).mean()