# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Geometric GA activations.

Magnitude-scaling and grade-wise gating functions that preserve geometric structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricGELU(nn.Module):
    """Geometric GELU activation: x' = x * GELU(||x|| + b) / ||x||.

    Scales magnitude while preserving direction.

    Attributes:
        algebra (CliffordAlgebra): The algebra instance.
        bias (torch.nn.Parameter): Learnable bias added to norm.
    """

    def __init__(self, algebra, channels: int = 1):
        """Initialize Geometric GELU.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of channels.
        """
        super().__init__()
        self.algebra = algebra
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply geometric GELU activation.

        Args:
            x (torch.Tensor): Input multivector [..., Dim].

        Returns:
            torch.Tensor: Activated multivector.
        """
        norm = x.norm(dim=-1, keepdim=True)
        
        eps = 1e-6
        scale = F.gelu(norm + self.bias.view(1, -1, 1)) / (norm + eps)
        
        return x * scale


class GradeSwish(nn.Module):
    """Per-grade gated activation.

    Each grade receives an independent sigmoid gate based on its norm.

    Attributes:
        algebra (CliffordAlgebra): The algebra instance.
        n_grades (int): Number of grades.
        grade_weights (torch.nn.Parameter): Weights for each grade gate.
        grade_biases (torch.nn.Parameter): Biases for each grade gate.
    """

    def __init__(self, algebra, channels: int = 1):
        """Initialize Grade Swish.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of channels.
        """
        super().__init__()
        self.algebra = algebra
        self.n_grades = algebra.n + 1
        
        self.grade_weights = nn.Parameter(torch.ones(self.n_grades))
        self.grade_biases = nn.Parameter(torch.zeros(self.n_grades))
        
        self.register_buffer('grade_masks', self._build_masks())

    def _build_masks(self) -> torch.Tensor:
        """Precompute grade masks.

        Returns:
            torch.Tensor: Boolean masks for each grade [n_grades, dim].
        """
        masks = torch.zeros(self.n_grades, self.algebra.dim, dtype=torch.bool)
        for i in range(self.algebra.dim):
            grade = bin(i).count('1')
            masks[grade, i] = True
        return masks

    def _build_grade_map(self) -> torch.Tensor:
        """Precompute per-component grade index for vectorized forward.

        Returns:
            torch.Tensor: Long tensor of grade indices [dim].
        """
        grade_map = torch.zeros(self.algebra.dim, dtype=torch.long)
        for i in range(self.algebra.dim):
            grade_map[i] = bin(i).count('1')
        return grade_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-grade gating.

        Args:
            x (torch.Tensor): Input multivector [..., Dim].

        Returns:
            torch.Tensor: Activated multivector.
        """
        # Build grade map buffer on first call or device change
        if not hasattr(self, '_grade_map') or self._grade_map is None:
            self.register_buffer('_grade_map', self._build_grade_map())
        grade_map = self._grade_map
        if grade_map.device != x.device:
            grade_map = grade_map.to(x.device)
            self._grade_map = grade_map

        # Compute per-grade norms via scatter
        # x: [..., D], grade_map: [D] -> group components by grade
        D = self.algebra.dim
        G = self.n_grades

        # Square, scatter-add by grade, sqrt -> per-grade norms
        x_sq = x * x  # [..., D]
        # Expand grade_map to match x shape for scatter
        batch_shape = x.shape[:-1]
        grade_idx = grade_map.expand(*batch_shape, D)  # [..., D]

        norm_sq = torch.zeros(*batch_shape, G, device=x.device, dtype=x.dtype)
        norm_sq.scatter_add_(-1, grade_idx, x_sq)  # [..., G]
        norms = torch.sqrt(norm_sq.clamp(min=1e-12))  # [..., G]

        # Compute gates: sigmoid(w * norm + b) for each grade
        gates = torch.sigmoid(
            self.grade_weights * norms + self.grade_biases
        )  # [..., G]

        # Broadcast gate per component: lookup gate[grade_map[d]] for each d
        per_component_gate = gates.gather(-1, grade_idx)  # [..., D]

        return x * per_component_gate