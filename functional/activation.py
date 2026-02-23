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

class GeometricGELU(nn.Module):
    """Geometric GELU activation: x' = x * GELU(||x|| + b) / ||x||.

    Scales magnitude while preserving direction.
    """

    def __init__(self, algebra, channels: int = 1):
        """Initialize the activation with learnable bias."""
        super().__init__()
        self.algebra = algebra
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        """Apply geometric GELU activation."""
        norm = x.norm(dim=-1, keepdim=True)
        
        eps = 1e-6
        scale = F.gelu(norm + self.bias.view(1, -1, 1)) / (norm + eps)
        
        return x * scale

class GradeSwish(nn.Module):
    """Per-grade gated activation.

    Each grade receives an independent sigmoid gate.
    """

    def __init__(self, algebra, channels: int = 1):
        """Initialize per-grade gate parameters."""
        super().__init__()
        self.algebra = algebra
        self.n_grades = algebra.n + 1
        
        self.grade_weights = nn.Parameter(torch.ones(self.n_grades))
        self.grade_biases = nn.Parameter(torch.zeros(self.n_grades))
        
        self.register_buffer('grade_masks', self._build_masks())

    def _build_masks(self):
        """Precomputes masks."""
        masks = torch.zeros(self.n_grades, self.algebra.dim, dtype=torch.bool)
        for i in range(self.algebra.dim):
            grade = bin(i).count('1')
            masks[grade, i] = True
        return masks

    def _build_grade_map(self):
        """Precompute per-component grade index for vectorized forward."""
        grade_map = torch.zeros(self.algebra.dim, dtype=torch.long)
        for i in range(self.algebra.dim):
            grade_map[i] = bin(i).count('1')
        return grade_map

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def forward(self, x: torch.Tensor):
        """Apply per-grade gating."""
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