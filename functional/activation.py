# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric GA activations.

Magnitude-scaling and grade-wise gating functions that preserve geometric structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.primitives.base import CliffordModule


class GeometricGELU(CliffordModule):
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
        super().__init__(algebra)
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


class GeometricSquare(CliffordModule):
    """Gated geometric self-product: x + gate * GP(x, x).

    GP(grade-1, grade-1) produces grade-0 (squares x_i^2) and grade-2
    (wedge products x_i ^ x_j).  Creates algebraic cross-terms that
    rotors can then rotate into the output.
    """

    def __init__(self, algebra, channels: int = 1):
        super().__init__(algebra)
        # sigmoid(-2) ~= 0.12 -- starts small so GP doesn't dominate
        self.gate_logit = nn.Parameter(torch.full((channels,), -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gp = self.algebra.geometric_product(x, x)  # [B, C, dim]
        gate = torch.sigmoid(self.gate_logit).view(1, -1, 1)  # [1, C, 1]
        return x + gate * gp


class GradeSwish(CliffordModule):
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
        super().__init__(algebra)
        self.n_grades = self.algebra.n + 1

        self.grade_weights = nn.Parameter(torch.ones(self.n_grades))
        self.grade_biases = nn.Parameter(torch.zeros(self.n_grades))

        # Reuse algebra's precomputed grade_index instead of building our own
        self.register_buffer('_grade_index', self.algebra.grade_index.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-grade gating.

        Args:
            x (torch.Tensor): Input multivector [..., Dim].

        Returns:
            torch.Tensor: Activated multivector.
        """
        D = self.algebra.dim
        G = self.n_grades

        # Square, scatter-add by grade, sqrt -> per-grade norms
        x_sq = x * x  # [..., D]
        batch_shape = x.shape[:-1]
        grade_idx = self._grade_index.expand(*batch_shape, D)  # [..., D]

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