# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from .base import CliffordModule

class CliffordLayerNorm(CliffordModule):
    """Geometric LayerNorm that preserves direction and recovers scale.

    Normalizes the multivector to unit norm (preserving geometric direction),
    then injects the original log-magnitude into the scalar (grade-0) part
    via a learnable gate.

    Attributes:
        weight (nn.Parameter): Per-channel direction scale [C].
        bias (nn.Parameter): Per-channel scalar bias [C].
        norm_scale (nn.Parameter): Per-channel gate for log-magnitude
            injection into grade-0.  Initialized to zero so the layer
            starts identical to the old (scale-discarding) behaviour.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, eps: float = 1e-6, recover: bool = True):
        """Sets up normalization.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Features.
            eps (float): Stability term.
            recover (bool): Whether to inject original scale into the scalar part.
        """
        super().__init__(algebra)
        self.eps = eps
        self.recover = recover

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        # Learnable gate: how much of the original log-magnitude to push
        # into the scalar part.  Zero-init -> backward compatible at start.
        if recover:
            self.norm_scale = nn.Parameter(torch.zeros(channels))
        else:
            self.register_buffer('norm_scale', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes energy, preserves direction, optionally recovers scale in grade-0.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Normalized input.
        """
        # Per-channel magnitude
        norm = x.norm(dim=-1, keepdim=True)  # [B, C, 1]

        # Normalize direction
        x_normalized = x / (norm + self.eps)

        # Affine transform on direction
        out = x_normalized * self.weight.view(1, -1, 1)

        # Add bias and optional log-magnitude to grade-0 via mask
        g0 = self.algebra.grade_masks_float[0]  # [D], 1.0 at index 0
        if g0.dtype != x.dtype:
            g0 = g0.to(dtype=x.dtype)
        out = out + self.bias.view(1, -1, 1) * g0

        if self.recover:
            # Push original magnitude into scalar (grade-0) part.
            # log1p keeps the value bounded and well-behaved for gradients.
            log_norm = torch.log1p(norm.squeeze(-1)).unsqueeze(-1)  # [B, C, 1]
            out = out + self.norm_scale.view(1, -1, 1) * log_norm * g0

        return out