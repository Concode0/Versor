# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

from typing import Iterable, Optional

import torch
import torch.nn as nn

from core.foundation.module import CliffordModule

from ..grade import check_multivector_lanes, lane_count, resolve_layer_layout


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

    def __init__(
        self,
        algebra,
        channels: int,
        eps: float = 1e-6,
        recover: bool = True,
        grades: Optional[Iterable[int]] = None,
    ):
        """Sets up normalization.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Features.
            eps (float): Stability term.
            recover (bool): Whether to inject original scale into the scalar part.
            grades: Optional layer-owned active grades for compact lane execution.
        """
        super().__init__(algebra)
        self.eps = eps
        self.recover = recover
        self.layout = resolve_layer_layout(algebra, grades)
        self.basis_dim = lane_count(algebra, self.layout)

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("_scalar_lane_mask", self._build_scalar_lane_mask())
        # Learnable gate: how much of the original log-magnitude to push
        # into the scalar part.  Zero-init -> backward compatible at start.
        if recover:
            self.norm_scale = nn.Parameter(torch.zeros(channels))
        else:
            self.register_buffer("norm_scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes energy, preserves direction, optionally recovers scale in grade-0.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Normalized input.
        """
        check_multivector_lanes(x, self.algebra, self.layout, "CliffordLayerNorm input")

        # Per-channel magnitude
        norm = x.norm(dim=-1, keepdim=True)  # [B, C, 1]

        # Normalize direction
        x_normalized = x / (norm + self.eps)

        # Affine transform on direction
        out = x_normalized * self.weight.view(1, -1, 1)

        # Add bias and optional log-magnitude to the declared scalar lane.
        g0 = self._scalar_lane_mask
        if g0.dtype != x.dtype:
            g0 = g0.to(dtype=x.dtype)
        out = out + self.bias.view(1, -1, 1) * g0

        if self.recover:
            # Push original magnitude into scalar (grade-0) part.
            # log1p keeps the value bounded and well-behaved for gradients.
            log_norm = torch.log1p(norm.squeeze(-1)).unsqueeze(-1)  # [B, C, 1]
            out = out + self.norm_scale.view(1, -1, 1) * log_norm * g0

        return out

    def _build_scalar_lane_mask(self) -> torch.Tensor:
        """Return a lane mask with 1 at scalar basis position when present."""
        mask = torch.zeros(self.basis_dim, dtype=self.algebra.dtype, device=self.algebra.device)
        if self.layout is None:
            mask[0] = 1.0
            return mask
        try:
            mask[self.layout.basis_indices.index(0)] = 1.0
        except ValueError:
            pass
        return mask
