# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn

from core.foundation.layout import GradeLayout
from core.foundation.module import CliffordModule
from core.runtime.algebra import CliffordAlgebra

from ._utils import (
    check_multivector_storage,
    lane_dim,
    require_positive_int,
    resolve_layer_layout,
    scalar_mask,
)


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
        algebra: CliffordAlgebra,
        channels: int,
        eps: float = 1e-6,
        recover: bool = True,
        *,
        grades=None,
        layout: GradeLayout = None,
    ):
        """Sets up normalization.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Features.
            eps (float): Stability term.
            recover (bool): Whether to inject original scale into the scalar part.
        """
        super().__init__(algebra)
        self.channels = require_positive_int(channels, "channels")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        self.recover = recover
        self.layout = resolve_layer_layout(algebra, layout=layout, grades=grades)
        self.lane_dim = lane_dim(algebra, self.layout)

        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.register_buffer("scalar_mask", scalar_mask(algebra, self.layout))
        if recover:
            self.norm_scale = nn.Parameter(torch.zeros(self.channels))
        else:
            self.register_buffer("norm_scale", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes energy, preserves direction, optionally recovers scale in grade-0.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Normalized input.
        """
        check_multivector_storage(
            x,
            self.algebra,
            channels=self.channels,
            name="CliffordLayerNorm input",
            layout=self.layout,
            allow_dense=self.layout is None or self.layout.dim == self.algebra.dim,
        )
        channel_shape = (1,) * (x.ndim - 2) + (self.channels, 1)

        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        x_normalized = x / norm
        out = x_normalized * self.weight.view(channel_shape)

        g0 = self.scalar_mask
        if g0.device != x.device or g0.dtype != x.dtype:
            g0 = g0.to(device=x.device, dtype=x.dtype)
        out = out + self.bias.view(channel_shape) * g0

        if self.recover:
            log_norm = torch.log1p(norm)
            out = out + self.norm_scale.view(channel_shape) * log_norm * g0

        return out
