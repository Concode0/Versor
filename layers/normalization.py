# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.base import CliffordModule

class CliffordLayerNorm(CliffordModule):
    """Geometric Layer Normalization.

    Normalizes the magnitude (energy) of each multivector channel to 1,
    then applies a learnable scalar scale and scalar bias. This preserves
    directional information while stabilizing training dynamics.

    Attributes:
        eps (float): Small constant for numerical stability.
        weight (nn.Parameter): Learnable scale factor [Channels].
        bias (nn.Parameter): Learnable scalar bias [Channels].
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, eps: float = 1e-6):
        """Initializes the normalization layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of feature channels.
            eps (float, optional): Epsilon for division. Defaults to 1e-6.
        """
        super().__init__(algebra)
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization.

        Args:
            x (torch.Tensor): Input multivectors [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Normalized multivectors.
        """
        # Calculate magnitude per channel (dim=-1 is the geometric dimension)
        norm = x.norm(dim=-1, keepdim=True)
        
        # Normalize direction
        x_normalized = x / (norm + self.eps)
        
        # Apply affine transform
        # Scale affects the whole multivector uniformly
        out = x_normalized * self.weight.view(1, -1, 1)
        
        # Bias is added only to the Scalar part (index 0) to shift the baseline energy
        bias_tensor = torch.zeros_like(out)
        bias_tensor[..., 0] = self.bias.view(1, -1)
        
        return out + bias_tensor
