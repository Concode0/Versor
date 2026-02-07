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
from core.algebra import CliffordAlgebra
from layers.base import CliffordModule

class CliffordLayerNorm(CliffordModule):
    """Geometric LayerNorm. Provides stability during training.

    Normalizes energy (magnitude) to 1, but respects the geometric direction.
    Learns a scalar scale and bias to allow for learnable affine transformations.

    Attributes:
        weight (nn.Parameter): Scale.
        bias (nn.Parameter): Bias.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, eps: float = 1e-6):
        """Sets up normalization.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Features.
            eps (float): Stability term.
        """
        super().__init__(algebra)
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes energy. Preserves direction.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Normalized input.
        """
        # Calculate magnitude per channel
        norm = x.norm(dim=-1, keepdim=True)
        
        # Normalize direction
        x_normalized = x / (norm + self.eps)
        
        # Affine transform
        out = x_normalized * self.weight.view(1, -1, 1)
        
        # Bias added to scalar part only
        bias_tensor = torch.zeros_like(out)
        bias_tensor[..., 0] = self.bias.view(1, -1)
        
        return out + bias_tensor