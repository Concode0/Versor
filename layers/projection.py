# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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

class BladeSelector(CliffordModule):
    """A soft projection/filtering layer for basis blades.

    Learns a scalar weight for each basis element (scalar, vectors, bivectors...)
    to emphasize or suppress specific geometric grades or directions.

    Attributes:
        weights (nn.Parameter): Learnable weights [Channels, Dim].
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """Initializes the Blade Selector.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of independent channels.
        """
        super().__init__(algebra)
        
        # Learn a weight for each basis element per channel
        # Weights: [Channels, Dim]
        self.weights = nn.Parameter(torch.Tensor(channels, algebra.dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights to 1 (pass-through)."""
        nn.init.ones_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies soft blade selection via element-wise multiplication.

        Args:
            x (torch.Tensor): Input multivectors [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Filtered multivectors [Batch, Channels, Dim].
        """
        # Apply Sigmoid to weights to act as a gate (0 to 1)
        # Or standard multiplicative weight?
        # Let's use sigmoid to enforce "selection" semantics.
        w = torch.sigmoid(self.weights).unsqueeze(0)
        return x * w
