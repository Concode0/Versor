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

class CliffordLinear(CliffordModule):
    """Fully connected layer. Mixing channels, preserving geometry.

    Learns a linear transformation. We use scalar weights to mix channels
    because full multivector weights would explode your VRAM.

    Attributes:
        in_channels (int): Input features.
        out_channels (int): Output features.
        weight (torch.Tensor): Weights [Out, In].
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, out_channels: int):
        """Sets up the linear layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input size.
            out_channels (int): Output size.
        """
        super().__init__(algebra)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels, algebra.dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Random initialization. Standard stuff."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixes the channels. Standard linear map.

        Args:
            x (torch.Tensor): Input [Batch, In, Dim].

        Returns:
            torch.Tensor: Output [Batch, Out, Dim].
        """
        # x: [Batch, In, Dim]
        # weight: [Out, In]
        # out: [Batch, Out, Dim]
        
        out = torch.einsum('oi,bid->bod', self.weight, x)
        out = out + self.bias.unsqueeze(0)
        return out