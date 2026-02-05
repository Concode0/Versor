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
from layers.linear import CliffordLinear

class CliffordGraphConv(CliffordModule):
    """Graph Convolutional Layer for Multivector signals.

    Aggregates geometric features from neighbors and applies a linear transformation.
    H' = Aggregate(H) * W + Bias

    Attributes:
        linear (CliffordLinear): The learnable linear transformation.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, out_channels: int):
        """Initializes the Graph Convolution layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input node feature size.
            out_channels (int): Output node feature size.
        """
        super().__init__(algebra)
        self.linear = CliffordLinear(algebra, in_channels, out_channels)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Performs graph convolution.

        Args:
            x (torch.Tensor): Node features [Num_Nodes, In_Channels, Dim].
            adj (torch.Tensor): Adjacency matrix [Num_Nodes, Num_Nodes].

        Returns:
            torch.Tensor: Updated node features [Num_Nodes, Out_Channels, Dim].
        """
        # 1. Aggregate Neighbor Information
        # Flatten channels for matrix multiplication
        N, C, D = x.shape
        x_flat = x.view(N, -1)
        
        # Sparse aggregation via dense matmul (simplified)
        x_agg_flat = torch.mm(adj, x_flat)
        x_agg = x_agg_flat.view(N, C, D)
        
        # 2. Geometric Transformation
        out = self.linear(x_agg)
        
        return out
