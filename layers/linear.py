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

class CliffordLinear(CliffordModule):
    """A fully connected layer for Multivectors.

    Learns a linear transformation W where W is a matrix of multivectors.
    Unlike standard linear layers, this respects the algebraic structure if
    designed with geometric constraints (here, it's a general linear map on coefficients).

    Attributes:
        in_channels (int): Number of input multivector channels.
        out_channels (int): Number of output multivector channels.
        weight (torch.Tensor): Learnable weights [Out, In, Dim, Dim] (Full Operator) 
                               or simplified [Out, In] (Scalar scaling per component).
                               Currently implements Component-wise Linear map.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, out_channels: int):
        """Initializes the Clifford Linear layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input feature size (number of multivectors).
            out_channels (int): Output feature size.
        """
        super().__init__(algebra)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # A full linear map on the algebra would be [Out, In, Dim, Dim].
        # For efficiency and standard GNN practices, we often use scalar weights 
        # mixing channels but preserving the multivector components structure 
        # (i.e. x_out_i = sum_j w_ij * x_in_j).
        # This is equivariant to basis changes if weights are scalars.
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels, algebra.dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using Xavier uniform and bias to zero."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the linear transformation.

        Args:
            x (torch.Tensor): Input tensor [Batch, In_Channels, Dim].

        Returns:
            torch.Tensor: Output tensor [Batch, Out_Channels, Dim].
        """
        # x: [Batch, In, Dim]
        # weight: [Out, In]
        # We want: [Batch, Out, Dim]
        
        # Einsum: b=batch, i=in_channels, o=out_channels, d=dim
        # w: oi
        # x: bid
        # out: bod -> sum_i (w_oi * x_bid)
        
        out = torch.einsum('oi,bid->bod', self.weight, x)
        out = out + self.bias.unsqueeze(0)
        return out