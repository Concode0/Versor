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
from layers.linear import CliffordLinear

class CliffordGraphConv(CliffordModule):
    """Geometric Graph Conv. Performs message passing using multivector features.

    Aggregates features based on graph topology.
    H' = Aggregate(H) * W + Bias.

    Attributes:
        linear (CliffordLinear): The transformation.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, out_channels: int):
        """Sets up the GNN layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input features.
            out_channels (int): Output features.
        """
        super().__init__(algebra)
        self.linear = CliffordLinear(algebra, in_channels, out_channels)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Aggregates and transforms node features using geometric operations.

        Args:
            x (torch.Tensor): Node features.
            adj (torch.Tensor): Adjacency matrix.

        Returns:
            torch.Tensor: Updated features.
        """
        # 1. Aggregate
        N, C, D = x.shape
        x_flat = x.view(N, -1)
        
        # Sparse aggregation
        x_agg_flat = torch.mm(adj, x_flat)
        x_agg = x_agg_flat.view(N, C, D)
        
        # 2. Transform
        out = self.linear(x_agg)
        
        return out