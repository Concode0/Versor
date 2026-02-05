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
