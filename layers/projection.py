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
    """Blade Selector. Filters insignificant components.

    Learns to weigh geometric grades, suppressing less relevant ones.

    Attributes:
        weights (nn.Parameter): Soft gates [Channels, Dim].
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """Sets up the selector.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Input features.
        """
        super().__init__(algebra)
        
        self.weights = nn.Parameter(torch.Tensor(channels, algebra.dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights to one (pass-through)."""
        nn.init.ones_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gates the grades.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Filtered input.
        """
        # Sigmoid gate
        w = torch.sigmoid(self.weights).unsqueeze(0)
        return x * w