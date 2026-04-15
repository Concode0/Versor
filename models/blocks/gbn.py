# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.primitives.base import CliffordModule
from layers import CliffordLinear
from layers import RotorLayer
from layers import BladeSelector
from functional.activation import GeometricGELU

class GeometricBladeNetwork(CliffordModule):
    """Geometric Blade Network (GBN) reference implementation.

    Stacks CliffordLinear and RotorLayer for geometric representation learning.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, hidden_channels: int, out_channels: int, layers: int = 2):
        super().__init__(algebra)

        self.net = nn.Sequential()

        # Input Layer
        self.net.add_module("input_linear", CliffordLinear(algebra, in_channels, hidden_channels))
        self.net.add_module("input_rotor", RotorLayer(algebra, hidden_channels))

        # Hidden Layers
        for i in range(layers):
            self.net.add_module(f"layer_{i}_linear", CliffordLinear(algebra, hidden_channels, hidden_channels))
            self.net.add_module(f"layer_{i}_act", GeometricGELU(algebra, channels=hidden_channels))
            self.net.add_module(f"layer_{i}_rotor", RotorLayer(algebra, hidden_channels))
            
        # Output Layer
        self.net.add_module("output_linear", CliffordLinear(algebra, hidden_channels, out_channels))
        self.net.add_module("output_selector", BladeSelector(algebra, out_channels))

    def forward(self, x):
        return self.net(x)