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
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.projection import BladeSelector
from functional.activation import split_relu

class GeometricBladeNetwork(nn.Module):
    def __init__(self, algebra: CliffordAlgebra, in_channels: int, hidden_channels: int, out_channels: int, layers: int = 2):
        super().__init__()
        self.algebra = algebra
        
        self.net = nn.Sequential()
        
        # Input Layer
        self.net.add_module("input_linear", CliffordLinear(algebra, in_channels, hidden_channels))
        self.net.add_module("input_rotor", RotorLayer(algebra, hidden_channels))
        
        # Hidden Layers
        for i in range(layers):
            self.net.add_module(f"layer_{i}_linear", CliffordLinear(algebra, hidden_channels, hidden_channels))
            # Activation (Split ReLU for now)
            # Note: nn.Sequential doesn't easily support custom functionals without a wrapper.
            self.net.add_module(f"layer_{i}_act", SplitReLU()) 
            self.net.add_module(f"layer_{i}_rotor", RotorLayer(algebra, hidden_channels))
            
        # Output Layer
        self.net.add_module("output_linear", CliffordLinear(algebra, hidden_channels, out_channels))
        self.net.add_module("output_selector", BladeSelector(algebra, out_channels))

    def forward(self, x):
        return self.net(x)

class SplitReLU(nn.Module):
    def forward(self, x):
        return split_relu(x)
