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
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer

class RotorTCN(nn.Module):
    def __init__(self, algebra: CliffordAlgebra, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1):
        """
        Temporal Convolutional Network where the 'conv' is a Clifford Linear operation across time?
        Or simply applying Rotor Layers per timestep with TCN mixing?
        
        Let's implement a simple 1D Conv over time but with Clifford Features.
        Input: [Batch, Time, Channels, Dim]
        """
        super().__init__()
        self.algebra = algebra
        
        # We treat Time as part of the Batch or spatial dim for the Linear layer?
        # CliffordLinear expects [Batch, In_C, Dim].
        # We need to process temporal context.
        
        # Simplified: Frame-wise Rotor Layer + 1D Conv on Coefficients
        self.rotor = RotorLayer(algebra, in_channels)
        
        # Standard Conv1d on the coefficients
        # Input: [B, T, C, D] -> [B, C*D, T]
        self.input_dim = in_channels * algebra.dim
        self.hidden_dim = hidden_channels * algebra.dim
        
        self.tcn = nn.Conv1d(
            self.input_dim, 
            self.hidden_dim, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            padding=(kernel_size-1)*dilation // 2
        )
        
        # Output projection back to GA structure?
        self.out_rotor = RotorLayer(algebra, hidden_channels)

    def forward(self, x: torch.Tensor):
        """
        x: [Batch, Time, Channels, Dim]
        """
        b, t, c, d = x.shape
        
        # 1. Apply Rotor (independent per time step)
        # Flatten time into batch
        x_flat = x.view(b * t, c, d)
        x_rot = self.rotor(x_flat)
        x_rot = x_rot.view(b, t, c, d)
        
        # 2. TCN Mixing
        # Rearrange to [B, C*D, T]
        x_in_tcn = x_rot.view(b, t, c * d).transpose(1, 2)
        
        y_tcn = self.tcn(x_in_tcn)
        
        # Rearrange back to [B, T, H_C, D]
        # Note: T might change if padding/stride varies, but here it's kept same.
        y_tcn = y_tcn.transpose(1, 2) # [B, T, Hidden*D]
        
        # We need to reshape carefully.
        # Assuming Hidden_Dim is divisible by D
        h_c = self.hidden_dim // d
        y_out = y_tcn.view(b, t, h_c, d)
        
        # 3. Output Rotor
        y_flat = y_out.view(b * t, h_c, d)
        res = self.out_rotor(y_flat)
        
        return res.view(b, t, h_c, d)
