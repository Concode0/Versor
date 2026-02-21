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
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer

class RotorTCN(nn.Module):
    """Temporal convolutional network with Clifford algebra features.

    Applies rotor transformations per frame and 1D convolution along time.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, hidden_channels: int, kernel_size: int = 3, dilation: int = 1):
        """Initialize the Rotor TCN.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input features.
            hidden_channels (int): Hidden features.
            kernel_size (int): Conv kernel.
            dilation (int): Dilation factor.
        """
        super().__init__()
        self.algebra = algebra
        
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
        
        # Project output back to GA structure.
        self.out_rotor = RotorLayer(algebra, hidden_channels)

    def forward(self, x: torch.Tensor):
        """Apply rotor per frame, then convolve.

        Args:
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
        y_tcn = y_tcn.transpose(1, 2) # [B, T, Hidden*D]
        
        # We need to reshape carefully.
        h_c = self.hidden_dim // d
        y_out = y_tcn.view(b, t, h_c, d)
        
        # 3. Output Rotor
        y_flat = y_out.view(b * t, h_c, d)
        res = self.out_rotor(y_flat)
        
        return res.view(b, t, h_c, d)