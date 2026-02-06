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
from layers.multi_rotor import MultiRotorLayer
from functional.activation import GeometricGELU

class MultiRotorModel(nn.Module):
    """Geometric FFT Engine. Breaking the curse.

    Uses overlapping rotors to approximate complex manifolds without
    exponential parameter explosion.

    Attributes:
        algebra (CliffordAlgebra): The algebra.
        net (nn.Sequential): The geometric backbone.
        readout (nn.Sequential): Invariant MLP.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, num_rotors: int = 8):
        """Sets up the Multi-Rotor model.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input features.
            hidden_channels (int): Hidden features.
            out_channels (int): Output features.
            num_layers (int): Depth.
            num_rotors (int): Width (Rotors).
        """
        super().__init__()
        self.algebra = algebra
        
        self.input_linear = CliffordLinear(algebra, in_channels, hidden_channels)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MultiRotorLayer(algebra, hidden_channels, num_rotors))
            self.layers.append(GeometricGELU(algebra, hidden_channels))

        # Final Multi-Rotor layer to extract Pure Invariants
        self.invariant_head = MultiRotorLayer(algebra, hidden_channels, num_rotors)
        
        # Reading out from Invariants (Dimensionless Structure)
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels * algebra.num_grades, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project, Transform, Invariants, Readout.

        Args:
            x (torch.Tensor): Input [Batch, In_Channels, Dim].

        Returns:
            torch.Tensor: Predictions [Batch, Out_Channels].
        """
        # 1. Project
        h = self.input_linear(x)
        
        # 2. Geometric FFT
        for layer in self.layers:
            h = layer(h)
            
        # 3. Extract Pure Invariants
        invariants = self.invariant_head(h, return_invariants=True) # [B, C, Num_Grades]
        
        # 4. Dimensionless Readout
        inv_flat = invariants.view(invariants.size(0), -1)
        
        return self.readout(inv_flat)