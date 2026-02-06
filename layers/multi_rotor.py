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

class MultiRotorLayer(CliffordModule):
    """Geometric FFT. The Multi-Rotor Engine.

    Spectral decomposition. Because one rotor isn't enough.
    Replaces rigid rotations with a flexible superposition.

    Attributes:
        channels (int): Input features.
        num_rotors (int): Number of overlapping rotors.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, num_rotors: int = 8):
        """Sets up the Multi-Rotor engine.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Input features.
            num_rotors (int): Parallel heads.
        """
        super().__init__(algebra)
        self.channels = channels
        self.num_rotors = num_rotors
        
        self.bivector_indices = self._get_bivector_indices()
        self.num_bivectors = len(self.bivector_indices)
        
        # Overlapping rotors
        self.rotor_bivectors = nn.Parameter(torch.Tensor(num_rotors, self.num_bivectors))
        
        # Mixing weights
        self.weights = nn.Parameter(torch.Tensor(channels, num_rotors))
        
        self.reset_parameters()
        
    def _get_bivector_indices(self) -> torch.Tensor:
        """Finds the bivectors."""
        indices = []
        for i in range(self.algebra.dim):
            cnt = bin(i).count('1')
            if cnt == 2:
                indices.append(i)
        return torch.tensor(indices, device=self.algebra.device, dtype=torch.long)

    def reset_parameters(self):
        """Random init. Small rotations, uniform weights."""
        nn.init.normal_(self.rotor_bivectors, std=0.01)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor, return_invariants: bool = False) -> torch.Tensor:
        """Unbends the manifold.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].
            return_invariants (bool): If True, returns grade norms.

        Returns:
            torch.Tensor: Transformed output.
        """
        # 1. Generate Rotors
        B = torch.zeros(self.num_rotors, self.algebra.dim, device=x.device, dtype=x.dtype)
        indices = self.bivector_indices.unsqueeze(0).expand(self.num_rotors, -1)
        B.scatter_(1, indices, self.rotor_bivectors)
        
        R = self.algebra.exp(-0.5 * B) # [K, D]
        R_rev = self.algebra.reverse(R)
        
        # 2. Sandwich Product
        x_expanded = x.unsqueeze(2)
        R_expanded = R.view(1, 1, self.num_rotors, -1)
        R_rev_expanded = R_rev.view(1, 1, self.num_rotors, -1)
        
        Rx = self.algebra.geometric_product(R_expanded, x_expanded)
        rotated_x = self.algebra.geometric_product(Rx, R_rev_expanded)
        
        # 3. Superposition
        out = torch.einsum('ck,bckd->bcd', self.weights, rotated_x)
        
        if return_invariants:
            return self.algebra.get_grade_norms(out)
            
        return out

    def sparsity_loss(self) -> torch.Tensor:
        """Sparsity loss. Don't learn what you don't need."""
        return torch.norm(self.rotor_bivectors, p=1) + torch.norm(self.weights, p=1)