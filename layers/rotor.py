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

class RotorLayer(CliffordModule):
    """Learnable Rotor. The heart of the GBN.

    Learns R = exp(-B/2). Rotates without distorting.
    Preserves origin, lengths, and angles. Pure isometry.

    Attributes:
        channels (int): Number of rotors.
        bivector_weights (nn.Parameter): Learnable B coefficients.
        use_decomposition (bool): If True, use power iteration decomposition.
        decomp_k (int, optional): Number of simple components for decomposition.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        use_decomposition: bool = False,
        decomp_k: int = None
    ):
        """Sets up the Rotor Layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            channels (int): Number of features.
            use_decomposition (bool): If True, use bivector decomposition.
                Reference: Pence et al. (2025), arXiv:2507.11688v1
            decomp_k (int, optional): Number of simple components for decomposition.
        """
        super().__init__(algebra)
        self.channels = channels
        self.use_decomposition = use_decomposition
        self.decomp_k = decomp_k
        
        self.bivector_indices = self._get_bivector_indices()
        self.num_bivectors = len(self.bivector_indices)
        
        self.bivector_weights = nn.Parameter(torch.Tensor(channels, self.num_bivectors))
        
        self.reset_parameters()
        
    def _get_bivector_indices(self) -> torch.Tensor:
        """Finds the bivectors (Grade 2)."""
        indices = []
        for i in range(self.algebra.dim):
            # Count set bits
            cnt = 0
            temp = i
            while temp > 0:
                if temp & 1: cnt += 1
                temp >>= 1
            if cnt == 2:
                indices.append(i)
        return torch.tensor(indices, device=self.algebra.device, dtype=torch.long)

    def reset_parameters(self):
        """Starts with near-identity rotations."""
        nn.init.normal_(self.bivector_weights, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spins the multivector. Sandwich product style.

        Args:
            x (torch.Tensor): Input [Batch, Channels, Dim].

        Returns:
            torch.Tensor: Rotated input.
        """
        # 1. Construct Bivector B
        B = torch.zeros(self.channels, self.algebra.dim, device=x.device, dtype=x.dtype)

        indices = self.bivector_indices.unsqueeze(0).expand(self.channels, -1)
        B.scatter_(1, indices, self.bivector_weights)

        # 2. Compute Rotor R = exp(-B/2)
        if self.use_decomposition:
            R = self.algebra.exp_decomposed(
                -0.5 * B, use_decomposition=True, k=self.decomp_k
            )
        else:
            R = self.algebra.exp(-0.5 * B)
        
        # 3. Reverse R
        R_rev = self.algebra.reverse(R)
        
        # 4. Sandwich: R * x * ~R
        R_expanded = R.unsqueeze(0) # [1, C, D]
        R_rev_expanded = R_rev.unsqueeze(0) # [1, C, D]
        
        # Rx
        Rx = self.algebra.geometric_product(R_expanded, x)
        
        # (Rx)~R
        res = self.algebra.geometric_product(Rx, R_rev_expanded)
        
        return res

    def prune_bivectors(self, threshold: float = 1e-4) -> int:
        """Trims the fat. Removes useless rotation planes.

        Args:
            threshold (float): Cutoff magnitude.

        Returns:
            int: Number of pruned parameters.
        """
        with torch.no_grad():
            mask = torch.abs(self.bivector_weights) >= threshold
            num_pruned = (~mask).sum().item()
            self.bivector_weights.data.mul_(mask.float())
        return num_pruned

    def sparsity_loss(self) -> torch.Tensor:
        """Penalizes complexity. Keep it simple.

        L1 norm on bivectors.
        """
        return torch.norm(self.bivector_weights, p=1)