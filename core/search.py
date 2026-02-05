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

"""Automatic Metric Search Utilities.

Provides algorithms to discover the optimal geometric signature (p, q) 
that best preserves the topological properties of the input data.
"""

import torch
import itertools
from typing import Tuple, List, Optional
from core.algebra import CliffordAlgebra
from core.metric import induced_norm

class MetricSearch:
    """Searches for the optimal metric signature (p, q).

    Analyzes the distortion induced by embedding the dataset into different
    Clifford Algebras Cl(p, q).
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def _compute_pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Computes pairwise Euclidean distances."""
        # x: [N, D]
        # dist: [N, N]
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        return torch.norm(diff, dim=-1)

    def evaluate_signature(self, data: torch.Tensor, p: int, q: int) -> float:
        """Computes the stress/distortion of the data under signature (p, q).

        Stress = || D_euc - D_geo(p,q) ||_F / || D_euc ||_F

        Args:
            data (torch.Tensor): Input data [Batch, Dim].
            p (int): Positive dimensions.
            q (int): Negative dimensions.

        Returns:
            float: Distortion score (lower is better).
        """
        N, D = data.shape
        if p + q != D:
            raise ValueError(f"Signature (p={p}, q={q}) must sum to data dimension {D}")

        # 1. Compute target pairwise distances (Euclidean/Intrinsic reference)
        algebra = CliffordAlgebra(p, q, device=self.device)
        
        # Embed data as vectors
        # vectors = sum x_i e_i
        # We can simulate this by keeping x as coefficients.
        
        # Calculate pairwise geometric distances
        # diff[i, j] = data[i] - data[j] (vector subtraction)
        diff_coeffs = data.unsqueeze(1) - data.unsqueeze(0) # [N, N, D]
        
        # To use `induced_norm`, we need to construct multivectors.
        # diff_coeffs is [N, N, D]. We flatten to [N*N, D]
        diff_flat = diff_coeffs.view(-1, D)
        
        # Map coefficients to multivector basis indices
        # We assume data columns map to basis vectors e_1, ..., e_D
        # In `CliffordAlgebra`, basis indices for vectors are 1, 2, 4, 8...
        # We need to scatter these coefficients into a multivector tensor [N*N, 2^D]
        
        mv = torch.zeros(diff_flat.shape[0], algebra.dim, device=self.device)
        for i in range(D):
            mv[:, 1 << i] = diff_flat[:, i]
            
        # Compute induced norms
        dists_geo = induced_norm(algebra, mv) # [N*N, 1]
        dists_geo = dists_geo.view(N, N)
        
        # Reference: Euclidean Distance
        dists_euc = self._compute_pairwise_distances(data)
        
        # Stress
        stress = torch.norm(dists_euc - dists_geo) / (torch.norm(dists_euc) + 1e-8)
        return stress.item()

    def search(self, data: torch.Tensor, limit_combinations: int = 10) -> Tuple[int, int]:
        """Finds the best (p, q) signature for the given data.

        Args:
            data (torch.Tensor): Input data [N, D].
            limit_combinations (int): Max number of signatures to try.

        Returns:
            Tuple[int, int]: The optimal (p, q).
        """
        N, D = data.shape
        best_pq = (D, 0) # Default to Euclidean
        best_score = float('inf')
        
        # Iterate over all p+q = D
        # q goes from 0 to D
        candidates = []
        for q in range(D + 1):
            p = D - q
            candidates.append((p, q))
            
        # If D is large, we might want to limit, but usually D < 10 for GA tasks
        
        results = []
        for p, q in candidates:
            score = self.evaluate_signature(data, p, q)
            results.append(((p, q), score))
            
            if score < best_score:
                best_score = score
                best_pq = (p, q)
                
        return best_pq
