# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Automated metric search. Because guessing (p, q) is for chumps.

Algorithms to discover the geometric signature that doesn't break your topology.
"""

import torch
import itertools
from typing import Tuple, List, Optional
from core.algebra import CliffordAlgebra
from core.metric import induced_norm

class MetricSearch:
    """Finds the optimal signature (p, q).

    It calculates distortion. If the distortion is low, the manifold is happy.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device

    def _compute_pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Standard Euclidean distances. The baseline."""
        # x: [N, D]
        # dist: [N, N]
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        return torch.norm(diff, dim=-1)

    def evaluate_signature(self, data: torch.Tensor, p: int, q: int) -> float:
        """Measures how much the algebra hates your data.

        Stress = || D_euc - D_geo(p,q) ||_F / || D_euc ||_F.

        Args:
            data (torch.Tensor): Input data [Batch, Dim].
            p (int): Positive dimensions.
            q (int): Negative dimensions.

        Returns:
            float: Distortion score. Lower is better.
        """
        N, D = data.shape
        if p + q != D:
            raise ValueError(f"Signature (p={p}, q={q}) must sum to data dimension {D}")

        # 1. Compute target pairwise distances (Euclidean/Intrinsic reference)
        algebra = CliffordAlgebra(p, q, device=self.device)
        
        # Embed data as vectors
        # Calculate pairwise geometric distances
        diff_coeffs = data.unsqueeze(1) - data.unsqueeze(0) # [N, N, D]
        
        # Map coefficients to multivector basis indices
        diff_flat = diff_coeffs.view(-1, D)
        
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
        """Finds the best signature. Brute force but effective.

        Args:
            data (torch.Tensor): Input data [N, D].
            limit_combinations (int): Ignored. We do all of them.

        Returns:
            Tuple[int, int]: The optimal (p, q).
        """
        N, D = data.shape
        best_pq = (D, 0) # Default to Euclidean
        best_score = float('inf')
        
        # Iterate over all p+q = D
        candidates = []
        for q in range(D + 1):
            p = D - q
            candidates.append((p, q))
            
        results = []
        for p, q in candidates:
            score = self.evaluate_signature(data, p, q)
            results.append(((p, q), score))
            
            if score < best_score:
                best_score = score
                best_pq = (p, q)
                
        return best_pq