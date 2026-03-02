# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Conformal Geometric Algebra (CGA) utilities.

Helpers to map between Euclidean space and the conformal null cone.
"""

import torch
from core.algebra import CliffordAlgebra


class ConformalAlgebra:
    """Helper for CGA. Handles Conformal Geometric Algebra operations.

    Maps Euclidean R^n to Null Cone in R^{n+1, 1}.

    Attributes:
        d (int): Euclidean dimension.
        algebra (CliffordAlgebra): The higher dimensional algebra.
        e_o (torch.Tensor): Origin basis vector (null).
        e_inf (torch.Tensor): Infinity basis vector (null).
        device (str): Computation device.
    """

    def __init__(self, euclidean_dim: int = 3, device='cpu'):
        """Sets up the CGA stage.

        Args:
            euclidean_dim (int): Physical dimension d.
            device (str): Computation device.
        """
        self.d = euclidean_dim
        # Standard CGA signature: p=d+1 (e1..ed, e+), q=1 (e-)
        self.algebra = CliffordAlgebra(p=euclidean_dim + 1, q=1, device=device)
        self.device = device
        
        # Basis indices
        # e_i corresponds to bit i-1. e+ is bit d, e- is bit d+1.
        self.idx_ep = 1 << euclidean_dim
        self.idx_em = 1 << (euclidean_dim + 1)
        
        # Construct e_o and e_inf
        # e_inf = e_- + e_+
        # e_o = 0.5 * (e_- - e_+)
        self.e_o = torch.zeros(1, self.algebra.dim, device=device)
        self.e_inf = torch.zeros(1, self.algebra.dim, device=device)
        
        self.e_inf[0, self.idx_em] = 1.0
        self.e_inf[0, self.idx_ep] = 1.0
        
        self.e_o[0, self.idx_em] = 0.5
        self.e_o[0, self.idx_ep] = -0.5

    def to_cga(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds Euclidean points into the conformal null cone.

        P(x) = x + 0.5 * x^2 * e_inf + e_o.

        Args:
            x (torch.Tensor): Euclidean points [Batch, d].

        Returns:
            torch.Tensor: Conformal points [Batch, 2^n].
        """
        batch_size = x.shape[0]
        
        # 1. Embed vector part x
        x_mv = torch.zeros(batch_size, self.algebra.dim, device=x.device, dtype=x.dtype)
        for i in range(self.d):
            basis_idx = 1 << i
            x_mv[:, basis_idx] = x[:, i]

        # 2. Compute x^2 (squared Euclidean norm)
        x_sq = (x ** 2).sum(dim=1, keepdim=True)

        # 3. Assemble P
        e_inf = self.e_inf.to(device=x.device, dtype=x.dtype)
        e_o = self.e_o.to(device=x.device, dtype=x.dtype)
        term2 = 0.5 * x_sq * e_inf
        term3 = e_o.expand(batch_size, -1)
        
        P = x_mv + term2 + term3
        return P

    def from_cga(self, P: torch.Tensor) -> torch.Tensor:
        """Projects conformal points back to Euclidean space.

        Normalization: P -> P / (-P . e_inf).

        Args:
            P (torch.Tensor): Conformal points [Batch, 2^n].

        Returns:
            torch.Tensor: Euclidean coordinates [Batch, d].
        """
        # 1. Normalize P such that P inner e_inf = -1
        e_inf = self.e_inf.to(device=P.device, dtype=P.dtype)
        P_einf = self.algebra.geometric_product(P, e_inf)
        scale = -P_einf[..., 0:1] # Scalar part

        P_norm = P / (scale + 1e-6)

        # 2. Extract vector components (indices 0 to d-1)
        x = torch.zeros(P.shape[0], self.d, device=P.device, dtype=P.dtype)
        for i in range(self.d):
            basis_idx = 1 << i
            x[:, i] = P_norm[:, basis_idx]
            
        return x