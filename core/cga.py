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
from core.algebra import CliffordAlgebra

class ConformalAlgebra:
    """Helper for Conformal Geometric Algebra (CGA).

    Implements embedding and reconstruction of Euclidean points into the Conformal
    model Cl(p+1, 1).

    Attributes:
        d (int): Dimension of the Euclidean space.
        algebra (CliffordAlgebra): Underlying Clifford algebra Cl(d+1, 1).
        e_o (torch.Tensor): Null basis vector representing origin.
        e_inf (torch.Tensor): Null basis vector representing infinity.
    """

    def __init__(self, euclidean_dim: int = 3, device='cpu'):
        """Initializes the CGA helper.

        Args:
            euclidean_dim (int): Euclidean dimension d.
            device (str): Computation device.
        """
        self.d = euclidean_dim
        # Standard CGA signature: p=d+1 (e1..ed, e+), q=1 (e-)
        self.algebra = CliffordAlgebra(p=euclidean_dim + 1, q=1, device=device)
        self.device = device
        
        # Basis indices
        # e_i corresponds to bit i-1 (0-indexed). e+ is bit d, e- is bit d+1.
        self.idx_ep = 1 << euclidean_dim
        self.idx_em = 1 << (euclidean_dim + 1)
        
        # Construct e_o and e_inf constant multivectors
        # e_inf = e_- + e_+
        # e_o = 0.5 * (e_- - e_+)
        self.e_o = torch.zeros(1, self.algebra.dim, device=device)
        self.e_inf = torch.zeros(1, self.algebra.dim, device=device)
        
        self.e_inf[0, self.idx_em] = 1.0
        self.e_inf[0, self.idx_ep] = 1.0
        
        self.e_o[0, self.idx_em] = 0.5
        self.e_o[0, self.idx_ep] = -0.5

    def to_cga(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds Euclidean points into Conformal points (null vectors).

        Mapping: P(x) = x + 0.5 * x^2 * e_inf + e_o

        Args:
            x (torch.Tensor): Euclidean points [Batch, d].

        Returns:
            torch.Tensor: Conformal points [Batch, 2^n].
        """
        batch_size = x.shape[0]
        
        # 1. Embed vector part x
        x_mv = torch.zeros(batch_size, self.algebra.dim, device=self.device)
        for i in range(self.d):
            basis_idx = 1 << i
            x_mv[:, basis_idx] = x[:, i]
            
        # 2. Compute x^2 (squared Euclidean norm)
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        
        # 3. Assemble P
        term2 = 0.5 * x_sq * self.e_inf
        term3 = self.e_o.expand(batch_size, -1)
        
        P = x_mv + term2 + term3
        return P

    def from_cga(self, P: torch.Tensor) -> torch.Tensor:
        """Extracts Euclidean points from normalized Conformal points.

        Normalization: P -> P / (-P . e_inf)

        Args:
            P (torch.Tensor): Conformal points [Batch, 2^n].

        Returns:
            torch.Tensor: Euclidean coordinates [Batch, d].
        """
        # 1. Normalize P such that P inner e_inf = -1
        # P . e_inf is the scalar part of the geometric product
        P_einf = self.algebra.geometric_product(P, self.e_inf)
        scale = -P_einf[..., 0:1] # Scalar part
        
        P_norm = P / (scale + 1e-6)
        
        # 2. Extract vector components (indices 0 to d-1)
        x = torch.zeros(P.shape[0], self.d, device=self.device)
        for i in range(self.d):
            basis_idx = 1 << i
            x[:, i] = P_norm[:, basis_idx]
            
        return x
