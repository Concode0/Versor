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
from layers.base import CliffordModule
from layers.linear import CliffordLinear
from torch_geometric.nn import global_add_pool

class GeometricInteraction(CliffordModule):
    """Geometric Algebra Interaction Layer.

    Updates multivector node features based on relative position vectors.
    Uses CliffordLinear for feature transformation and multivector gating.
    """

    def __init__(self, algebra: CliffordAlgebra, hidden_dim: int):
        super().__init__(algebra)
        # Clifford Linear Transformation
        self.lin = CliffordLinear(algebra, hidden_dim, hidden_dim)
        
        # Multivector Gate Generator (Learns coefficients for a gate multivector)
        # Input: scalar invariants -> Output: [Hidden, Algebra_Dim]
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * algebra.dim)
        )

    def forward(self, h, pos, edge_index):
        """
        Args:
            h: Node features [N, Hidden_Dim, Algebra_Dim] (Multivectors).
            pos: Coordinates [N, 3].
            edge_index: Graph connectivity [2, E].
        """
        row, col = edge_index
        
        # 1. Construct Relative Position Vector r_ij
        coord_diff = pos[row] - pos[col] # [E, 3]
        dist = torch.norm(coord_diff, dim=-1, keepdim=True) # [E, 1]
        
        E = row.shape[0]
        r_mv = torch.zeros(E, 1, self.algebra.dim, device=h.device)
        r_mv[:, 0, 1] = coord_diff[:, 0]
        r_mv[:, 0, 2] = coord_diff[:, 1]
        r_mv[:, 0, 4] = coord_diff[:, 2]
        
        # Normalize direction
        r_mv = r_mv / (dist.unsqueeze(-1) + 1e-6)
        
        # 2. Transform Neighbor Features
        h_j = self.lin(h[col]) 
        
        # 3. Geometric Mixing: prod = h_j * r_ij
        prod = self.algebra.geometric_product(h_j, r_mv)
        
        # 4. Multivector Gating (Fully Geometric)
        # Compute scalar invariants for gating
        h_scalar = h[..., 0] 
        edge_feat = torch.cat([h_scalar[row], h_scalar[col], dist], dim=-1)
        
        # Generate Gate Multivectors [E, H, D]
        gate_coeffs = self.gate_mlp(edge_feat).view(E, -1, self.algebra.dim)
        
        # Apply gate via Geometric Product: msg = gate * (h_j * r_ij)
        msg = self.algebra.geometric_product(gate_coeffs, prod)
        
        # 5. Aggregate
        N = h.shape[0]
        out = torch.zeros_like(h)
        out.index_add_(0, row, msg)
        
        return h + out

class MoleculeGNN(CliffordModule):
    """Geometric GNN for molecular property prediction."""

    def __init__(self, algebra: CliffordAlgebra, hidden_dim=16, layers=3):
        super().__init__(algebra)
        self.embedding = nn.Embedding(10, hidden_dim) # Atomic Num -> Scalar
        self.layers = nn.ModuleList([
            GeometricInteraction(algebra, hidden_dim) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm([hidden_dim, algebra.dim]) for _ in range(layers)
        ])
        self.readout_mlp = nn.Linear(hidden_dim, 1)

    def forward(self, z, pos, batch):
        # Build edges (Radius graph)
        # We can use radius_graph from PyG, but sticking to fully connected for now
        edge_indices = []
        count = 0
        for i in range(batch.max().item() + 1):
            mask = (batch == i)
            n = mask.sum().item()
            idx = torch.arange(n, device=z.device) + count
            r = idx.repeat_interleave(n)
            c = idx.repeat(n)
            mask_self = r != c
            edge_indices.append(torch.stack([r[mask_self], c[mask_self]]))
            count += n
        edge_index = torch.cat(edge_indices, dim=1)
        
        # Embedding
        h_scalar = self.embedding(z)
        h = torch.zeros(z.shape[0], h_scalar.shape[1], self.algebra.dim, device=z.device)
        h[..., 0] = h_scalar
        
        # Message Passing
        for layer, norm in zip(self.layers, self.norms):
            h = norm(layer(h, pos, edge_index))
            
        # 3. Global Pooling (Efficient)
        # h is [N, H, D]. PyG global_add_pool expects [N, F].
        # We flatten H and D into features.
        N, H, D = h.shape
        h_flat = h.view(N, -1)
        pooled_flat = global_add_pool(h_flat, batch)
        pooled = pooled_flat.view(-1, H, D) # [Batch, H, D]
            
        # 4. Readout
        coeffs = pooled.permute(0, 2, 1) 
        out_coeffs = self.readout_mlp(coeffs).squeeze(-1) 
        
        return out_coeffs
