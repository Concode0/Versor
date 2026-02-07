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
from layers.linear import CliffordLinear
from torch_geometric.nn import global_add_pool

from layers.multi_rotor import MultiRotorLayer
from functional.activation import GeometricGELU

class GeometricInvariantBlock(CliffordModule):
    """Rotation Invariant Block. Physics doesn't care about your coordinate system.

    Uses relative positions and geometric products to compute features
    remain invariant under global rotations.
    """
    def __init__(self, algebra: CliffordAlgebra, hidden_dim: int):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim
        
        self.lin_h = CliffordLinear(algebra, hidden_dim, hidden_dim)
        
        # 2. Invariant Message Generator:
        # Fixed: num_grades instead of num_grades * 2 if we only use norms
        # Updated: Input dim is hidden_dim * num_grades (flattened) instead of num_grades (averaged)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * algebra.num_grades, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Gating Mechanism
        self.gate = nn.Linear(hidden_dim, algebra.dim)

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        
        # Relative vector r_ij
        r_ij = pos[row] - pos[col]
        r_ij_mv = self.algebra.embed_vector(r_ij) # [E, Dim]
        
        # Transform node features
        h_j = self.lin_h(h)[col] # [E, Hidden, Dim]
        
        # Interaction via Geometric Product
        psi = self.algebra.geometric_product(h_j, r_ij_mv.unsqueeze(1)) # [E, Hidden, Dim]
        
        # Extract Invariants
        inv_features = self.algebra.get_grade_norms(psi) # [E, Hidden, Num_Grades]
        # Flatten instead of mean to preserve channel info
        inv_flat = inv_features.view(inv_features.size(0), -1) # [E, Hidden * Num_Grades]
        
        msg = self.msg_mlp(inv_flat)
        
        out_msg = torch.zeros_like(h)
        out_msg.index_add_(0, row, msg.unsqueeze(-1) * self.gate(msg).unsqueeze(1))
        
        return h + out_msg

class MoleculeGNN(CliffordModule):
    """Pure Geometric GNN. Atoms in space.

    Standard MPNN, but with multivectors.
    """
    def __init__(self, algebra: CliffordAlgebra, hidden_dim: int, num_layers: int = 4):
        super().__init__(algebra)
        
        self.atom_embedding = nn.Embedding(10, hidden_dim)
        
        self.layers = nn.ModuleList([
            GeometricInvariantBlock(algebra, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * algebra.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, pos, batch, edge_index):
        h_scalar = self.atom_embedding(z) # [N, Hidden]
        h = torch.zeros(z.size(0), h_scalar.size(1), self.algebra.dim, device=z.device)
        h[..., 0] = h_scalar
        
        for layer in self.layers:
            h = layer(h, pos, edge_index)
            
        h_flat = h.view(h.size(0), -1)
        graph_repr = global_add_pool(h_flat, batch)
        
        return self.readout(graph_repr).squeeze(-1)

class MultiRotorInteractionBlock(CliffordModule):
    """Multi-Rotor Block. Geometric FFT in action.

    Uses superposition to handle complex interactions.
    """
    def __init__(self, algebra: CliffordAlgebra, hidden_dim: int, num_rotors: int = 8):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim
        
        self.lin_h = CliffordLinear(algebra, hidden_dim, hidden_dim)
        
        # Multi-Rotor Superposition for message passing
        self.multi_rotor = MultiRotorLayer(algebra, hidden_dim, num_rotors)
        
        # Invariant Message Generator:
        # Updated: Input dim is hidden_dim * num_grades (flattened)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * algebra.num_grades, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gating Mechanism
        self.gate = nn.Linear(hidden_dim, algebra.dim)

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        
        # Relative vector r_ij
        r_ij = pos[row] - pos[col]
        r_ij_mv = self.algebra.embed_vector(r_ij) # [E, Dim]
        
        # Transform node features
        h_j = self.lin_h(h)[col] # [E, Hidden, Dim]
        
        # Interaction via Geometric Product
        psi = self.algebra.geometric_product(h_j, r_ij_mv.unsqueeze(1)) # [E, Hidden, Dim]
        
        # Apply Multi-Rotor Superposition (Geometric FFT)
        # This "unbends" the interaction manifold
        phi = self.multi_rotor(psi)
        
        # Extract Invariants (Dimensionless Structure)
        inv_features = self.algebra.get_grade_norms(phi) # [E, Hidden, Num_Grades]
        # Flatten instead of mean
        inv_flat = inv_features.view(inv_features.size(0), -1) # [E, Hidden * Num_Grades]
        
        msg = self.msg_mlp(inv_flat)
        
        out_msg = torch.zeros_like(h)
        # Message aggregation
        out_msg.index_add_(0, row, msg.unsqueeze(-1) * self.gate(msg).unsqueeze(1))
        
        return h + out_msg

class MultiRotorQuantumNet(CliffordModule):
    """Multi-Rotor Quantum Net. Unbending molecules.

    The flagship model for molecular property prediction.
    """
    def __init__(self, algebra: CliffordAlgebra, hidden_dim: int, num_layers: int = 4, num_rotors: int = 8):
        super().__init__(algebra)
        
        self.atom_embedding = nn.Embedding(10, hidden_dim)
        
        self.layers = nn.ModuleList([
            MultiRotorInteractionBlock(algebra, hidden_dim, num_rotors) 
            for _ in range(num_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * algebra.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def total_sparsity_loss(self) -> torch.Tensor:
        """Collects sparsity loss from all MultiRotor layers."""
        loss = 0
        for layer in self.layers:
            if hasattr(layer, 'multi_rotor'):
                loss += layer.multi_rotor.sparsity_loss()
        return loss

    def forward(self, z, pos, batch, edge_index):
        # Initial scalar embedding
        h_scalar = self.atom_embedding(z) # [N, Hidden]
        h = torch.zeros(z.size(0), h_scalar.size(1), self.algebra.dim, device=z.device)
        h[..., 0] = h_scalar
        
        for layer in self.layers:
            h = layer(h, pos, edge_index)
            
        h_flat = h.view(h.size(0), -1)
        graph_repr = global_add_pool(h_flat, batch)
        
        return self.readout(graph_repr).squeeze(-1)