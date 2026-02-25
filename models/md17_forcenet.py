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
from layers.multi_rotor import MultiRotorLayer
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from functional.activation import GeometricGELU
from torch_geometric.nn import global_add_pool


class MD17InteractionBlock(CliffordModule):
    """Geometric interaction block for molecular dynamics.

    Combines relative positions with node features via geometric product,
    preserving physical symmetries (rotation, translation invariance).
    """

    def __init__(self, algebra: CliffordAlgebra, hidden_dim: int, num_rotors: int = 8,
                 use_decomposition: bool = False, decomp_k: int = 10,
                 use_rotor_backend: bool = False):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim

        # Feature transformation
        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.lin_h = CliffordLinear(algebra, hidden_dim, hidden_dim, backend=backend)
        self.norm = CliffordLayerNorm(algebra, hidden_dim)
        self.act = GeometricGELU(algebra, channels=hidden_dim)

        # Multi-Rotor layer for geometric mixing
        self.multi_rotor = MultiRotorLayer(
            algebra, hidden_dim, num_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        self.edge_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * algebra.num_grades, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_rotors)
        )

        # Invariant-based scalar gate: controls message strength per channel
        # Uses grade norms (invariant) to compute scalar attention weights,
        # while the full geometric multivector phi carries directional info.
        self.msg_gate = nn.Sequential(
            nn.Linear(hidden_dim * algebra.num_grades, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h, pos, edge_index):
        """Forward pass with geometric message passing.

        Args:
            h: Node features [N, Hidden, Dim]
            pos: Atom positions [N, 3]
            edge_index: Edge connectivity [2, E]

        Returns:
            Updated node features [N, Hidden, Dim]
        """
        row, col = edge_index
        N = h.size(0)

        # Relative vector r_ij
        r_ij = pos[row] - pos[col]
        r_ij_mv = self.algebra.embed_vector(r_ij)  # [E, Dim]

        # Transform, normalize, activate node features
        h_t = self.lin_h(h)
        h_t = self.norm(h_t)
        h_t = self.act(h_t)
        h_j = h_t[col]  # [E, Hidden, Dim]

        psi = self.algebra.geometric_product(h_j, r_ij_mv.unsqueeze(1)) # [E, Hidden, Dim]

        psi_inv = self.algebra.get_grade_norms(psi)
        psi_inv_flat = psi_inv.reshape(psi_inv.size(0), -1)

        edge_weights = self.edge_weight_net(psi_inv_flat)
        edge_weights = torch.softmax(edge_weights, dim=-1)

        R, R_rev = self.multi_rotor._compute_rotors(psi.device, psi.dtype)

        psi_expanded = psi.unsqueeze(2) # [E, Hidden, 1, Dim]
        R_expanded = R.view(1, 1, self.multi_rotor.num_rotors, -1) # [1, 1, K, Dim]
        R_rev_expanded = R_rev.view(1, 1, self.multi_rotor.num_rotors, -1) # [1, 1, K, Dim]

        R_psi = self.algebra.geometric_product(R_expanded, psi_expanded)
        rotated_psi = self.algebra.geometric_product(R_psi, R_rev_expanded) # [E, Hidden, K, Dim]

        phi_dynamic = torch.einsum('ek,ehkd->ehd', edge_weights, rotated_psi) # [E, Hidden, Dim]

        # Invariant gating: grade norms for scalar attention, phi for geometry
        inv_features = self.algebra.get_grade_norms(phi_dynamic) # [E, Hidden, Num_Grades]
        inv_flat = inv_features.reshape(inv_features.size(0), -1)
        gate = self.msg_gate(inv_flat) # [E, Hidden] — scalar weights

        phi_gated = phi_dynamic * gate.unsqueeze(-1) # [E, Hidden, Dim]

        out_msg = torch.zeros_like(h)
        out_msg.index_add_(0, row, phi_gated)

        return h + out_msg


class MD17ForceNet(CliffordModule):
    """Force prediction network for MD17.

    Dual-head architecture:
    - Energy head: scalar projection → total energy
    - Force head: vector projection → per-atom forces

    Uses geometric algebra to maintain E(3) equivariance for forces.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        hidden_dim: int,
        num_layers: int = 4,
        num_rotors: int = 8,
        max_z: int = 100,
        use_decomposition: bool = False,
        decomp_k: int = 10,
        use_rotor_backend: bool = False
    ):
        """Initialize MD17ForceNet.

        Args:
            algebra: Clifford algebra (should be Cl(3,0))
            hidden_dim: Hidden dimension
            num_layers: Number of interaction layers
            num_rotors: Number of rotors in multi-rotor layers
            max_z: Maximum atomic number (for embedding)
            use_decomposition: Enable bivector decomposition in MultiRotorLayers
            decomp_k: Number of simple components for decomposition
            use_rotor_backend: Use RotorGadget backend for CliffordLinear
        """
        super().__init__(algebra)
        self._last_features = None

        # Atomic number embedding
        self.atom_embedding = nn.Embedding(max_z, hidden_dim)

        # Geometric interaction layers
        self.layers = nn.ModuleList([
            MD17InteractionBlock(
                algebra, hidden_dim, num_rotors,
                use_decomposition=use_decomposition,
                decomp_k=decomp_k,
                use_rotor_backend=use_rotor_backend
            )
            for _ in range(num_layers)
        ])

        # Grade selection before output
        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)

        # Energy head: grade-0 (scalar) → global pool → energy
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim * algebra.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, pos, batch, edge_index):
        """Forward pass predicting both energy and forces via Autograd.

        Args:
            z: Atomic numbers [N]
            pos: Atom positions [N, 3]
            batch: Batch assignment [N]
            edge_index: Edge connectivity [2, E]

        Returns:
            energy: Predicted energy per molecule [B]
            force: Predicted forces per atom [N, 3]
        """
        with torch.enable_grad():
            pos.requires_grad_(True)

            # Initial scalar embedding
            h_scalar = self.atom_embedding(z)  # [N, Hidden]
            h = torch.zeros(z.size(0), h_scalar.size(1), self.algebra.dim, device=z.device)
            h[..., 0] = h_scalar

            # Geometric message passing
            for layer in self.layers:
                h = layer(h, pos, edge_index)

            # Store features before output heads
            self._last_features = h.detach()

            # Grade selection and normalization before output
            h = self.blade_selector(h)
            h = self.output_norm(h)

            # Flatten multivector features
            h_flat = h.reshape(h.size(0), -1)  # [N, Hidden * Dim]

            # Energy prediction (per-molecule scalar)
            graph_repr = global_add_pool(h_flat, batch)  # [B, Hidden * Dim]
            energy = self.energy_head(graph_repr).squeeze(-1)  # [B]
            
            force = -torch.autograd.grad(
                outputs=energy,
                inputs=pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=self.training, 
                retain_graph=True
            )[0]
            
        return energy, force

    def get_latent_features(self):
        """Return last intermediate multivector features (before output heads)."""
        return self._last_features

    def total_sparsity_loss(self) -> torch.Tensor:
        """Collects sparsity loss from all MultiRotor layers."""
        loss = 0
        for layer in self.layers:
            if hasattr(layer, 'multi_rotor'):
                loss += layer.multi_rotor.sparsity_loss()
        return loss


