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
from layers.rotor import RotorLayer
from layers.multi_rotor import MultiRotorLayer
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from functional.activation import GeometricGELU


def _scatter_mean(src, index, dim_size, dim=0):
    """Scatter mean for variable-size graph pooling."""
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    out.index_add_(dim, index, src)
    count.index_add_(dim, index, torch.ones(src.size(dim), device=src.device, dtype=src.dtype))
    count = count.clamp(min=1)
    # Reshape count for broadcasting
    for _ in range(src.dim() - 1):
        count = count.unsqueeze(-1)
    return out / count


def _scatter_add(src, index, dim_size, dim=0):
    """Scatter add for variable-size graph pooling."""
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.index_add_(dim, index, src)
    return out


class ProteinEncoder(CliffordModule):
    """Encodes protein pocket atoms into multivector features.

    Uses CliffordLinear layers with optional rotor backend,
    followed by MultiRotorLayer for geometric mixing.
    """

    def __init__(self, algebra, hidden_dim=64, num_layers=4, num_rotors=8,
                 max_z=100, num_aa=20, use_decomposition=False, decomp_k=10,
                 use_rotor_backend=False):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim

        self.atom_embed = nn.Embedding(max_z, hidden_dim)
        self.aa_embed = nn.Embedding(num_aa, hidden_dim)

        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'lin': CliffordLinear(algebra, hidden_dim, hidden_dim, backend=backend),
                'norm': CliffordLayerNorm(algebra, hidden_dim),
                'act': GeometricGELU(algebra, channels=hidden_dim),
                'rotor': MultiRotorLayer(
                    algebra, hidden_dim, num_rotors,
                    use_decomposition=use_decomposition, decomp_k=decomp_k
                ),
            }))

        # BladeSelector on output to suppress irrelevant grades
        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * algebra.num_grades, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Linear(hidden_dim, algebra.dim)

    def forward(self, pos, z, aa, edge_index, batch):
        """Encode protein pocket.

        Args:
            pos: Atom positions [N, 3]
            z: Atom types [N]
            aa: Amino acid types [N]
            edge_index: Connectivity [2, E]
            batch: Batch assignment [N]

        Returns:
            h: Multivector features [N, Hidden, Dim]
        """
        # Initial embedding (scalar + AA info)
        h_scalar = self.atom_embed(z) + self.aa_embed(aa)
        h = torch.zeros(z.size(0), h_scalar.size(1), self.algebra.dim, device=pos.device)
        h[..., 0] = h_scalar

        for layer in self.layers:
            row, col = edge_index

            # Relative positions
            r_ij = pos[row] - pos[col]
            r_ij_mv = self.algebra.embed_vector(r_ij)

            # Transform + normalize + activate
            h_t = layer['lin'](h)
            h_t = layer['norm'](h_t)
            h_t = layer['act'](h_t)

            # Geometric interaction
            h_j = h_t[col]
            psi = self.algebra.geometric_product(h_j, r_ij_mv.unsqueeze(1))
            phi = layer['rotor'](psi)

            # Invariant message
            inv = self.algebra.get_grade_norms(phi)
            inv_flat = inv.view(inv.size(0), -1)
            msg = self.msg_mlp(inv_flat)

            out_msg = torch.zeros_like(h)
            out_msg.index_add_(0, row, msg.unsqueeze(-1) * self.gate(msg).unsqueeze(1))
            h = h + out_msg

        return self.blade_selector(h)


class LigandEncoder(CliffordModule):
    """Encodes ligand atoms into multivector features.

    Smaller than ProteinEncoder (fewer layers) since ligands are typically
    much smaller molecules.
    """

    def __init__(self, algebra, hidden_dim=32, num_layers=3, num_rotors=8,
                 max_z=100, use_decomposition=False, decomp_k=10,
                 use_rotor_backend=False):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim

        self.atom_embed = nn.Embedding(max_z, hidden_dim)

        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'lin': CliffordLinear(algebra, hidden_dim, hidden_dim, backend=backend),
                'norm': CliffordLayerNorm(algebra, hidden_dim),
                'act': GeometricGELU(algebra, channels=hidden_dim),
                'rotor': MultiRotorLayer(
                    algebra, hidden_dim, num_rotors,
                    use_decomposition=use_decomposition, decomp_k=decomp_k
                ),
            }))

        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * algebra.num_grades, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Linear(hidden_dim, algebra.dim)

    def forward(self, pos, z, edge_index, batch):
        """Encode ligand.

        Args:
            pos: Atom positions [N, 3]
            z: Atom types [N]
            edge_index: Connectivity [2, E]
            batch: Batch assignment [N]

        Returns:
            h: Multivector features [N, Hidden, Dim]
        """
        h_scalar = self.atom_embed(z)
        h = torch.zeros(z.size(0), h_scalar.size(1), self.algebra.dim, device=pos.device)
        h[..., 0] = h_scalar

        for layer in self.layers:
            row, col = edge_index
            r_ij = pos[row] - pos[col]
            r_ij_mv = self.algebra.embed_vector(r_ij)

            h_t = layer['lin'](h)
            h_t = layer['norm'](h_t)
            h_t = layer['act'](h_t)

            h_j = h_t[col]
            psi = self.algebra.geometric_product(h_j, r_ij_mv.unsqueeze(1))
            phi = layer['rotor'](psi)

            inv = self.algebra.get_grade_norms(phi)
            inv_flat = inv.view(inv.size(0), -1)
            msg = self.msg_mlp(inv_flat)

            out_msg = torch.zeros_like(h)
            out_msg.index_add_(0, row, msg.unsqueeze(-1) * self.gate(msg).unsqueeze(1))
            h = h + out_msg

        return self.blade_selector(h)


class GeometricCrossAttention(CliffordModule):
    """Geometric cross-attention between protein and ligand.

    Computes protein-ligand interactions via rotor-mediated geometric products.
    Uses distance-gated attention to focus on nearby interactions.
    """

    def __init__(self, algebra, protein_dim, ligand_dim, interaction_dim,
                 num_rotors=8, use_decomposition=False, decomp_k=10):
        super().__init__(algebra)

        self.protein_rotor = RotorLayer(
            algebra, protein_dim,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )
        self.ligand_rotor = RotorLayer(
            algebra, ligand_dim,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # Project to common interaction dimension
        self.protein_proj = nn.Linear(protein_dim * algebra.dim, interaction_dim)
        self.ligand_proj = nn.Linear(ligand_dim * algebra.dim, interaction_dim)

        # Attention over interactions
        self.attn_mlp = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim // 2),
            nn.SiLU(),
            nn.Linear(interaction_dim // 2, 1)
        )

        # Distance gate (closer atoms interact more)
        self.dist_gate = nn.Sequential(
            nn.Linear(1, interaction_dim // 4),
            nn.SiLU(),
            nn.Linear(interaction_dim // 4, 1),
            nn.Sigmoid()
        )

        self.interaction_dim = interaction_dim

    def forward(self, protein_h, ligand_h, protein_pos, ligand_pos,
                protein_batch, ligand_batch, batch_size):
        """Compute cross-attention between protein and ligand.

        Args:
            protein_h: Protein features [N_prot, H_p, Dim]
            ligand_h: Ligand features [N_lig, H_l, Dim]
            protein_pos: Protein positions [N_prot, 3]
            ligand_pos: Ligand positions [N_lig, 3]
            protein_batch: Batch indices [N_prot]
            ligand_batch: Batch indices [N_lig]
            batch_size: Number of graphs in batch

        Returns:
            interaction: Pooled interaction features [B, interaction_dim]
        """
        # Apply rotors
        p_aligned = self.protein_rotor(protein_h)  # [N_prot, H_p, Dim]
        l_aligned = self.ligand_rotor(ligand_h)    # [N_lig, H_l, Dim]

        # Flatten multivector dims for projection
        p_flat = p_aligned.view(p_aligned.size(0), -1)  # [N_prot, H_p*Dim]
        l_flat = l_aligned.view(l_aligned.size(0), -1)  # [N_lig, H_l*Dim]

        p_proj = self.protein_proj(p_flat)  # [N_prot, inter_dim]
        l_proj = self.ligand_proj(l_flat)  # [N_lig, inter_dim]

        # Per-graph cross-attention
        results = []
        for b in range(batch_size):
            p_mask = (protein_batch == b)
            l_mask = (ligand_batch == b)

            p_b = p_proj[p_mask]   # [n_p, inter_dim]
            l_b = l_proj[l_mask]   # [n_l, inter_dim]
            p_pos_b = protein_pos[p_mask]  # [n_p, 3]
            l_pos_b = ligand_pos[l_mask]   # [n_l, 3]

            if p_b.size(0) == 0 or l_b.size(0) == 0:
                results.append(torch.zeros(self.interaction_dim, device=p_proj.device))
                continue

            # Pairwise distances
            dists = torch.cdist(p_pos_b, l_pos_b)  # [n_p, n_l]
            dist_weights = self.dist_gate(dists.unsqueeze(-1)).squeeze(-1)  # [n_p, n_l]

            # Pairwise interaction features
            p_exp = p_b.unsqueeze(1).expand(-1, l_b.size(0), -1)  # [n_p, n_l, dim]
            l_exp = l_b.unsqueeze(0).expand(p_b.size(0), -1, -1)  # [n_p, n_l, dim]
            interaction = p_exp * l_exp  # Element-wise interaction

            # Attention weights
            attn_scores = self.attn_mlp(interaction).squeeze(-1)  # [n_p, n_l]
            attn_scores = attn_scores * dist_weights
            attn_weights = torch.softmax(attn_scores.view(-1), dim=0).view_as(attn_scores)

            # Weighted pool
            pooled = (attn_weights.unsqueeze(-1) * interaction).sum(dim=[0, 1])
            results.append(pooled)

        return torch.stack(results, dim=0)  # [B, inter_dim]


class PDBBindNet(CliffordModule):
    """Dual-graph network for protein-ligand binding affinity prediction.

    Architecture:
        1. ProteinEncoder: process pocket atoms
        2. LigandEncoder: process ligand atoms
        3. GeometricCrossAttention: rotor-mediated interactions
        4. Affinity head: MLP → scalar prediction
    """

    def __init__(self, algebra, protein_hidden_dim=64, ligand_hidden_dim=32,
                 interaction_dim=64, num_protein_layers=4, num_ligand_layers=3,
                 num_rotors=8, use_decomposition=False, decomp_k=10,
                 use_rotor_backend=False):
        super().__init__(algebra)
        self._last_protein_features = None
        self._last_ligand_features = None

        self.protein_encoder = ProteinEncoder(
            algebra, hidden_dim=protein_hidden_dim, num_layers=num_protein_layers,
            num_rotors=num_rotors, use_decomposition=use_decomposition,
            decomp_k=decomp_k, use_rotor_backend=use_rotor_backend
        )
        self.ligand_encoder = LigandEncoder(
            algebra, hidden_dim=ligand_hidden_dim, num_layers=num_ligand_layers,
            num_rotors=num_rotors, use_decomposition=use_decomposition,
            decomp_k=decomp_k, use_rotor_backend=use_rotor_backend
        )
        self.cross_attention = GeometricCrossAttention(
            algebra, protein_hidden_dim, ligand_hidden_dim,
            interaction_dim=interaction_dim, num_rotors=num_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # Affinity prediction head
        self.affinity_head = nn.Sequential(
            nn.Linear(interaction_dim, interaction_dim),
            nn.SiLU(),
            nn.Linear(interaction_dim, interaction_dim // 2),
            nn.SiLU(),
            nn.Linear(interaction_dim // 2, 1)
        )

    def forward(self, protein_pos, protein_z, protein_aa, protein_edge_index,
                protein_batch, ligand_pos, ligand_z, ligand_edge_index,
                ligand_batch, batch_size):
        """Predict binding affinity.

        Returns:
            affinity: Predicted pKd values [B]
        """
        # Encode protein and ligand
        protein_h = self.protein_encoder(
            protein_pos, protein_z, protein_aa, protein_edge_index, protein_batch
        )
        ligand_h = self.ligand_encoder(
            ligand_pos, ligand_z, ligand_edge_index, ligand_batch
        )

        # Store features for metric computation
        self._last_protein_features = protein_h.detach()
        self._last_ligand_features = ligand_h.detach()

        # Cross-attention interaction
        interaction = self.cross_attention(
            protein_h, ligand_h, protein_pos, ligand_pos,
            protein_batch, ligand_batch, batch_size
        )

        # Predict affinity
        affinity = self.affinity_head(interaction).squeeze(-1)
        return affinity

    def get_latent_features(self):
        """Return last protein and ligand multivector features."""
        return self._last_protein_features, self._last_ligand_features

    def total_sparsity_loss(self):
        """Collect sparsity loss from all MultiRotor and Rotor layers."""
        loss = torch.tensor(0.0)
        for module in self.modules():
            if hasattr(module, 'sparsity_loss') and module is not self:
                loss = loss + module.sparsity_loss()
        return loss
