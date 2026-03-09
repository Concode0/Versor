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
from layers import CliffordModule
from layers import CliffordLinear
from layers import MultiRotorLayer
from layers import CliffordLayerNorm
from layers import BladeSelector
from functional.activation import GeometricGELU, GeometricSquare
from torch_geometric.nn import global_add_pool


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions for distance encoding.

    Centers are evenly spaced in [0, cutoff] with fixed width sigma.
    """

    def __init__(self, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0, cutoff, num_rbf)
        self.register_buffer('centers', centers)
        self.sigma = cutoff / num_rbf

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Encode distances as Gaussian RBF features.

        Args:
            distances: Pairwise distances [E].

        Returns:
            RBF features [E, num_rbf].
        """
        d = distances.unsqueeze(-1)  # [E, 1]
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.sigma ** 2))


class DynamicRotorGenerator(CliffordModule):
    """Generates per-edge bivector coefficients from invariant features.

    Maps edge-level invariant features to bivector space, then exponentiates
    to produce per-edge rotors. Zero-initialized so dynamic rotors start
    as identity (exp(0) = 1).
    """

    def __init__(self, algebra: CliffordAlgebra, input_dim: int, num_dynamic_rotors: int = 4):
        super().__init__(algebra)
        self.num_dynamic_rotors = num_dynamic_rotors

        bv_mask = algebra.grade_masks[2]
        self.register_buffer('bivector_indices', bv_mask.nonzero(as_tuple=False).squeeze(-1))
        self.num_bivectors = len(self.bivector_indices)

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, num_dynamic_rotors * self.num_bivectors),
        )
        # Zero-init last layer so dynamic rotors start as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inv_features: torch.Tensor) -> tuple:
        """Generate per-edge rotors from invariant features.

        Args:
            inv_features: Invariant edge features [E, input_dim].

        Returns:
            R: Per-edge rotors [E, K_d, dim].
            R_rev: Per-edge reversed rotors [E, K_d, dim].
        """
        bv_coeffs = self.net(inv_features)  # [E, K_d * num_bv]
        bv_coeffs = bv_coeffs.view(-1, self.num_dynamic_rotors, self.num_bivectors)  # [E, K_d, num_bv]

        E = bv_coeffs.size(0)
        B = torch.zeros(E, self.num_dynamic_rotors, self.algebra.dim,
                        device=bv_coeffs.device, dtype=bv_coeffs.dtype)
        indices = self.bivector_indices.unsqueeze(0).unsqueeze(0).expand(E, self.num_dynamic_rotors, -1)
        B.scatter_(2, indices, bv_coeffs)

        # Flatten for exp, then reshape back
        B_flat = B.reshape(E * self.num_dynamic_rotors, self.algebra.dim)
        R_flat = self.algebra.exp(-0.5 * B_flat)
        R = R_flat.reshape(E, self.num_dynamic_rotors, self.algebra.dim)
        R_rev = self.algebra.reverse(R_flat).reshape(E, self.num_dynamic_rotors, self.algebra.dim)

        return R, R_rev


def _embed_pga_vector(algebra: CliffordAlgebra, vectors: torch.Tensor) -> torch.Tensor:
    """Embed 3D vectors into PGA Cl(3,0,1) as direction vectors.

    In PGA, a 3D point p = (x,y,z) is represented as the grade-1 element
    x*e1 + y*e2 + z*e3. The null basis e4 (e0 in PGA convention) is used
    for the homogeneous coordinate in point representations, but for
    relative displacement vectors we only need the spatial components.

    Falls back to algebra.embed_vector for non-PGA algebras.

    Args:
        algebra: CliffordAlgebra instance.
        vectors: 3D vectors [..., 3].

    Returns:
        Multivector [..., dim].
    """
    if algebra.n <= 3:
        return algebra.embed_vector(vectors)
    # For PGA Cl(3,0,1): embed into first 3 grade-1 slots (e1, e2, e3)
    batch_shape = vectors.shape[:-1]
    mv = torch.zeros(*batch_shape, algebra.dim, device=vectors.device, dtype=vectors.dtype)
    for i in range(3):
        mv[..., 1 << i] = vectors[..., i]
    return mv


class MD17InteractionBlock(CliffordModule):
    """Geometric interaction block for molecular dynamics with PGA support.

    Combines relative positions with node features via geometric product,
    applies static + dynamic rotors weighted by edge invariants, and
    optionally uses GeometricSquare activation for algebraic cross-terms.
    """

    def __init__(self, algebra: CliffordAlgebra, hidden_dim: int,
                 num_static_rotors: int = 8, num_dynamic_rotors: int = 4,
                 num_rbf: int = 20, rbf_cutoff: float = 5.0,
                 use_decomposition: bool = False, decomp_k: int = 10,
                 use_rotor_backend: bool = False, use_geo_square: bool = True):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim
        self.num_static_rotors = num_static_rotors
        self.num_dynamic_rotors = num_dynamic_rotors
        self.num_total_rotors = num_static_rotors + num_dynamic_rotors

        # Feature transformation
        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.lin_h = CliffordLinear(algebra, hidden_dim, hidden_dim, backend=backend)
        self.norm = CliffordLayerNorm(algebra, hidden_dim)
        self.act = GeometricGELU(algebra, channels=hidden_dim)

        # GeometricSquare activation (gated GP self-product)
        self.use_geo_square = use_geo_square
        if use_geo_square:
            self.geo_square = GeometricSquare(algebra, channels=hidden_dim)

        # RBF distance encoding
        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=rbf_cutoff)
        self.rbf_proj = nn.Linear(num_rbf, hidden_dim)

        # Static rotors (shared across edges)
        self.multi_rotor = MultiRotorLayer(
            algebra, hidden_dim, num_static_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # Dynamic rotors (per-edge, from invariant features)
        # Input: grade norms (hidden_dim * num_grades) + rbf projection (hidden_dim)
        inv_dim = hidden_dim * algebra.num_grades + hidden_dim
        self.dynamic_rotor_gen = DynamicRotorGenerator(
            algebra, input_dim=inv_dim, num_dynamic_rotors=num_dynamic_rotors
        )

        # Edge weight network: maps invariants to weights over all rotors
        self.edge_weight_net = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_total_rotors)
        )

        # Invariant-based scalar gate
        self.msg_gate = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h, pos, edge_index):
        """Forward pass with PGA geometric message passing.

        Args:
            h: Node features [N, Hidden, Dim]
            pos: Atom positions [N, 3]
            edge_index: Edge connectivity [2, E]

        Returns:
            Updated node features [N, Hidden, Dim]
        """
        row, col = edge_index

        # 1. Relative vector and distance
        r_ij = pos[row] - pos[col]  # [E, 3]
        d_ij = r_ij.norm(dim=-1)  # [E]
        r_ij_mv = _embed_pga_vector(self.algebra, r_ij)  # [E, Dim]

        # 2. RBF distance features
        rbf_feat = self.rbf(d_ij)  # [E, num_rbf]
        rbf_proj = self.rbf_proj(rbf_feat)  # [E, hidden_dim]

        # 3. Transform node features
        h_t = self.lin_h(h)
        h_t = self.norm(h_t)
        h_t = self.act(h_t)

        # 4. Geometric product with relative position
        psi = self.algebra.geometric_product(h_t[col], r_ij_mv.unsqueeze(1))  # [E, Hidden, Dim]

        # 5. GeometricSquare activation
        if self.use_geo_square:
            psi = self.geo_square(psi)  # [E, Hidden, Dim]

        # 6. Compute invariant features (grade norms + rbf)
        psi_inv = self.algebra.get_grade_norms(psi)  # [E, Hidden, num_grades]
        psi_inv_flat = psi_inv.reshape(psi_inv.size(0), -1)  # [E, Hidden * num_grades]
        inv_feat = torch.cat([psi_inv_flat, rbf_proj], dim=-1)  # [E, Hidden*num_grades + Hidden]

        # 7. Edge weights over all rotors (static + dynamic)
        edge_weights = self.edge_weight_net(inv_feat)  # [E, K_total]
        edge_weights = torch.softmax(edge_weights, dim=-1)

        # 8. Static rotors
        R_static, R_static_rev = self.multi_rotor._compute_rotors(psi.device, psi.dtype)
        # R_static: [K_s, Dim], R_static_rev: [K_s, Dim]

        # 9. Dynamic rotors
        R_dynamic, R_dynamic_rev = self.dynamic_rotor_gen(inv_feat)
        # R_dynamic: [E, K_d, Dim], R_dynamic_rev: [E, K_d, Dim]

        # 10. Apply static rotors via sandwich product
        psi_expanded = psi.unsqueeze(2)  # [E, Hidden, 1, Dim]
        R_s = R_static.view(1, 1, self.num_static_rotors, -1)  # [1, 1, K_s, Dim]
        R_s_rev = R_static_rev.view(1, 1, self.num_static_rotors, -1)

        R_psi_s = self.algebra.geometric_product(R_s, psi_expanded)
        rotated_static = self.algebra.geometric_product(R_psi_s, R_s_rev)  # [E, Hidden, K_s, Dim]

        # 11. Apply dynamic rotors via sandwich product
        R_d = R_dynamic.unsqueeze(1)  # [E, 1, K_d, Dim]
        R_d_rev = R_dynamic_rev.unsqueeze(1)  # [E, 1, K_d, Dim]

        R_psi_d = self.algebra.geometric_product(R_d, psi_expanded)
        rotated_dynamic = self.algebra.geometric_product(R_psi_d, R_d_rev)  # [E, Hidden, K_d, Dim]

        # 12. Concatenate and weight-sum all rotated messages
        rotated_all = torch.cat([rotated_static, rotated_dynamic], dim=2)  # [E, Hidden, K_total, Dim]
        phi = torch.einsum('ek,ehkd->ehd', edge_weights, rotated_all)  # [E, Hidden, Dim]

        # 13. Invariant gating
        gate = self.msg_gate(inv_feat)  # [E, Hidden]
        phi_gated = phi * gate.unsqueeze(-1)  # [E, Hidden, Dim]

        # 14. Aggregate messages with skip connection
        out_msg = torch.zeros_like(h)
        out_msg.index_add_(0, row, phi_gated)

        return h + out_msg


class MD17ForceNet(CliffordModule):
    """Force prediction network for MD17 with PGA motors.

    Uses Cl(3,0,1) for SE(3) equivariant molecular dynamics predictions.
    Combines static shared rotors with input-dependent dynamic rotors
    and RBF distance encoding.

    Dual-head architecture:
    - Energy head: scalar projection -> global pool -> energy
    - Force head: F = -grad(E) via autograd
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        hidden_dim: int,
        num_layers: int = 4,
        num_static_rotors: int = 8,
        num_dynamic_rotors: int = 4,
        max_z: int = 100,
        num_rbf: int = 20,
        rbf_cutoff: float = 5.0,
        use_decomposition: bool = False,
        decomp_k: int = 10,
        use_rotor_backend: bool = False,
        use_geo_square: bool = True,
    ):
        super().__init__(algebra)
        self._last_features = None

        # Atomic number embedding
        self.atom_embedding = nn.Embedding(max_z, hidden_dim)

        # Geometric interaction layers
        self.layers = nn.ModuleList([
            MD17InteractionBlock(
                algebra, hidden_dim,
                num_static_rotors=num_static_rotors,
                num_dynamic_rotors=num_dynamic_rotors,
                num_rbf=num_rbf,
                rbf_cutoff=rbf_cutoff,
                use_decomposition=use_decomposition,
                decomp_k=decomp_k,
                use_rotor_backend=use_rotor_backend,
                use_geo_square=use_geo_square,
            )
            for _ in range(num_layers)
        ])

        # Grade selection before output
        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)

        # Energy head: flatten multivector -> energy
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
        """Collects sparsity loss from all MultiRotor layers and dynamic rotor generators."""
        loss = torch.tensor(0.0)
        for layer in self.layers:
            if hasattr(layer, 'multi_rotor'):
                loss = loss + layer.multi_rotor.sparsity_loss()
            if hasattr(layer, 'dynamic_rotor_gen'):
                # L1 on dynamic rotor generator weights
                for p in layer.dynamic_rotor_gen.net.parameters():
                    loss = loss + torch.norm(p, p=1) * 0.01
        return loss
