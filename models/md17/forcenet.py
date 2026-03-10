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
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from layers import CliffordModule
from layers import CliffordLinear
from layers import MultiRotorLayer
from layers import CliffordLayerNorm
from layers import BladeSelector
from functional.activation import GeometricGELU, GeometricSquare
try:
    from torch_geometric.nn import global_add_pool
except ImportError:
    global_add_pool = None


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

        # Precompute one-hot projection: [num_bv, D] -- avoids in-place scatter_ in forward
        one_hot = torch.zeros(self.num_bivectors, algebra.dim)
        one_hot[torch.arange(self.num_bivectors), self.bivector_indices] = 1.0
        self.register_buffer('bv_one_hot', one_hot)

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, num_dynamic_rotors * self.num_bivectors),
        )
        # Zero-init last layer so dynamic rotors start as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, inv_features: torch.Tensor) -> tuple:
        bv_coeffs = self.net(inv_features)  # [E, K_d * num_bv]
        bv_coeffs = bv_coeffs.view(-1, self.num_dynamic_rotors, self.num_bivectors)  # [E, K_d, num_bv]

        # Out-of-place scatter via matmul: [E, K_d, num_bv] x [num_bv, D] -> [E, K_d, D]
        B = bv_coeffs @ self.bv_one_hot.to(bv_coeffs.dtype)

        E = bv_coeffs.size(0)
        B_flat = B.reshape(E * self.num_dynamic_rotors, self.algebra.dim)
        R_flat = self.algebra.exp(-0.5 * B_flat)
        R = R_flat.reshape(E, self.num_dynamic_rotors, self.algebra.dim)
        R_rev = self.algebra.reverse(R_flat).reshape(E, self.num_dynamic_rotors, self.algebra.dim)

        return R, R_rev


def _embed_pga_vector(algebra: CliffordAlgebra, vectors: torch.Tensor) -> torch.Tensor:
    """Embed 3D vectors into grade-1 slots (e1, e2, e3); falls back to embed_vector for n<=3."""
    if algebra.n <= 3:
        return algebra.embed_vector(vectors)
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

        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.lin_h = CliffordLinear(algebra, hidden_dim, hidden_dim, backend=backend)
        self.norm = CliffordLayerNorm(algebra, hidden_dim)
        self.act = GeometricGELU(algebra, channels=hidden_dim)

        self.use_geo_square = use_geo_square
        if use_geo_square:
            self.geo_square = GeometricSquare(algebra, channels=hidden_dim)

        self.rbf = GaussianRBF(num_rbf=num_rbf, cutoff=rbf_cutoff)
        self.rbf_proj = nn.Linear(num_rbf, hidden_dim)

        self.multi_rotor = MultiRotorLayer(
            algebra, hidden_dim, num_static_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # inv_dim: grade norms (hidden_dim * num_grades) + rbf projection (hidden_dim)
        inv_dim = hidden_dim * algebra.num_grades + hidden_dim
        self.dynamic_rotor_gen = DynamicRotorGenerator(
            algebra, input_dim=inv_dim, num_dynamic_rotors=num_dynamic_rotors
        )

        self.edge_weight_net = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_total_rotors)
        )

        self.msg_gate = nn.Sequential(
            nn.Linear(inv_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h, pos, edge_index):
        row, col = edge_index

        r_ij = pos[row] - pos[col]          # [E, 3]
        d_ij = r_ij.norm(dim=-1)            # [E]
        r_ij_mv = _embed_pga_vector(self.algebra, r_ij)  # [E, Dim]

        rbf_feat = self.rbf(d_ij)           # [E, num_rbf]
        rbf_proj = self.rbf_proj(rbf_feat)  # [E, hidden_dim]

        h_t = self.act(self.norm(self.lin_h(h)))

        psi = self.algebra.geometric_product(h_t[col], r_ij_mv.unsqueeze(1))  # [E, Hidden, Dim]

        if self.use_geo_square:
            psi = self.geo_square(psi)

        psi_inv = self.algebra.get_grade_norms(psi)
        psi_inv_flat = psi_inv.reshape(psi_inv.size(0), -1)
        inv_feat = torch.cat([psi_inv_flat, rbf_proj], dim=-1)

        edge_weights = torch.softmax(self.edge_weight_net(inv_feat), dim=-1)  # [E, K_total]

        gp = self.algebra.geometric_product
        D = psi.size(-1)
        device, dtype = psi.device, psi.dtype
        basis = torch.eye(D, device=device, dtype=dtype)  # [D, D]

        # Vectorized sandwich product via precomputed action matrices [K, D, D]
        R_static, R_static_rev = self.multi_rotor._compute_rotors(device, dtype)
        if self.num_static_rotors > 0:
            Rb_s = gp(R_static.unsqueeze(1), basis.unsqueeze(0))   # [K, D, D]
            M_s = gp(Rb_s, R_static_rev.unsqueeze(1))              # [K, D, D]
            phi = torch.einsum('ek, kdl, ehd -> ehl',
                               edge_weights[:, :self.num_static_rotors], M_s, psi)
        else:
            phi = torch.zeros_like(psi)

        if self.num_dynamic_rotors > 0:
            R_dynamic, R_dynamic_rev = self.dynamic_rotor_gen(inv_feat)
            E_size, K_d = R_dynamic.shape[:2]
            R_flat = R_dynamic.reshape(E_size * K_d, D)
            R_rev_flat = R_dynamic_rev.reshape(E_size * K_d, D)
            Rb_d = gp(R_flat.unsqueeze(1), basis.unsqueeze(0))     # [E*K_d, D, D]
            M_d = gp(Rb_d, R_rev_flat.unsqueeze(1))                # [E*K_d, D, D]
            M_d = M_d.reshape(E_size, K_d, D, D)
            phi = phi + torch.einsum('ek, ekdl, ehd -> ehl',
                                     edge_weights[:, self.num_static_rotors:], M_d, psi)

        gate = self.msg_gate(inv_feat)  # [E, Hidden]
        phi_gated = phi * gate.unsqueeze(-1)

        # Out-of-place index_add to preserve 2nd-order gradient path
        out_msg = torch.zeros_like(h).index_add(0, row, phi_gated)

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
        use_checkpoint: bool = False,
    ):
        super().__init__(algebra)
        self._last_features = None
        self.use_checkpoint = use_checkpoint

        self.atom_embedding = nn.Embedding(max_z, hidden_dim)

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

        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim * algebra.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _run_layers(self, h, pos, edge_index, force_checkpoint=True):
        """Run interaction layers, optionally with gradient checkpointing.

        Args:
            force_checkpoint: If False, always skip checkpointing regardless of self.use_checkpoint.
                Must be False when called inside create_graph=True autograd.grad (they are incompatible).
        """
        import torch.utils.checkpoint as cp
        for layer in self.layers:
            if self.use_checkpoint and self.training and force_checkpoint:
                h = cp.checkpoint(layer, h, pos, edge_index, use_reentrant=False)
            else:
                h = layer(h, pos, edge_index)
        return h

    def forward_energy(self, z, pos, batch, edge_index):
        """Return only energy. Use in train_step to compute forces externally.

        This avoids retain_graph=True overhead by not running autograd.grad
        inside the forward pass. The caller computes forces via:
            force = -autograd.grad(energy_pred, pos, grad_outputs=ones, create_graph=True)[0]
        """
        h_scalar = self.atom_embedding(z)  # [N, Hidden]
        h = F.pad(h_scalar.unsqueeze(-1), (0, self.algebra.dim - 1))  # [N, H, D]

        # Disable checkpointing: incompatible with create_graph=True in force gradient
        h = self._run_layers(h, pos, edge_index, force_checkpoint=False)

        self._last_features = h.detach()

        h = self.output_norm(self.blade_selector(h))
        h_flat = h.reshape(h.size(0), -1)  # [N, Hidden * Dim]

        graph_repr = global_add_pool(h_flat, batch)  # [B, Hidden * Dim]
        return self.energy_head(graph_repr).squeeze(-1)  # [B]

    def forward(self, z, pos, batch, edge_index):
        with torch.enable_grad():
            pos.requires_grad_(True)

            h_scalar = self.atom_embedding(z)  # [N, Hidden]
            h = F.pad(h_scalar.unsqueeze(-1), (0, self.algebra.dim - 1))  # [N, H, D]

            h = self._run_layers(h, pos, edge_index)

            self._last_features = h.detach()

            h = self.output_norm(self.blade_selector(h))
            h_flat = h.reshape(h.size(0), -1)  # [N, Hidden * Dim]

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
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for layer in self.layers:
            if hasattr(layer, 'multi_rotor'):
                loss = loss + layer.multi_rotor.sparsity_loss()
            if hasattr(layer, 'dynamic_rotor_gen'):
                for p in layer.dynamic_rotor_gen.net.parameters():
                    loss = loss + torch.norm(p, p=1) * 0.01
        return loss
