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


class SphericalGraphConv(CliffordModule):
    """Message passing layer on spherical lat/lon grid.

    Accounts for latitude-dependent area weighting and handles
    periodic boundary conditions in longitude via the graph structure.
    Uses RotorLayer with decomposition for geometric mixing.
    """

    def __init__(self, algebra, hidden_dim, num_rotors=8,
                 use_decomposition=False, decomp_k=10, use_rotor_backend=False):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim

        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.lin_node = CliffordLinear(algebra, hidden_dim, hidden_dim, backend=backend)
        self.lin_msg = CliffordLinear(algebra, hidden_dim, hidden_dim, backend=backend)

        self.norm = CliffordLayerNorm(algebra, hidden_dim)
        self.act = GeometricGELU(algebra, channels=hidden_dim)

        self.multi_rotor = MultiRotorLayer(
            algebra, hidden_dim, num_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # Invariant message network
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * algebra.num_grades, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Linear(hidden_dim, algebra.dim)

    def forward(self, h, edge_index, lat_weights=None):
        """Message passing on spherical graph.

        Args:
            h: Node features [N, Hidden, Dim]
            edge_index: Graph connectivity [2, E]
            lat_weights: Optional latitude-dependent weights [H]

        Returns:
            Updated features [N, Hidden, Dim]
        """
        row, col = edge_index

        # Transform, normalize, activate
        h_transformed = self.lin_node(h)
        h_transformed = self.norm(h_transformed)
        h_transformed = self.act(h_transformed)

        h_msg = self.lin_msg(h)[col]

        # Multi-rotor geometric mixing
        phi = self.multi_rotor(h_msg)

        # Extract invariants and compute messages
        inv = self.algebra.get_grade_norms(phi)
        inv_flat = inv.view(inv.size(0), -1)
        msg = self.msg_mlp(inv_flat)

        # Aggregate messages
        out_msg = torch.zeros_like(h)
        gate_values = self.gate(msg).unsqueeze(1)
        out_msg.index_add_(0, row, msg.unsqueeze(-1) * gate_values)

        return h_transformed + out_msg


class TemporalRotorLayer(CliffordModule):
    """Learnable spacetime rotors for temporal evolution.

    Each temporal step applies a separate rotor transformation,
    modeling the evolution of weather state through time.
    Autoregressive: state_t -> state_{t+dt} -> state_{t+2dt} -> ...
    """

    def __init__(self, algebra, channels, num_steps=12,
                 use_decomposition=False, decomp_k=10):
        super().__init__(algebra)
        self.num_steps = num_steps

        self.step_rotors = nn.ModuleList([
            RotorLayer(algebra, channels,
                       use_decomposition=use_decomposition, decomp_k=decomp_k)
            for _ in range(num_steps)
        ])

        # Learnable temporal mixing per step
        self.step_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1))
            for _ in range(num_steps)
        ])

    def forward(self, x, num_steps=None):
        """Apply temporal rotor evolution.

        Args:
            x: Input state features [B*N, Hidden, Dim]
            num_steps: Number of steps to evolve (default: all)

        Returns:
            Evolved state features [B*N, Hidden, Dim]
        """
        steps = num_steps or self.num_steps
        steps = min(steps, self.num_steps)

        for i in range(steps):
            residual = x
            x = self.step_rotors[i](x)
            x = residual + self.step_scales[i] * (x - residual)

        return x


class WeatherGBN(CliffordModule):
    """Weather forecasting model using spacetime geometric algebra.

    Architecture:
        1. Input embedding: variables -> multivector features
        2. Spatial encoding: SphericalGraphConv layers
        3. Multi-scale extraction: MultiRotorLayer
        4. Temporal evolution: TemporalRotorLayer
        5. Output decoder: grade projections -> variables

    Uses Cl(2,1) spacetime algebra for causal temporal structure.
    """

    def __init__(self, algebra, num_variables=2, spatial_hidden_dim=64,
                 num_spatial_layers=6, num_temporal_steps=12,
                 num_rotors=16, use_decomposition=False, decomp_k=10,
                 use_rotor_backend=False):
        super().__init__(algebra)
        self.num_variables = num_variables
        self.spatial_hidden = spatial_hidden_dim
        self._last_features = None

        # Learnable pole embeddings (north + south virtual singularity nodes)
        self.north_pole_embed = nn.Parameter(
            torch.randn(1, spatial_hidden_dim, algebra.dim) * 0.02
        )
        self.south_pole_embed = nn.Parameter(
            torch.randn(1, spatial_hidden_dim, algebra.dim) * 0.02
        )

        # Input embedding: variable values -> multivector features
        self.input_embed = nn.Sequential(
            nn.Linear(num_variables, spatial_hidden_dim),
            nn.SiLU(),
            nn.Linear(spatial_hidden_dim, spatial_hidden_dim * algebra.dim)
        )

        # Spatial encoding layers
        self.spatial_layers = nn.ModuleList([
            SphericalGraphConv(
                algebra, spatial_hidden_dim, num_rotors=num_rotors,
                use_decomposition=use_decomposition, decomp_k=decomp_k,
                use_rotor_backend=use_rotor_backend
            )
            for _ in range(num_spatial_layers)
        ])

        # Multi-scale feature extraction
        self.multi_scale = MultiRotorLayer(
            algebra, spatial_hidden_dim, num_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # Temporal evolution
        self.temporal = TemporalRotorLayer(
            algebra, spatial_hidden_dim, num_steps=num_temporal_steps,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # Grade selection before decode
        self.blade_selector = BladeSelector(algebra, channels=spatial_hidden_dim)
        self.output_norm = CliffordLayerNorm(algebra, spatial_hidden_dim)

        # Output decoder
        self.output_decoder = nn.Sequential(
            nn.Linear(spatial_hidden_dim * algebra.dim, spatial_hidden_dim),
            nn.SiLU(),
            nn.Linear(spatial_hidden_dim, num_variables)
        )

    def forward(self, state_t, edge_index, lat_weights=None, num_steps=None):
        """Forecast weather state.

        The spherical graph includes virtual north/south pole nodes that
        close the sphere topology. Pole nodes receive learnable embeddings,
        participate in message passing, and are stripped before output.

        Args:
            state_t: Current state [B, H, W, C] or [B, N, C]
            edge_index: Spherical graph [2, E] (includes pole node edges)
            lat_weights: Latitude area weights [H]
            num_steps: Temporal steps to evolve

        Returns:
            forecast: Predicted state [B, H, W, C] or [B, N, C]
        """
        B = state_t.size(0)
        original_shape = state_t.shape

        # Flatten spatial dims: [B, H, W, C] -> [B, N_grid, C]
        if state_t.dim() == 4:
            H, W = state_t.size(1), state_t.size(2)
            state_flat = state_t.view(B, H * W, -1)
        else:
            state_flat = state_t
            H, W = None, None

        N_grid = state_flat.size(1)  # Grid nodes only (H*W)

        # Embed grid nodes to multivector space: [B, N_grid, C] -> [B*N_grid, Hidden, Dim]
        h_grid = self.input_embed(state_flat.view(B * N_grid, -1))
        h_grid = h_grid.view(B, N_grid, self.spatial_hidden, self.algebra.dim)

        # Append pole node embeddings: [B, N_grid+2, Hidden, Dim]
        pole_north = self.north_pole_embed.expand(B, -1, -1, -1)  # [B, 1, H, D]
        pole_south = self.south_pole_embed.expand(B, -1, -1, -1)  # [B, 1, H, D]
        h = torch.cat([h_grid, pole_north, pole_south], dim=1)    # [B, N_grid+2, H, D]

        N_total = N_grid + 2  # Grid + 2 poles
        h = h.view(B * N_total, self.spatial_hidden, self.algebra.dim)

        # Expand edge_index for batched graph (offset by N_total per batch)
        edge_indices = []
        for b in range(B):
            edge_indices.append(edge_index + b * N_total)
        batched_edges = torch.cat(edge_indices, dim=1).to(state_t.device)

        # Spatial message passing (full graph including pole nodes)
        for layer in self.spatial_layers:
            h = layer(h, batched_edges, lat_weights)

        # Multi-scale extraction
        h = self.multi_scale(h)

        # Temporal evolution
        h = self.temporal(h, num_steps=num_steps)

        # Store features before output heads (all nodes including poles)
        self._last_features = h.detach()

        # Grade selection and normalization before decode
        h = self.blade_selector(h)
        h = self.output_norm(h)

        # Strip pole nodes: keep only grid nodes for decoding
        h = h.view(B, N_total, self.spatial_hidden, self.algebra.dim)
        h = h[:, :N_grid, :, :]  # [B, N_grid, Hidden, Dim]
        h = h.reshape(B * N_grid, self.spatial_hidden, self.algebra.dim)

        # Decode to variable space
        h_flat = h.reshape(B * N_grid, -1)
        forecast_flat = self.output_decoder(h_flat)  # [B*N_grid, C]

        # Reshape back to original spatial layout
        if H is not None:
            forecast = forecast_flat.view(B, H, W, -1)
        else:
            forecast = forecast_flat.view(B, N_grid, -1)

        return forecast

    def get_latent_features(self):
        """Return last intermediate multivector features (after temporal evolution)."""
        return self._last_features

    def total_sparsity_loss(self):
        """Collect sparsity loss from all rotor layers."""
        loss = torch.tensor(0.0)
        for module in self.modules():
            if hasattr(module, 'sparsity_loss') and module is not self:
                loss = loss + module.sparsity_loss()
        return loss
