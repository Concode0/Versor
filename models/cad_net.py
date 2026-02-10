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
from core.cga import ConformalAlgebra
from layers.base import CliffordModule
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.multi_rotor import MultiRotorLayer
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from functional.activation import GeometricGELU


class ConformalPointNetEncoder(CliffordModule):
    """Encodes point cloud into conformal latent space.

    Embeds points into CGA null cone, processes with rotor layers,
    and aggregates via multi-rotor global pooling.
    """

    def __init__(self, algebra, cga, latent_dim=128, num_layers=3,
                 num_rotors=16, use_decomposition=False, decomp_k=10,
                 use_rotor_backend=False):
        super().__init__(algebra)
        self.cga = cga
        self.latent_dim = latent_dim

        # Point-wise embedding from conformal space
        cga_dim = cga.algebra.dim
        self.point_embed = nn.Sequential(
            nn.Linear(cga_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * algebra.dim)
        )

        # Rotor processing layers
        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'lin': CliffordLinear(algebra, latent_dim, latent_dim, backend=backend),
                'norm': CliffordLayerNorm(algebra, latent_dim),
                'act': GeometricGELU(algebra, channels=latent_dim),
                'rotor': RotorLayer(
                    algebra, latent_dim,
                    use_decomposition=use_decomposition, decomp_k=decomp_k
                ),
            }))

        # BladeSelector to suppress irrelevant grades before pooling
        self.blade_selector = BladeSelector(algebra, channels=latent_dim)

        # Multi-rotor global aggregation
        self.global_rotor = MultiRotorLayer(
            algebra, latent_dim, num_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

    def forward(self, points):
        """Encode point cloud to latent representation.

        Args:
            points: Point cloud [B, N, 3]

        Returns:
            latent: Latent features [B, LatentDim, AlgDim]
        """
        B, N, _ = points.shape

        # Embed into conformal space
        points_flat = points.view(B * N, 3)
        conf_points = self.cga.to_cga(points_flat)  # [B*N, CGA_dim]

        # Point-wise MLP to multivector features
        h = self.point_embed(conf_points)  # [B*N, latent*dim]
        h = h.view(B * N, self.latent_dim, self.algebra.dim)

        # Rotor processing with normalization and activation
        for layer in self.layers:
            h = layer['lin'](h)
            h = layer['norm'](h)
            h = layer['act'](h)
            h = layer['rotor'](h)

        # Grade selection
        h = self.blade_selector(h)

        # Reshape to per-sample
        h = h.view(B, N, self.latent_dim, self.algebra.dim)

        # Global pooling: max + mean
        h_max = h.max(dim=1)[0]   # [B, Latent, Dim]
        h_mean = h.mean(dim=1)    # [B, Latent, Dim]
        h_global = h_max + h_mean

        # Multi-rotor aggregation
        latent = self.global_rotor(h_global)

        return latent


class PointCloudDecoder(CliffordModule):
    """Decodes latent code to reconstructed point cloud.

    Maps latent multivector features through CliffordLinear layers
    and projects back to Euclidean coordinates via CGA.
    """

    def __init__(self, algebra, cga, latent_dim=128, output_points=2048,
                 use_decomposition=False, decomp_k=10, use_rotor_backend=False):
        super().__init__(algebra)
        self.cga = cga
        self.output_points = output_points

        # Expand latent to point features
        backend = 'rotor' if use_rotor_backend else 'traditional'

        alg_dim = algebra.dim
        self.alg_dim = alg_dim

        self.expand = nn.Sequential(
            nn.Linear(latent_dim * alg_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, output_points * latent_dim * alg_dim)
        )

        self.refine = CliffordLinear(algebra, latent_dim, latent_dim, backend=backend)
        self.norm = CliffordLayerNorm(algebra, latent_dim)
        self.act = GeometricGELU(algebra, channels=latent_dim)
        self.rotor = RotorLayer(
            algebra, latent_dim,
            use_decomposition=use_decomposition, decomp_k=decomp_k
        )

        # Project to 3D coordinates
        self.output_proj = nn.Linear(latent_dim * alg_dim, 3)

    def forward(self, latent):
        """Decode latent to point cloud.

        Args:
            latent: Latent features [B, LatentDim, AlgDim]

        Returns:
            points: Reconstructed point cloud [B, M, 3]
        """
        B = latent.size(0)
        latent_dim = latent.size(1)

        # Flatten and expand
        latent_flat = latent.reshape(B, -1)
        h = self.expand(latent_flat)  # [B, M * latent * alg_dim]
        h = h.reshape(B * self.output_points, latent_dim, self.alg_dim)

        # Refine with normalization, activation, and rotor
        h = self.refine(h)
        h = self.norm(h)
        h = self.act(h)
        h = self.rotor(h)

        # Project to 3D
        h_flat = h.reshape(B * self.output_points, -1)
        points = self.output_proj(h_flat)
        points = points.view(B, self.output_points, 3)

        # Normalize to unit sphere
        points = torch.tanh(points)

        return points


class PrimitiveDecoder(CliffordModule):
    """Decodes latent to geometric primitive parameters.

    Predicts primitive type and parameters for each detected primitive.
    """

    def __init__(self, algebra, latent_dim=128, num_primitives=8, num_params=13):
        super().__init__(algebra)
        self.num_primitives = num_primitives
        self.num_params = num_params

        self.type_head = nn.Sequential(
            nn.Linear(latent_dim * algebra.dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, num_primitives * 5)  # 5 primitive types
        )

        self.param_head = nn.Sequential(
            nn.Linear(latent_dim * algebra.dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, num_primitives * num_params)
        )

    def forward(self, latent):
        """Predict primitive parameters.

        Args:
            latent: Latent features [B, LatentDim, AlgDim]

        Returns:
            types: Primitive type logits [B, K, 5]
            params: Primitive parameters [B, K, P]
        """
        B = latent.size(0)
        latent_flat = latent.reshape(B, -1)

        types = self.type_head(latent_flat).view(B, self.num_primitives, 5)
        params = self.param_head(latent_flat).view(B, self.num_primitives, self.num_params)

        return types, params


class CADAutoEncoder(CliffordModule):
    """Conformal geometric algebra autoencoder for CAD model reconstruction.

    Uses Cl(4,1) conformal space for unified primitive representation.
    Encoder embeds point clouds into conformal null cone, processes
    with multi-rotor layers, and decodes to reconstructed geometry.

    Architecture:
        1. ConformalPointNetEncoder: point cloud -> latent CGA representation
        2. PointCloudDecoder: latent -> reconstructed point cloud
        OR PrimitiveDecoder: latent -> primitive parameters
    """

    def __init__(self, algebra, cga, latent_dim=128, num_rotors=16,
                 output_points=2048, use_decomposition=False, decomp_k=10,
                 use_rotor_backend=False, decoder_type='reconstruction'):
        super().__init__(algebra)
        self.cga = cga
        self.output_points = output_points
        self.decoder_type = decoder_type
        self._last_latent = None

        self.encoder = ConformalPointNetEncoder(
            algebra, cga, latent_dim=latent_dim, num_rotors=num_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k,
            use_rotor_backend=use_rotor_backend
        )

        if decoder_type == 'reconstruction':
            self.decoder = PointCloudDecoder(
                algebra, cga, latent_dim=latent_dim, output_points=output_points,
                use_decomposition=use_decomposition, decomp_k=decomp_k,
                use_rotor_backend=use_rotor_backend
            )
        else:
            self.decoder = PrimitiveDecoder(algebra, latent_dim=latent_dim)

    def forward(self, points):
        """Full autoencoder: encode then decode.

        Args:
            points: Input point cloud [B, N, 3]

        Returns:
            If reconstruction: reconstructed points [B, M, 3]
            If primitive: (types [B, K, 5], params [B, K, P])
        """
        latent = self.encoder(points)
        self._last_latent = latent.detach()
        return self.decoder(latent)

    def encode(self, points):
        """Encode point cloud to latent."""
        return self.encoder(points)

    def decode(self, latent):
        """Decode latent to output."""
        return self.decoder(latent)

    def decode_primitives(self, latent):
        """Decode latent to primitive parameters (requires primitive decoder)."""
        if self.decoder_type != 'primitive':
            raise ValueError("Model uses reconstruction decoder, not primitive")
        return self.decoder(latent)

    def get_latent_features(self):
        """Return last latent multivector features (after encoder)."""
        return self._last_latent

    def total_sparsity_loss(self):
        """Collect sparsity loss from all rotor layers."""
        loss = torch.tensor(0.0)
        for module in self.modules():
            if hasattr(module, 'sparsity_loss') and module is not self:
                loss = loss + module.sparsity_loss()
        return loss
