# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import pytest
import torch

from core.algebra import CliffordAlgebra
from layers import CliffordGraphConv
from layers.adapters.conformal import ConformalEmbedding
from layers.adapters.projective import ProjectiveEmbedding

pytestmark = pytest.mark.unit


class TestExtensions:
    def test_cga_null_property(self):
        """
        Test that points embedded in CGA are null vectors (P * P = 0).
        """
        algebra = CliffordAlgebra(p=4, q=1, device="cpu")
        embed = ConformalEmbedding(algebra, euclidean_dim=3)

        # Random Euclidean points
        x = torch.randn(5, 3, device="cpu")

        # Embed
        P = embed.embed(x)

        # Geometric Product P * P
        # Should be scalar 0
        sq = algebra.geometric_product(P, P)

        # Check norm of the result (should be 0)
        assert torch.allclose(sq, torch.zeros_like(sq), atol=1e-5), f"P^2 should be 0, got {sq[0, :5]}..."

        # Check reconstruction
        x_recon = embed.extract(P)
        assert torch.allclose(x, x_recon, atol=1e-5), "Reconstructed x should match original"

    def test_pga_embed_extract_roundtrip(self):
        """Points embedded in PGA can be extracted back."""
        algebra = CliffordAlgebra(p=3, q=0, r=1, device="cpu")
        embed = ProjectiveEmbedding(algebra, euclidean_dim=3)

        x = torch.randn(5, 3, device="cpu")
        P = embed.embed(x)

        # Should be grade-1 only
        g1_mask = algebra.grade_masks[1]
        assert torch.allclose(P[:, ~g1_mask], torch.zeros_like(P[:, ~g1_mask]))

        # e_0 coefficient should be 1.0
        assert torch.allclose(P[:, embed._idx_e0], torch.ones(5))

        # Roundtrip
        x_recon = embed.extract(P)
        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_pga_direction_has_no_e0(self):
        """Directions (ideal points) have e_0 = 0."""
        algebra = CliffordAlgebra(p=3, q=0, r=1, device="cpu")
        embed = ProjectiveEmbedding(algebra, euclidean_dim=3)

        v = torch.tensor([[1.0, 0.0, 0.0]])
        P = embed.embed_direction(v)
        assert P[0, embed._idx_e0].item() == 0.0

    def test_pga_rotation_via_rotor(self):
        """Rotors in PGA correctly rotate embedded points."""
        import math

        algebra = CliffordAlgebra(p=3, q=0, r=1, device="cpu")
        embed = ProjectiveEmbedding(algebra, euclidean_dim=3)

        # Embed (1, 0, 0)
        x = torch.tensor([[1.0, 0.0, 0.0]])
        P = embed.embed(x)

        # 90-degree rotation in e1-e2 plane: R = exp(-pi/4 * e12)
        B = torch.zeros(1, algebra.dim)
        B[0, 3] = -math.pi / 4  # e12 bivector
        R = algebra.exp(B)
        R_rev = algebra.reverse(R)

        # Sandwich product
        RP = algebra.geometric_product(R, P)
        P_rot = algebra.geometric_product(RP, R_rev)

        # Extract: should be (0, 1, 0)
        x_out = embed.extract(P_rot)
        assert torch.allclose(x_out, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-4)

        # e_0 component preserved (degenerate dimension unaffected by rotor)
        assert abs(P_rot[0, embed._idx_e0].item() - 1.0) < 1e-5

    def test_gnn_layer(self):
        """
        Test Clifford GCN forward pass.
        """
        # Algebra (e.g., 2D)
        algebra = CliffordAlgebra(p=2, q=0, device="cpu")
        gnn = CliffordGraphConv(algebra, in_channels=2, out_channels=4)

        # 3 Nodes, 2 Channels, 2^2=4 Dim
        x = torch.randn(3, 2, 4)

        # Adjacency (3x3) - e.g., line graph 0-1-2
        adj = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        # Normalize (simplified)
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)

        out = gnn(x, adj)

        # Check shape: [3, 4, 4]
        assert out.shape == (3, 4, 4)
        # Check values are not NaN
        assert not torch.isnan(out).any()
