# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import pytest
from core.algebra import CliffordAlgebra
from layers import CliffordLinear
from layers import RotorLayer
from layers import MultiRotorLayer
from layers.primitives.reflection import ReflectionLayer

pytestmark = pytest.mark.unit


class TestLayers:
    def test_linear_shape(self, algebra_3d):
        # Batch=4, In=2 channels, Out=3 channels
        # x: [4, 2, 8]
        x = torch.randn(4, 2, 8)
        layer = CliffordLinear(algebra_3d, 2, 3)
        y = layer(x)
        assert y.shape == (4, 3, 8)

    def test_rotor_shape(self, algebra_3d):
        # Batch=4, Channels=5
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(algebra_3d, 5)
        y = layer(x)
        assert y.shape == (4, 5, 8)

        # Test equivariance (norm preservation for vector part)
        # Vector part is indices 1,2,4 (for 3D basis 1, e1, e2, e3... indices are bitmasks)
        # 1=001, 2=010, 4=100
        vec_indices = [1, 2, 4]

        # Create pure vector input
        x_vec = torch.zeros(4, 5, 8)
        x_vec[..., vec_indices] = torch.randn(4, 5, 3)

        y_vec = layer(x_vec)

        # Norm should be preserved
        x_norm = x_vec.norm(dim=-1)
        y_norm = y_vec.norm(dim=-1)

        # Note: Rotor preserves magnitude of the multivector,
        # and specifically rotates k-vectors to k-vectors.
        # So the norm of the whole multivector should be preserved exactly.

        assert torch.allclose(x_norm, y_norm, atol=1e-5)

    def test_multi_rotor_shape(self, algebra_3d):
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(algebra_3d, 5, num_rotors=4)
        y = layer(x)
        assert y.shape == (4, 5, 8)

    def test_multi_rotor_invariants(self, algebra_3d):
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(algebra_3d, 5, num_rotors=4)
        inv = layer(x, return_invariants=True)
        # 3D algebra has 4 grades (0, 1, 2, 3)
        assert inv.shape == (4, 5, 4)

    def test_rotor_layer_decomposition(self, algebra_3d):
        """Test RotorLayer with decomposition enabled."""
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(algebra_3d, 5, use_decomposition=True, decomp_k=2)
        y = layer(x)

        # Check output shape
        assert y.shape == (4, 5, 8)

        # Check norm preservation (rotor property)
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        assert torch.allclose(x_norm, y_norm, atol=1e-4)

    def test_rotor_layer_decomposition_vs_standard(self, algebra_3d):
        """Compare RotorLayer with and without decomposition."""
        # Use same weights for both
        layer_standard = RotorLayer(algebra_3d, 3, use_decomposition=False)
        layer_decomposed = RotorLayer(algebra_3d, 3, use_decomposition=True, decomp_k=2)

        # Copy weights
        layer_decomposed.bivector_weights.data = layer_standard.bivector_weights.data.clone()

        x = torch.randn(2, 3, 8)

        y_standard = layer_standard(x)
        y_decomposed = layer_decomposed(x)

        # Results should be similar (decomposition is approximate)
        # Use relaxed tolerance due to iterative nature
        assert torch.allclose(y_standard, y_decomposed, atol=1e-3)

    def test_rotor_layer_backward_decomposed(self, algebra_3d):
        """Test gradient flow through RotorLayer with decomposition."""
        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = RotorLayer(algebra_3d, 3, use_decomposition=True, decomp_k=2)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.bivector_weights.grad is not None

        # Check gradients are not nan or inf
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(layer.bivector_weights.grad).any()
        assert not torch.isinf(layer.bivector_weights.grad).any()

    def test_multi_rotor_layer_decomposition(self, algebra_3d):
        """Test MultiRotorLayer with decomposition enabled."""
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(algebra_3d, 5, num_rotors=4, use_decomposition=True, decomp_k=2)
        y = layer(x)

        # Check output shape
        assert y.shape == (4, 5, 8)

    def test_multi_rotor_layer_backward_decomposed(self, algebra_3d):
        """Test gradient flow through MultiRotorLayer with decomposition."""
        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = MultiRotorLayer(algebra_3d, 3, num_rotors=4, use_decomposition=True, decomp_k=2)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.rotor_bivectors.grad is not None
        assert layer.weights.grad is not None

        # Check gradients are not nan or inf
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(layer.rotor_bivectors.grad).any()
        assert not torch.isinf(layer.rotor_bivectors.grad).any()

    def test_rotor_layer_decomposition_rotor_property(self, algebra_3d):
        """Verify that decomposed rotors satisfy R * ~R = 1."""
        layer = RotorLayer(algebra_3d, 2, use_decomposition=True, decomp_k=2)

        # Construct bivector B
        B = torch.zeros(layer.channels, algebra_3d.dim)
        indices = layer.bivector_indices.unsqueeze(0).expand(layer.channels, -1)
        B.scatter_(1, indices, layer.bivector_weights)

        # Compute rotor using decomposition
        R = algebra_3d.exp_decomposed(-0.5 * B, use_decomposition=True, k=2)
        R_rev = algebra_3d.reverse(R)

        # R * ~R should be identity
        for i in range(layer.channels):
            identity = algebra_3d.geometric_product(R[i:i+1], R_rev[i:i+1])

            expected_identity = torch.zeros_like(identity)
            expected_identity[..., 0] = 1.0

            assert torch.allclose(identity, expected_identity, atol=1e-4)

    def test_reflection_shape(self, algebra_3d):
        B, C = 4, 5
        layer = ReflectionLayer(algebra_3d, channels=C)
        x = torch.randn(B, C, 8)
        y = layer(x)
        assert y.shape == (B, C, 8)

    def test_reflection_preserves_norm(self, algebra_3d):
        C = 3
        layer = ReflectionLayer(algebra_3d, channels=C)
        x = torch.randn(2, C, 8)
        y = layer(x)
        x_norms = algebra_3d.norm_sq(x.reshape(-1, 8))
        y_norms = algebra_3d.norm_sq(y.reshape(-1, 8))
        assert torch.allclose(x_norms, y_norms, atol=1e-4)

    def test_reflection_gradient_flow(self, algebra_3d):
        C = 4
        layer = ReflectionLayer(algebra_3d, channels=C)
        x = torch.randn(2, C, 8)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert layer.vector_weights.grad is not None
        assert not torch.all(layer.vector_weights.grad == 0)

    def test_reflection_eval_caching(self, algebra_3d):
        C = 3
        layer = ReflectionLayer(algebra_3d, channels=C)
        layer.eval()
        x = torch.randn(2, C, 8)
        _ = layer(x)
        assert layer._cached_n is not None
        assert layer._cached_n_inv is not None
        layer.train()
        assert layer._cached_n is None
        assert layer._cached_n_inv is None

    def test_reflection_different_signatures(self):
        for p, q in [(2, 0), (3, 0), (2, 1), (3, 1)]:
            alg = CliffordAlgebra(p, q, device='cpu')
            C = 2
            layer = ReflectionLayer(alg, channels=C)
            x = torch.randn(3, C, alg.dim)
            y = layer(x)
            assert y.shape == x.shape

    def test_reflection_sparsity_loss(self, algebra_3d):
        layer = ReflectionLayer(algebra_3d, channels=4)
        loss = layer.sparsity_loss()
        assert loss.dim() == 0
        assert loss.item() > 0
