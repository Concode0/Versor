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
import unittest
from core.algebra import CliffordAlgebra
from layers.linear import CliffordLinear
from layers.rotor import RotorLayer
from layers.multi_rotor import MultiRotorLayer

class TestLayers(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.alg = CliffordAlgebra(3, 0, device=self.device) # 3D Euclidean
        
    def test_linear_shape(self):
        # Batch=4, In=2 channels, Out=3 channels
        # x: [4, 2, 8]
        x = torch.randn(4, 2, 8)
        layer = CliffordLinear(self.alg, 2, 3)
        y = layer(x)
        self.assertEqual(y.shape, (4, 3, 8))
        
    def test_rotor_shape(self):
        # Batch=4, Channels=5
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(self.alg, 5)
        y = layer(x)
        self.assertEqual(y.shape, (4, 5, 8))
        
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
        
        self.assertTrue(torch.allclose(x_norm, y_norm, atol=1e-5))

    def test_multi_rotor_shape(self):
        from layers.multi_rotor import MultiRotorLayer
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(self.alg, 5, num_rotors=4)
        y = layer(x)
        self.assertEqual(y.shape, (4, 5, 8))

    def test_multi_rotor_invariants(self):
        from layers.multi_rotor import MultiRotorLayer
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(self.alg, 5, num_rotors=4)
        inv = layer(x, return_invariants=True)
        # 3D algebra has 4 grades (0, 1, 2, 3)
        self.assertEqual(inv.shape, (4, 5, 4))

    def test_rotor_layer_decomposition(self):
        """Test RotorLayer with decomposition enabled."""
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(self.alg, 5, use_decomposition=True, decomp_k=2)
        y = layer(x)

        # Check output shape
        self.assertEqual(y.shape, (4, 5, 8))

        # Check norm preservation (rotor property)
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        self.assertTrue(torch.allclose(x_norm, y_norm, atol=1e-4))

    def test_rotor_layer_decomposition_vs_standard(self):
        """Compare RotorLayer with and without decomposition."""
        # Use same weights for both
        layer_standard = RotorLayer(self.alg, 3, use_decomposition=False)
        layer_decomposed = RotorLayer(self.alg, 3, use_decomposition=True, decomp_k=2)

        # Copy weights
        layer_decomposed.bivector_weights.data = layer_standard.bivector_weights.data.clone()

        x = torch.randn(2, 3, 8)

        y_standard = layer_standard(x)
        y_decomposed = layer_decomposed(x)

        # Results should be similar (decomposition is approximate)
        # Use relaxed tolerance due to iterative nature
        self.assertTrue(torch.allclose(y_standard, y_decomposed, atol=1e-3))

    def test_rotor_layer_backward_decomposed(self):
        """Test gradient flow through RotorLayer with decomposition."""
        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = RotorLayer(self.alg, 3, use_decomposition=True, decomp_k=2)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.bivector_weights.grad)

        # Check gradients are not nan or inf
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isinf(x.grad).any())
        self.assertFalse(torch.isnan(layer.bivector_weights.grad).any())
        self.assertFalse(torch.isinf(layer.bivector_weights.grad).any())

    def test_multi_rotor_layer_decomposition(self):
        """Test MultiRotorLayer with decomposition enabled."""
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(self.alg, 5, num_rotors=4, use_decomposition=True, decomp_k=2)
        y = layer(x)

        # Check output shape
        self.assertEqual(y.shape, (4, 5, 8))

    def test_multi_rotor_layer_backward_decomposed(self):
        """Test gradient flow through MultiRotorLayer with decomposition."""
        from layers.multi_rotor import MultiRotorLayer

        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = MultiRotorLayer(self.alg, 3, num_rotors=4, use_decomposition=True, decomp_k=2)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.rotor_bivectors.grad)
        self.assertIsNotNone(layer.weights.grad)

        # Check gradients are not nan or inf
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isinf(x.grad).any())
        self.assertFalse(torch.isnan(layer.rotor_bivectors.grad).any())
        self.assertFalse(torch.isinf(layer.rotor_bivectors.grad).any())

    def test_rotor_layer_decomposition_rotor_property(self):
        """Verify that decomposed rotors satisfy R * ~R = 1."""
        layer = RotorLayer(self.alg, 2, use_decomposition=True, decomp_k=2)

        # Construct bivector B
        B = torch.zeros(layer.channels, self.alg.dim)
        indices = layer.bivector_indices.unsqueeze(0).expand(layer.channels, -1)
        B.scatter_(1, indices, layer.bivector_weights)

        # Compute rotor using decomposition
        R = self.alg.exp_decomposed(-0.5 * B, use_decomposition=True, k=2)
        R_rev = self.alg.reverse(R)

        # R * ~R should be identity
        for i in range(layer.channels):
            identity = self.alg.geometric_product(R[i:i+1], R_rev[i:i+1])

            expected_identity = torch.zeros_like(identity)
            expected_identity[..., 0] = 1.0

            self.assertTrue(torch.allclose(identity, expected_identity, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
