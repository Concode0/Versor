# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

import torch
import unittest
import math
from core.algebra import CliffordAlgebra
from layers.normalization import CliffordLayerNorm

class TestGeometricProperties(unittest.TestCase):
    def setUp(self):
        # Setup a standard 3D Euclidean algebra
        self.algebra = CliffordAlgebra(p=3, q=0, device='cpu')

    def test_exp_log_identity(self):
        """
        Test if exp(B) is a rotation and R * R_rev ~= 1.
        """
        # Create a random bivector
        # Bivector indices in 3D: 3 (e1e2), 5 (e1e3), 6 (e2e3)
        B = torch.zeros(1, self.algebra.dim)
        B[0, 3] = 0.5
        B[0, 5] = -0.2
        B[0, 6] = 0.8
        
        # R = exp(-B/2)
        R = self.algebra.exp(-0.5 * B)
        R_rev = self.algebra.reverse(R)
        
        # Check Isometry: R * R_rev should be scalar 1
        prod = self.algebra.geometric_product(R, R_rev)
        
        # Expected: [1, 0, 0, ...]
        expected = torch.zeros_like(prod)
        expected[0, 0] = 1.0
        
        self.assertTrue(torch.allclose(prod, expected, atol=1e-5), 
                        f"R * R~ should be 1, got {prod}")

    def test_normalization_layer(self):
        """
        Test if CliffordLayerNorm correctly normalizes magnitudes.
        """
        layer = CliffordLayerNorm(self.algebra, channels=1)
        
        # Random input with large magnitude
        x = torch.randn(2, 1, self.algebra.dim) * 10.0
        
        out = layer(x)
        
        # Norm of output should be close to 1 (since weights are initialized to 1)
        norms = out.norm(dim=-1)
        expected_norms = torch.ones_like(norms)
        
        # Note: Bias in our implementation affects the scalar part, 
        # but initialized to 0. So norm should be 1.
        
        self.assertTrue(torch.allclose(norms, expected_norms, atol=1e-5),
                        f"Norms should be 1, got {norms}")

    def test_scaling_squaring_stability(self):
        """
        Test exponential of a very large bivector.
        Naive Taylor would fail or be inaccurate.
        """
        # Large angle rotation
        B = torch.zeros(1, self.algebra.dim)
        B[0, 3] = 100.0 # Huge angle
        
        # R = exp(-B/2)
        R = self.algebra.exp(-0.5 * B)
        
        # Norm of a rotor should always be 1
        norm = R.norm(dim=-1)
        self.assertTrue(torch.allclose(norm, torch.tensor([1.0]), atol=1e-4),
                        f"Rotor norm should be 1 even for large inputs, got {norm}")

if __name__ == '__main__':
    unittest.main()
