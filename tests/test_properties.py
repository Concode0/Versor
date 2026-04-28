# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import pytest
from core.algebra import CliffordAlgebra
from layers import CliffordLayerNorm

pytestmark = pytest.mark.unit


class TestGeometricProperties:
    def test_exp_log_identity(self, algebra_3d):
        """
        Test if exp(B) is a rotation and R * R_rev ~= 1.
        """
        # Create a random bivector
        # Bivector indices in 3D: 3 (e1e2), 5 (e1e3), 6 (e2e3)
        B = torch.zeros(1, algebra_3d.dim)
        B[0, 3] = 0.5
        B[0, 5] = -0.2
        B[0, 6] = 0.8

        # R = exp(-B/2)
        R = algebra_3d.exp(-0.5 * B)
        R_rev = algebra_3d.reverse(R)

        # Check Isometry: R * R_rev should be scalar 1
        prod = algebra_3d.geometric_product(R, R_rev)

        # Expected: [1, 0, 0, ...]
        expected = torch.zeros_like(prod)
        expected[0, 0] = 1.0

        assert torch.allclose(prod, expected, atol=1e-5), f"R * R~ should be 1, got {prod}"

    def test_normalization_layer(self, algebra_3d):
        """
        Test if CliffordLayerNorm correctly normalizes magnitudes.
        """
        layer = CliffordLayerNorm(algebra_3d, channels=1)

        # Random input with large magnitude
        x = torch.randn(2, 1, algebra_3d.dim) * 10.0

        out = layer(x)

        # Norm of output should be close to 1 (since weights are initialized to 1)
        norms = out.norm(dim=-1)
        expected_norms = torch.ones_like(norms)

        # Note: Bias in our implementation affects the scalar part,
        # but initialized to 0. So norm should be 1.

        assert torch.allclose(norms, expected_norms, atol=1e-5), f"Norms should be 1, got {norms}"

    def test_scaling_squaring_stability(self, algebra_3d):
        """
        Test exponential of a very large bivector.
        Naive Taylor would fail or be inaccurate.
        """
        # Large angle rotation
        B = torch.zeros(1, algebra_3d.dim)
        B[0, 3] = 100.0  # Huge angle

        # R = exp(-B/2)
        R = algebra_3d.exp(-0.5 * B)

        # Norm of a rotor should always be 1
        norm = R.norm(dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-4), (
            f"Rotor norm should be 1 even for large inputs, got {norm}"
        )
