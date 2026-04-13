# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import pytest
from core.algebra import CliffordAlgebra
from core.decomposition import ExpPolicy
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

    def test_rotor_layer_exact_policy(self, algebra_3d):
        """Test RotorLayer with EXACT exp policy."""
        from core.decomposition import ExpPolicy
        algebra_3d.exp_policy = ExpPolicy.EXACT
        x = torch.randn(4, 5, 8)
        layer = RotorLayer(algebra_3d, 5)
        y = layer(x)
        algebra_3d.exp_policy = ExpPolicy.AUTO

        # Check output shape
        assert y.shape == (4, 5, 8)

        # Check norm preservation (rotor property)
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        assert torch.allclose(x_norm, y_norm, atol=1e-4)

    def test_rotor_layer_policy_vs_standard(self, algebra_3d):
        """Compare RotorLayer with FAST vs EXACT policy."""
        from core.decomposition import ExpPolicy

        layer_a = RotorLayer(algebra_3d, 3)
        layer_b = RotorLayer(algebra_3d, 3)
        layer_b.grade_weights.data = layer_a.grade_weights.data.clone()

        x = torch.randn(2, 3, 8)

        algebra_3d.exp_policy = ExpPolicy.FAST
        y_fast = layer_a(x)

        algebra_3d.exp_policy = ExpPolicy.EXACT
        y_exact = layer_b(x)

        algebra_3d.exp_policy = ExpPolicy.AUTO

        # For n=3, all bivectors are simple so results should match
        assert torch.allclose(y_fast, y_exact, atol=1e-3)

    def test_rotor_layer_backward_exact(self, algebra_3d):
        """Test gradient flow through RotorLayer with EXACT policy."""
        from core.decomposition import ExpPolicy
        algebra_3d.exp_policy = ExpPolicy.EXACT

        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = RotorLayer(algebra_3d, 3)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.grade_weights.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(layer.grade_weights.grad).any()
        assert not torch.isinf(layer.grade_weights.grad).any()

        algebra_3d.exp_policy = ExpPolicy.AUTO

    def test_multi_rotor_layer_exact_policy(self, algebra_3d):
        """Test MultiRotorLayer with EXACT policy."""
        from core.decomposition import ExpPolicy
        algebra_3d.exp_policy = ExpPolicy.EXACT
        x = torch.randn(4, 5, 8)
        layer = MultiRotorLayer(algebra_3d, 5, num_rotors=4)
        y = layer(x)
        algebra_3d.exp_policy = ExpPolicy.AUTO

        assert y.shape == (4, 5, 8)

    def test_multi_rotor_layer_backward_exact(self, algebra_3d):
        """Test gradient flow through MultiRotorLayer with EXACT policy."""
        from core.decomposition import ExpPolicy
        algebra_3d.exp_policy = ExpPolicy.EXACT

        x = torch.randn(2, 3, 8, requires_grad=True)
        layer = MultiRotorLayer(algebra_3d, 3, num_rotors=4)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.rotor_bivectors.grad is not None
        assert layer.weights.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        assert not torch.isnan(layer.rotor_bivectors.grad).any()
        assert not torch.isinf(layer.rotor_bivectors.grad).any()

        algebra_3d.exp_policy = ExpPolicy.AUTO

    def test_rotor_layer_rotor_property(self, algebra_3d):
        """Verify that exp-produced rotors satisfy R * ~R = 1."""
        from core.decomposition import ExpPolicy
        algebra_3d.exp_policy = ExpPolicy.EXACT

        layer = RotorLayer(algebra_3d, 2)

        B = torch.zeros(layer.channels, algebra_3d.dim)
        indices = layer.grade_indices.unsqueeze(0).expand(layer.channels, -1)
        B.scatter_(1, indices, layer.grade_weights)

        R = algebra_3d.exp(-0.5 * B)
        R_rev = algebra_3d.reverse(R)

        # R * ~R should be identity
        for i in range(layer.channels):
            identity = algebra_3d.geometric_product(R[i:i+1], R_rev[i:i+1])

            expected_identity = torch.zeros_like(identity)
            expected_identity[..., 0] = 1.0

            assert torch.allclose(identity, expected_identity, atol=1e-4)

        algebra_3d.exp_policy = ExpPolicy.AUTO

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

    # --- Multi-rotor action-matrix equivalence ---

    def test_multi_rotor_action_matrix_equivalence(self, algebra_3d):
        """Verify action-matrix sandwich matches two-GP sandwich numerically."""
        K, C, B = 4, 5, 3
        layer = MultiRotorLayer(algebra_3d, C, num_rotors=K)
        x = torch.randn(B, C, algebra_3d.dim)

        V_left, V_right = layer._compute_versors(x.device, x.dtype)

        # Action-matrix path (current implementation)
        y_action = algebra_3d.multi_rotor_sandwich(V_left, x, V_right)

        # Two-GP reference path
        x_expanded = x.unsqueeze(2)                        # [B, C, 1, D]
        VL = V_left.view(1, 1, K, -1)                      # [1, 1, K, D]
        VR = V_right.view(1, 1, K, -1)                     # [1, 1, K, D]
        Vx = algebra_3d.geometric_product(VL, x_expanded)  # [B, C, K, D]
        y_gp = algebra_3d.geometric_product(Vx, VR)        # [B, C, K, D]

        assert torch.allclose(y_action, y_gp, atol=1e-5), \
            f"Max diff: {(y_action - y_gp).abs().max().item():.2e}"


# --- torch.compile smoke tests ---

@pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
class TestCompile:

    def test_compile_rotor_layer(self, algebra_3d):
        """RotorLayer compiles with aot_eager fullgraph."""
        layer = RotorLayer(algebra_3d, channels=4)
        compiled = torch.compile(layer, backend='aot_eager', fullgraph=True)
        x = torch.randn(2, 4, 8)
        y = compiled(x)
        assert y.shape == (2, 4, 8)

    def test_compile_multi_rotor_layer(self, algebra_3d):
        """MultiRotorLayer compiles with aot_eager fullgraph."""
        layer = MultiRotorLayer(algebra_3d, channels=4, num_rotors=3)
        compiled = torch.compile(layer, backend='aot_eager', fullgraph=True)
        x = torch.randn(2, 4, 8)
        y = compiled(x)
        assert y.shape == (2, 4, 8)

    def test_compile_backward(self, algebra_3d):
        """Gradients flow through compiled RotorLayer."""
        layer = RotorLayer(algebra_3d, channels=4)
        compiled = torch.compile(layer, backend='aot_eager')
        x = torch.randn(2, 4, 8, requires_grad=True)
        y = compiled(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.skipif(
        not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_mps_compile_smoke(self):
        """RotorLayer compiles and runs on MPS."""
        alg = CliffordAlgebra(3, 0, device='mps')
        layer = RotorLayer(alg, channels=4).to('mps')
        compiled = torch.compile(layer, backend='aot_eager')
        x = torch.randn(2, 4, 8, device='mps')
        y = compiled(x)
        torch.mps.synchronize()
        assert y.shape == (2, 4, 8)
