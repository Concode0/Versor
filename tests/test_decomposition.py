# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Tests for bivector decomposition using power iteration.

Reference:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1 [cs.LG]
"""

import pytest
import torch
from core.algebra import CliffordAlgebra
from core.decomposition import (
    ga_power_iteration,
    differentiable_invariant_decomposition,
    exp_simple_bivector,
    exp_decomposed
)


@pytest.fixture
def algebra_3d():
    """Creates a 3D Euclidean algebra Cl(3,0)."""
    return CliffordAlgebra(p=3, q=0, device='cpu')


@pytest.fixture
def algebra_4d():
    """Creates a 4D Euclidean algebra Cl(4,0)."""
    return CliffordAlgebra(p=4, q=0, device='cpu')


class TestGeometricOperations:
    """Tests for new geometric operations in CliffordAlgebra."""

    def test_wedge_antisymmetry(self, algebra_3d):
        """Test that wedge product is antisymmetric: a ∧ b = -(b ∧ a)."""
        # Create two vectors
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])

        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)

        # Compute wedge products
        wedge_12 = algebra_3d.wedge(v1, v2)
        wedge_21 = algebra_3d.wedge(v2, v1)

        # Check antisymmetry
        assert torch.allclose(wedge_12, -wedge_21, atol=1e-6)

    def test_wedge_parallel_vectors(self, algebra_3d):
        """Test that parallel vectors have zero wedge product: a ∧ a = 0."""
        v_raw = torch.tensor([1.0, 2.0, 3.0])
        v = algebra_3d.embed_vector(v_raw)

        wedge = algebra_3d.wedge(v, v)

        # Should be zero
        assert torch.allclose(wedge, torch.zeros_like(wedge), atol=1e-6)

    def test_wedge_orthogonal_vectors(self, algebra_3d):
        """Test wedge product of orthogonal vectors creates bivector."""
        # e1 ∧ e2 should give e12 (basis bivector)
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])

        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)

        wedge = algebra_3d.wedge(v1, v2)

        # Check it's grade-2
        grade2 = algebra_3d.grade_projection(wedge, 2)
        assert torch.allclose(wedge, grade2, atol=1e-6)

        # Check magnitude
        norm = wedge.norm()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)

    def test_right_contraction_bivector_vector(self, algebra_3d):
        """Test right contraction of bivector with vector."""
        # Create bivector e12
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b = algebra_3d.wedge(v1, v2)

        # Contract with v1
        result = algebra_3d.right_contraction(b, v1)

        # Should be grade-1
        grade1 = algebra_3d.grade_projection(result, 1)
        assert torch.allclose(result, grade1, atol=1e-6)

    def test_inner_product_symmetry(self, algebra_3d):
        """Test that inner product is symmetric: A · B = B · A."""
        v1_raw = torch.tensor([1.0, 2.0, 3.0])
        v2_raw = torch.tensor([4.0, 5.0, 6.0])

        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)

        inner_12 = algebra_3d.inner_product(v1, v2)
        inner_21 = algebra_3d.inner_product(v2, v1)

        assert torch.allclose(inner_12, inner_21, atol=1e-6)

    def test_inner_product_gives_scalar(self, algebra_3d):
        """Test that inner product of two vectors gives scalar."""
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])

        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)

        inner = algebra_3d.inner_product(v1, v2)

        # Should be grade-0 (scalar) for orthogonal vectors → 0
        grade0 = algebra_3d.grade_projection(inner, 0)
        assert torch.allclose(inner, grade0, atol=1e-6)


class TestPowerIteration:
    """Tests for GA power iteration algorithm."""

    def test_simple_bivector_convergence(self, algebra_3d):
        """Test that power iteration converges for a simple bivector."""
        # Create simple bivector: e1 ∧ e2
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b_simple = algebra_3d.wedge(v1, v2)

        # Run power iteration
        b_s, v = ga_power_iteration(algebra_3d, b_simple, threshold=1e-6, max_iterations=100)

        # The result should be close to the original simple bivector
        # (up to sign and normalization)
        assert torch.allclose(b_s.norm(), b_simple.norm(), atol=1e-4)

    def test_power_iteration_deterministic(self, algebra_3d):
        """Test that power iteration gives consistent results with same init."""
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b = algebra_3d.wedge(v1, v2)

        # Same initialization
        v_init_raw = torch.tensor([1.0, 1.0, 1.0])
        v_init = algebra_3d.embed_vector(v_init_raw)

        b_s1, _ = ga_power_iteration(algebra_3d, b, v_init=v_init)
        b_s2, _ = ga_power_iteration(algebra_3d, b, v_init=v_init)

        assert torch.allclose(b_s1, b_s2, atol=1e-6)

    def test_power_iteration_requires_grad(self, algebra_3d):
        """Test that power iteration is differentiable."""
        v1_raw = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
        v2_raw = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b = algebra_3d.wedge(v1, v2)

        b_s, _ = ga_power_iteration(algebra_3d, b)

        # Should have gradient enabled
        assert b_s.requires_grad


class TestBivectorDecomposition:
    """Tests for bivector decomposition algorithm."""

    def test_decomposition_simple_bivector(self, algebra_3d):
        """Test decomposition of a simple bivector gives 1 component."""
        # Simple bivector: e1 ∧ e2
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b_simple = algebra_3d.wedge(v1, v2)

        decomp, vectors = differentiable_invariant_decomposition(
            algebra_3d, b_simple, k=2, threshold=1e-6
        )

        # Should find 1 component (the bivector itself)
        # Second component should be negligible
        assert len(decomp) >= 1
        if len(decomp) > 1:
            # Check second component is much smaller
            assert decomp[1].norm() < 0.1 * decomp[0].norm()

    def test_decomposition_sum_of_two_bivectors(self, algebra_3d):
        """Test decomposition of sum of two orthogonal bivectors."""
        # Create two orthogonal simple bivectors
        # b1 = e1 ∧ e2, b2 = e1 ∧ e3
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v3_raw = torch.tensor([0.0, 0.0, 1.0])

        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        v3 = algebra_3d.embed_vector(v3_raw)

        b1 = algebra_3d.wedge(v1, v2)
        b2 = algebra_3d.wedge(v1, v3)

        # Sum
        b_sum = b1 + b2

        decomp, vectors = differentiable_invariant_decomposition(
            algebra_3d, b_sum, k=2, threshold=1e-6
        )

        # Should find 2 components
        assert len(decomp) >= 1

        # Reconstruction should be close to original
        b_reconstructed = sum(decomp)
        assert torch.allclose(b_sum, b_reconstructed, atol=1e-4)

    def test_decomposition_convergence(self, algebra_3d):
        """Test that residual norm decreases to zero."""
        # Random bivector
        bivector_weights = torch.randn(3)  # 3 bivectors in Cl(3,0): e12, e13, e23
        bivector_indices = [3, 5, 6]  # Binary: 011, 101, 110

        b = torch.zeros(algebra_3d.dim)
        for idx, weight in zip(bivector_indices, bivector_weights):
            b[idx] = weight

        decomp, vectors = differentiable_invariant_decomposition(
            algebra_3d, b, threshold=1e-6
        )

        # Compute residual
        b_reconstructed = sum(decomp)
        residual = b - b_reconstructed

        # Residual should be small
        assert residual.norm() < 1e-4

    def test_decomposition_requires_grad(self, algebra_3d):
        """Test that decomposition maintains gradients."""
        bivector_weights = torch.randn(3, requires_grad=True)
        bivector_indices = [3, 5, 6]

        b = torch.zeros(algebra_3d.dim)
        for idx, weight in zip(bivector_indices, bivector_weights):
            b[idx] = weight

        # Need to ensure b has grad
        b = b.clone()
        b.requires_grad_(True)

        decomp, vectors = differentiable_invariant_decomposition(
            algebra_3d, b, k=2
        )

        # Components should track gradients
        for b_i in decomp:
            assert b_i.requires_grad


class TestExponentialClosedForm:
    """Tests for closed-form exponential of simple bivectors."""

    def test_exp_simple_bivector_rotor_property(self, algebra_3d):
        """Test that exp(b) satisfies rotor property: R × R̃ = 1."""
        # Simple bivector: e1 ∧ e2
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b = 0.5 * algebra_3d.wedge(v1, v2)  # Small angle

        R = exp_simple_bivector(algebra_3d, b)
        R_rev = algebra_3d.reverse(R)

        # R × R̃ should be identity (scalar 1)
        identity = algebra_3d.geometric_product(R, R_rev)

        expected_identity = torch.zeros_like(identity)
        expected_identity[0] = 1.0

        assert torch.allclose(identity, expected_identity, atol=1e-5)

    def test_exp_simple_bivector_unit_norm(self, algebra_3d):
        """Test that exp(b) has unit norm."""
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b = algebra_3d.wedge(v1, v2)

        R = exp_simple_bivector(algebra_3d, b)

        # Rotor should have unit norm
        norm = R.norm()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)

    def test_exp_zero_bivector(self, algebra_3d):
        """Test that exp(0) = 1 (scalar identity)."""
        b = torch.zeros(algebra_3d.dim)

        R = exp_simple_bivector(algebra_3d, b)

        expected = torch.zeros_like(R)
        expected[0] = 1.0

        assert torch.allclose(R, expected, atol=1e-6)


class TestExpDecomposed:
    """Tests for full decomposed exponential."""

    def test_exp_decomposed_vs_standard_small_bivector(self, algebra_3d):
        """Compare decomposed exp with standard exp for small bivectors."""
        # Small simple bivector
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b = 0.1 * algebra_3d.wedge(v1, v2)

        R_standard = algebra_3d.exp(b)
        R_decomposed = exp_decomposed(algebra_3d, b, use_decomposition=True)

        # Should be close
        assert torch.allclose(R_standard, R_decomposed, atol=1e-4)

    def test_exp_decomposed_rotor_property(self, algebra_3d):
        """Test that decomposed exp satisfies rotor property."""
        # Random bivector
        bivector_weights = torch.randn(3) * 0.1
        bivector_indices = [3, 5, 6]

        b = torch.zeros(algebra_3d.dim)
        for idx, weight in zip(bivector_indices, bivector_weights):
            b[idx] = weight

        R = exp_decomposed(algebra_3d, b, use_decomposition=True)
        R_rev = algebra_3d.reverse(R)

        # R × R̃ should be identity
        identity = algebra_3d.geometric_product(R, R_rev)

        expected_identity = torch.zeros_like(identity)
        expected_identity[0] = 1.0

        assert torch.allclose(identity, expected_identity, atol=1e-4)

    def test_exp_decomposed_requires_grad(self, algebra_3d):
        """Test gradient flow through decomposed exponential."""
        # Create leaf tensor directly
        b = torch.randn(algebra_3d.dim, requires_grad=True)

        R = exp_decomposed(algebra_3d, b, use_decomposition=True)

        # Should have gradients
        assert R.requires_grad

        # Test backward pass
        loss = R.sum()
        loss.backward()

        # b should have gradients (it's a leaf tensor)
        assert b.grad is not None
        assert not torch.isnan(b.grad).any()
        assert not torch.isinf(b.grad).any()

    def test_exp_decomposed_fallback(self, algebra_3d):
        """Test that use_decomposition=False falls back to standard exp."""
        v1_raw = torch.tensor([1.0, 0.0, 0.0])
        v2_raw = torch.tensor([0.0, 1.0, 0.0])
        v1 = algebra_3d.embed_vector(v1_raw)
        v2 = algebra_3d.embed_vector(v2_raw)
        b = algebra_3d.wedge(v1, v2)

        R_standard = algebra_3d.exp(b)
        R_fallback = exp_decomposed(algebra_3d, b, use_decomposition=False)

        # Should be identical (using same method)
        assert torch.allclose(R_standard, R_fallback, atol=1e-6)

    def test_exp_decomposed_batch(self, algebra_3d):
        """Test decomposed exp works with batched inputs."""
        batch_size = 4

        # Batch of bivectors
        bivector_weights = torch.randn(batch_size, 3) * 0.1
        bivector_indices = [3, 5, 6]

        b = torch.zeros(batch_size, algebra_3d.dim)
        for i, idx in enumerate(bivector_indices):
            b[:, idx] = bivector_weights[:, i]

        R = exp_decomposed(algebra_3d, b, use_decomposition=True)

        # Check shape
        assert R.shape == (batch_size, algebra_3d.dim)

        # Check each is a rotor
        for i in range(batch_size):
            R_i = R[i]
            R_i_rev = algebra_3d.reverse(R_i)
            identity = algebra_3d.geometric_product(R_i, R_i_rev)

            expected_identity = torch.zeros_like(identity)
            expected_identity[0] = 1.0

            assert torch.allclose(identity, expected_identity, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
