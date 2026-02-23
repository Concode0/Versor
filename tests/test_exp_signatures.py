"""Tests for bivector exponential across all signature types.

Verifies closed-form exp works correctly for:
- Euclidean Cl(p,0): all bivectors square to -1 (trigonometric)
- Minkowski Cl(p,q): mixed signs (trigonometric + hyperbolic)
- Degenerate cases: zero bivector, near-zero bivector
"""

import math
import pytest
import torch

from core.algebra import CliffordAlgebra


DEVICE = "cpu"


# ── Helpers ────────────────────────────────────────────────────────────

def _make_bivector(algebra, coeffs):
    """Build a multivector with only grade-2 components from a dict/list."""
    bv_mask = algebra.grade_masks[2]
    bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)
    mv = torch.zeros(algebra.dim, device=DEVICE, dtype=torch.float64)
    if isinstance(coeffs, dict):
        for idx, val in coeffs.items():
            mv[idx] = val
    else:
        for i, val in enumerate(coeffs):
            if i < len(bv_indices):
                mv[bv_indices[i].item()] = val
    return mv.unsqueeze(0)


def _rotor_norm_sq(algebra, R):
    """Compute R * R~ (should be scalar 1 for a unit rotor)."""
    R_rev = algebra.reverse(R)
    product = algebra.geometric_product(R, R_rev)
    return product[..., 0]


def _exp_taylor_reference(algebra, B, order=20):
    """High-order Taylor exp as reference (always correct but slow)."""
    return algebra._exp_taylor(B, order=order)


# ── Euclidean signatures ──────────────────────────────────────────────

class TestExpEuclidean:
    """Closed-form exp for Cl(p, 0) — all bivectors square to -1."""

    @pytest.fixture(params=[(2, 0), (3, 0), (4, 0), (6, 0)])
    def algebra(self, request):
        p, q = request.param
        return CliffordAlgebra(p, q, device=DEVICE)

    def test_zero_bivector_gives_identity(self, algebra):
        B = torch.zeros(1, algebra.dim, dtype=torch.float64)
        R = algebra.exp(B)
        assert abs(R[0, 0].item() - 1.0) < 1e-10
        assert R[0, 1:].abs().max().item() < 1e-10

    def test_unit_rotor_norm(self, algebra):
        """exp(B) should always be a unit rotor: R * R~ = 1."""
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(bv_indices) == 0:
            pytest.skip("No bivectors in this algebra")

        B = torch.zeros(1, algebra.dim, dtype=torch.float64)
        B[0, bv_indices[0].item()] = 0.7
        R = algebra.exp(-0.5 * B)
        norm = _rotor_norm_sq(algebra, R)
        assert abs(norm.item() - 1.0) < 1e-8

    def test_matches_taylor(self, algebra):
        """Closed-form should match Taylor series."""
        torch.manual_seed(42)
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(bv_indices) == 0:
            pytest.skip("No bivectors")

        B = torch.zeros(1, algebra.dim, dtype=torch.float64)
        for idx in bv_indices[:3]:
            B[0, idx.item()] = torch.randn(1, dtype=torch.float64).item() * 0.5

        R_closed = algebra.exp(B)
        R_taylor = _exp_taylor_reference(algebra, B)
        assert torch.allclose(R_closed, R_taylor, atol=1e-6)

    def test_cl20_rotation(self):
        """Cl(2,0): exp(-θ/2 e12) rotates e1 → e2 for θ=π/2."""
        alg = CliffordAlgebra(2, 0, device=DEVICE)
        B = torch.zeros(1, 4, dtype=torch.float64)
        B[0, 3] = 1.0  # e12
        theta = math.pi / 2
        R = alg.exp(-0.5 * theta * B)

        val = math.cos(math.pi / 4)
        assert abs(R[0, 0].item() - val) < 1e-7
        assert abs(R[0, 3].item() + val) < 1e-7

    def test_cl30_rotation(self):
        """Cl(3,0): rotation in e12 plane preserves e3."""
        alg = CliffordAlgebra(3, 0, device=DEVICE)
        # B = e12 (index 3 in binary: 0b11)
        B = torch.zeros(1, 8, dtype=torch.float64)
        B[0, 3] = 1.0
        theta = math.pi / 3
        R = alg.exp(-0.5 * theta * B)
        R_rev = alg.reverse(R)

        # Rotate e3
        v = torch.zeros(1, 8, dtype=torch.float64)
        v[0, 4] = 1.0  # e3 = index 4 (0b100)
        v_rot = alg.geometric_product(alg.geometric_product(R, v), R_rev)

        # e3 should be unchanged by rotation in e12 plane
        assert abs(v_rot[0, 4].item() - 1.0) < 1e-7


# ── Minkowski / mixed signatures ─────────────────────────────────────

class TestExpMinkowski:
    """Closed-form exp for Cl(p, q) with q > 0."""

    def test_cl11_hyperbolic(self):
        """Cl(1,1): e12 squares to +1 → hyperbolic rotation (boost)."""
        alg = CliffordAlgebra(1, 1, device=DEVICE)
        # In Cl(1,1): e1²=+1, e2²=-1, so (e12)²= -e1²e2² = -1*(-1) = +1
        # This means exp(φ e12) = cosh(φ) + sinh(φ) e12
        B = torch.zeros(1, 4, dtype=torch.float64)
        B[0, 3] = 1.0  # e12

        phi = 0.5
        R = alg.exp(phi * B)

        expected_scalar = math.cosh(phi)
        expected_bv = math.sinh(phi)
        assert abs(R[0, 0].item() - expected_scalar) < 1e-7
        assert abs(R[0, 3].item() - expected_bv) < 1e-7

    def test_cl21_mixed_bivectors(self):
        """Cl(2,1): has both elliptic and hyperbolic bivectors."""
        alg = CliffordAlgebra(2, 1, device=DEVICE)
        # e1²=+1, e2²=+1, e3²=-1
        # e12 = e1e2: (e12)² = -e1²e2² = -1 → elliptic
        # e13 = e1e3: (e13)² = -e1²e3² = +1 → hyperbolic
        # e23 = e2e3: (e23)² = -e2²e3² = +1 → hyperbolic

        # Test elliptic bivector e12 (index 3 = 0b011)
        B = torch.zeros(1, 8, dtype=torch.float64)
        B[0, 3] = 0.8  # e12
        R = alg.exp(B)
        R_taylor = _exp_taylor_reference(alg, B)
        assert torch.allclose(R, R_taylor, atol=1e-6), \
            f"Cl(2,1) elliptic: max diff {(R - R_taylor).abs().max()}"

        # Test hyperbolic bivector e13 (index 5 = 0b101)
        B2 = torch.zeros(1, 8, dtype=torch.float64)
        B2[0, 5] = 0.8  # e13
        R2 = alg.exp(B2)
        R2_taylor = _exp_taylor_reference(alg, B2)
        assert torch.allclose(R2, R2_taylor, atol=1e-6), \
            f"Cl(2,1) hyperbolic: max diff {(R2 - R2_taylor).abs().max()}"

    def test_cl31_spacetime(self):
        """Cl(3,1) Minkowski spacetime — used by GA-Transformer."""
        alg = CliffordAlgebra(3, 1, device=DEVICE)
        # e1²=e2²=e3²=+1, e4²=-1
        # Spatial rotations (e_ij, i,j<4): elliptic (square to -1)
        # Boosts (e_i4, i<4): hyperbolic (square to +1)

        torch.manual_seed(123)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        # Test each basis bivector individually
        for bv_idx in bv_indices:
            B = torch.zeros(1, alg.dim, dtype=torch.float64)
            B[0, bv_idx.item()] = 0.3
            R = alg.exp(B)
            R_taylor = _exp_taylor_reference(alg, B)
            assert torch.allclose(R, R_taylor, atol=1e-6), \
                f"Cl(3,1) bv_idx={bv_idx.item()}: max diff {(R - R_taylor).abs().max()}"

    def test_cl41_conformal_simple_bivectors(self):
        """Cl(4,1) conformal GA — each basis bivector should match Taylor exactly."""
        alg = CliffordAlgebra(4, 1, device=DEVICE)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        # Test each basis bivector individually (simple = exact)
        for bv_idx in bv_indices:
            B = torch.zeros(1, alg.dim, dtype=torch.float64)
            B[0, bv_idx.item()] = 0.4
            R = alg.exp(B)
            R_taylor = _exp_taylor_reference(alg, B)
            assert torch.allclose(R, R_taylor, atol=1e-6), \
                f"Cl(4,1) bv_idx={bv_idx.item()}: max diff {(R - R_taylor).abs().max()}"

    def test_cl41_conformal_general_bivector(self):
        """Cl(4,1) general bivector — closed form is approximate for non-simple B."""
        alg = CliffordAlgebra(4, 1, device=DEVICE)
        torch.manual_seed(456)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        # General (non-simple) bivector: closed form is an approximation
        B = torch.zeros(1, alg.dim, dtype=torch.float64)
        for idx in bv_indices[:4]:
            B[0, idx.item()] = torch.randn(1, dtype=torch.float64).item() * 0.3
        R = alg.exp(B)
        # Rotor should still be approximately unit norm
        norm = _rotor_norm_sq(alg, R)
        assert abs(norm.item() - 1.0) < 0.5, f"Rotor norm deviates: {norm.item()}"

    def test_cl21_unit_rotor(self):
        """Rotors in Cl(2,1) should satisfy R*R~ = 1."""
        alg = CliffordAlgebra(2, 1, device=DEVICE)
        torch.manual_seed(789)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(1, alg.dim, dtype=torch.float64)
        for idx in bv_indices:
            B[0, idx.item()] = torch.randn(1, dtype=torch.float64).item() * 0.5
        R = alg.exp(-0.5 * B)
        norm = _rotor_norm_sq(alg, R)
        assert abs(norm.item() - 1.0) < 1e-6, f"Rotor norm: {norm.item()}"

    def test_cl11_boost_preserves_interval(self):
        """Cl(1,1) boost should preserve spacetime interval."""
        alg = CliffordAlgebra(1, 1, device=DEVICE)
        # Boost in e12 plane
        B = torch.zeros(1, 4, dtype=torch.float64)
        B[0, 3] = 0.5  # e12 (hyperbolic)
        R = alg.exp(-0.5 * B)
        R_rev = alg.reverse(R)

        # Apply to a timelike vector: t*e1 + x*e2
        v = torch.zeros(1, 4, dtype=torch.float64)
        v[0, 1] = 2.0  # e1 (time)
        v[0, 2] = 1.0  # e2 (space)

        v_boosted = alg.geometric_product(alg.geometric_product(R, v), R_rev)

        # Minkowski interval: t² - x² should be preserved
        # Original: 4 - 1 = 3
        # (note: e1²=+1, e2²=-1 in Cl(1,1))
        interval_orig = v[0, 1].item()**2 - v[0, 2].item()**2
        interval_boost = v_boosted[0, 1].item()**2 - v_boosted[0, 2].item()**2
        assert abs(interval_orig - interval_boost) < 1e-7


# ── Batch and gradient tests ─────────────────────────────────────────

class TestExpBatchGrad:
    """Batch processing and gradient flow."""

    @pytest.fixture(params=[(3, 0), (2, 1), (3, 1)])
    def algebra(self, request):
        p, q = request.param
        return CliffordAlgebra(p, q, device=DEVICE)

    def test_batch_exp(self, algebra):
        """Exp should work on batched inputs."""
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(5, algebra.dim, dtype=torch.float64)
        for i in range(5):
            for idx in bv_indices[:2]:
                B[i, idx.item()] = (i + 1) * 0.1
        R = algebra.exp(B)
        assert R.shape == (5, algebra.dim)

        # Each should be a valid rotor
        for i in range(5):
            norm = _rotor_norm_sq(algebra, R[i:i+1])
            assert abs(norm.item() - 1.0) < 1e-6

    def test_gradient_flow(self, algebra):
        """Gradients should flow through exp."""
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(1, algebra.dim, requires_grad=True)
        # Can't set indices on leaf tensor, so use scatter
        coeffs = torch.randn(len(bv_indices)) * 0.3
        B_full = B + torch.zeros_like(B).scatter(
            -1, bv_indices.unsqueeze(0), coeffs.unsqueeze(0)
        )
        R = algebra.exp(B_full)
        loss = R.sum()
        loss.backward()
        assert B.grad is not None
        assert not torch.isnan(B.grad).any()

    def test_multichannel_exp(self, algebra):
        """Exp should work with [Channels, Dim] input (used by RotorLayer)."""
        channels = 8
        bv_mask = algebra.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(channels, algebra.dim, dtype=torch.float64)
        for c in range(channels):
            for idx in bv_indices[:2]:
                B[c, idx.item()] = torch.randn(1, dtype=torch.float64).item() * 0.4

        R = algebra.exp(-0.5 * B)
        assert R.shape == (channels, algebra.dim)

        R_taylor = _exp_taylor_reference(algebra, -0.5 * B)
        assert torch.allclose(R, R_taylor, atol=1e-5)


# ── Edge cases ────────────────────────────────────────────────────────

class TestExpEdgeCases:
    """Boundary conditions and numerical stability."""

    def test_large_bivector(self):
        """Large bivector norms should still produce valid rotors."""
        alg = CliffordAlgebra(3, 0, device=DEVICE)
        B = torch.zeros(1, 8, dtype=torch.float64)
        B[0, 3] = 5.0  # Large angle
        R = alg.exp(B)
        norm = _rotor_norm_sq(alg, R)
        assert abs(norm.item() - 1.0) < 1e-6

    def test_very_small_bivector(self):
        """Near-zero bivector should give near-identity rotor."""
        alg = CliffordAlgebra(3, 0, device=DEVICE)
        B = torch.zeros(1, 8, dtype=torch.float64)
        B[0, 3] = 1e-10
        R = alg.exp(B)
        assert abs(R[0, 0].item() - 1.0) < 1e-8

    def test_bv_sq_scalar_values(self):
        """Verify precomputed bivector squared scalars."""
        # Cl(2,0): all bivectors square to -1
        alg20 = CliffordAlgebra(2, 0, device=DEVICE)
        assert (alg20.bv_sq_scalar == -1).all()

        # Cl(1,1): e12 squares to +1 (since e1²=+1, e2²=-1, -(+1)(-1)=+1)
        alg11 = CliffordAlgebra(1, 1, device=DEVICE)
        assert alg11.bv_sq_scalar[0].item() == 1.0

        # Cl(3,0): all 3 bivectors square to -1
        alg30 = CliffordAlgebra(3, 0, device=DEVICE)
        assert (alg30.bv_sq_scalar == -1).all()

        # Cl(2,1): e12 → -1, e13 → +1, e23 → +1
        alg21 = CliffordAlgebra(2, 1, device=DEVICE)
        bv_sq = alg21.bv_sq_scalar
        assert bv_sq[0].item() == -1.0  # e12: -(+1)(+1) = -1
        assert bv_sq[1].item() == 1.0   # e13: -(+1)(-1) = +1
        assert bv_sq[2].item() == 1.0   # e23: -(+1)(-1) = +1

    def test_algebra_init_validation(self):
        """Algebra constructor should reject invalid signatures."""
        with pytest.raises(AssertionError):
            CliffordAlgebra(-1, 0, device=DEVICE)
        with pytest.raises(AssertionError):
            CliffordAlgebra(0, -1, device=DEVICE)
        with pytest.raises(AssertionError):
            CliffordAlgebra(7, 6, device=DEVICE)  # p+q=13 > 12
