"""Tests for degenerate (null) dimensions in Cl(p, q, r).

Covers:
- Null vector squares to 0
- Cayley table zeros for null self-products
- Bivector exp with null components (parabolic branch)
- Adaptive exp on n=4: simple vs non-simple bivectors
- r=0 backward compatibility
- Cache isolation: Cl(2,0,1) != Cl(2,1,0)
- Wedge with null vectors (non-zero)
"""

import math
import pytest
import torch

pytestmark = pytest.mark.unit

from core.algebra import CliffordAlgebra


DEVICE = "cpu"


# == Cl(p,q,r) basics ====================================================

class TestDegenerate:
    """Core properties of degenerate dimensions."""

    def test_cl201_null_vector_squares_to_zero(self):
        """In Cl(2,0,1): e3^2 = 0 (null dimension)."""
        alg = CliffordAlgebra(2, 0, 1, device=DEVICE)
        assert alg.n == 3
        assert alg.dim == 8
        assert alg.r == 1

        # e3 is index 4 = 0b100 (third basis vector)
        e3 = torch.zeros(1, 8, dtype=torch.float64)
        e3[0, 4] = 1.0
        e3_sq = alg.geometric_product(e3, e3)
        # Should be zero (null vector)
        assert e3_sq.abs().max().item() < 1e-10

    def test_cl201_positive_vectors_still_work(self):
        """In Cl(2,0,1): e1^2 = +1, e2^2 = +1 still hold."""
        alg = CliffordAlgebra(2, 0, 1, device=DEVICE)

        # e1 = index 1 (0b001)
        e1 = torch.zeros(1, 8, dtype=torch.float64)
        e1[0, 1] = 1.0
        e1_sq = alg.geometric_product(e1, e1)
        assert abs(e1_sq[0, 0].item() - 1.0) < 1e-10

        # e2 = index 2 (0b010)
        e2 = torch.zeros(1, 8, dtype=torch.float64)
        e2[0, 2] = 1.0
        e2_sq = alg.geometric_product(e2, e2)
        assert abs(e2_sq[0, 0].item() - 1.0) < 1e-10

    def test_cl301_null_vector(self):
        """In Cl(3,0,1): e4^2 = 0."""
        alg = CliffordAlgebra(3, 0, 1, device=DEVICE)
        assert alg.n == 4
        assert alg.dim == 16

        # e4 = index 8 (0b1000)
        e4 = torch.zeros(1, 16, dtype=torch.float64)
        e4[0, 8] = 1.0
        e4_sq = alg.geometric_product(e4, e4)
        assert e4_sq.abs().max().item() < 1e-10

    def test_cayley_zeros_for_null_self_products(self):
        """Cayley table has zeros where null vectors are squared."""
        alg = CliffordAlgebra(2, 0, 1, device=DEVICE)
        # e3 * e3: basis index 4, result index 4 XOR 4 = 0 (scalar)
        # cayley_signs[4, 4] is the raw sign for this product, should be 0
        assert abs(alg.cayley_signs[4, 4].item()) < 1e-10
        # gp_signs[4, 0] = cayley_signs[4, 4^0] = cayley_signs[4, 4] = 0
        assert abs(alg.gp_signs[4, 0].item()) < 1e-10

    def test_bv_sq_scalar_with_null(self):
        """Bivectors involving null dimensions have bv_sq_scalar = 0."""
        alg = CliffordAlgebra(2, 0, 1, device=DEVICE)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        for idx_pos, blade_idx in enumerate(bv_indices.tolist()):
            bits = []
            for bit in range(alg.n):
                if blade_idx & (1 << bit):
                    bits.append(bit)
            if len(bits) == 2:
                a, b = bits
                pq = alg.p + alg.q
                if a >= pq or b >= pq:
                    # Contains null dimension -> bv_sq_scalar should be 0
                    assert abs(alg.bv_sq_scalar[idx_pos].item()) < 1e-10, \
                        f"Bivector e_{a}{b} (blade {blade_idx}) should have bv_sq=0"
                elif a < alg.p and b < alg.p:
                    # Both positive -> -1
                    assert abs(alg.bv_sq_scalar[idx_pos].item() - (-1.0)) < 1e-10

    def test_bivector_exp_null_parabolic(self):
        """Bivector with null component should use parabolic branch: exp(B) ~ 1 + B."""
        alg = CliffordAlgebra(2, 0, 1, device=DEVICE)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        # Find a bivector involving the null dimension (e.g., e13 or e23)
        # e13 = e1 ^ e3 = index 0b101 = 5, e23 = e2 ^ e3 = index 0b110 = 6
        B = torch.zeros(1, 8, dtype=torch.float64)
        B[0, 5] = 0.3  # e13 (involves null e3)

        R = alg.exp(B)
        # Parabolic: exp(B) = 1 + B
        assert abs(R[0, 0].item() - 1.0) < 1e-7, f"Scalar should be ~1, got {R[0,0].item()}"
        assert abs(R[0, 5].item() - 0.3) < 1e-7, f"e13 coeff should be ~0.3, got {R[0,5].item()}"

    def test_wedge_with_null_vectors(self):
        """Wedge product with null vectors should be non-zero."""
        alg = CliffordAlgebra(2, 0, 1, device=DEVICE)

        # e1 ^ e3 should give e13 (non-zero even though e3 is null)
        e1 = torch.zeros(1, 8, dtype=torch.float64)
        e1[0, 1] = 1.0
        e3 = torch.zeros(1, 8, dtype=torch.float64)
        e3[0, 4] = 1.0

        wedge = alg.wedge(e1, e3)
        # e13 = index 5 (0b101)
        assert abs(wedge[0, 5].item()) > 0.1, "e1 ^ e3 should be non-zero"

    def test_r0_backward_compatible(self):
        """Cl(p, q, 0) should behave identically to Cl(p, q)."""
        alg_old = CliffordAlgebra(3, 0, device=DEVICE)
        alg_new = CliffordAlgebra(3, 0, 0, device=DEVICE)

        assert alg_old.n == alg_new.n
        assert alg_old.dim == alg_new.dim
        assert alg_old.r == 0
        assert alg_new.r == 0

        # Cayley tables should match
        assert torch.allclose(alg_old.gp_signs.float(), alg_new.gp_signs.float())
        assert torch.equal(alg_old.cayley_indices, alg_new.cayley_indices)

    def test_cache_isolation(self):
        """Cl(2,0,1) and Cl(2,1,0) must be different algebras."""
        alg_201 = CliffordAlgebra(2, 0, 1, device=DEVICE)
        alg_210 = CliffordAlgebra(2, 1, 0, device=DEVICE)

        # Same total dimension but different metrics
        assert alg_201.n == alg_210.n
        assert alg_201.dim == alg_210.dim

        # gp_signs should differ
        assert not torch.allclose(alg_201.gp_signs.float(), alg_210.gp_signs.float()), \
            "Cl(2,0,1) and Cl(2,1,0) should have different Cayley tables"

    def test_cl201_mixed_bivector_exp(self):
        """Cl(2,0,1) with mixed bivector: e12 (elliptic) + e13 (parabolic)."""
        alg = CliffordAlgebra(2, 0, 1, device=DEVICE)

        # e12 is purely Euclidean, e13 involves null e3
        B = torch.zeros(1, 8, dtype=torch.float64)
        B[0, 3] = 0.5   # e12 (elliptic: bv_sq = -1)
        B[0, 5] = 0.3   # e13 (parabolic: bv_sq = 0)

        R = alg.exp(B)
        # Rotor should be valid (unit norm via reverse product)
        R_rev = alg.reverse(R)
        norm = alg.geometric_product(R, R_rev)
        # For degenerate algebras, norm may not be exactly 1 but should be close
        assert abs(norm[0, 0].item() - 1.0) < 0.1

    def test_validation_rejects_negative_r(self):
        """Constructor should reject r < 0."""
        with pytest.raises(AssertionError):
            CliffordAlgebra(2, 0, -1, device=DEVICE)

    def test_validation_rejects_too_large(self):
        """Constructor should reject p+q+r > 12."""
        with pytest.raises(AssertionError):
            CliffordAlgebra(6, 4, 3, device=DEVICE)  # 13 > 12


# == Adaptive exp for n >= 4 =============================================

class TestAdaptiveExp:
    """Tests for the adaptive exp strategy on n >= 4."""

    def test_simple_bivector_n4_closed_form(self):
        """A single basis bivector in Cl(4,0) should use closed-form path."""
        alg = CliffordAlgebra(4, 0, device=DEVICE)
        B = torch.zeros(1, 16, dtype=torch.float64)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)
        B[0, bv_indices[0].item()] = 0.5

        R = alg.exp(B)
        R_taylor = alg._exp_taylor(B, order=20)
        assert torch.allclose(R, R_taylor, atol=1e-5), \
            f"Simple bivector: max diff {(R - R_taylor).abs().max()}"

    def test_non_simple_bivector_n4_closed_form_approx(self):
        """exp() uses closed-form for non-simple bivectors (training-friendly approx)."""
        alg = CliffordAlgebra(4, 0, device=DEVICE)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        # Create non-simple: e12 + e34 (disjoint planes)
        B = torch.zeros(1, 16, dtype=torch.float64)
        B[0, bv_indices[0].item()] = 0.3  # e12
        B[0, bv_indices[5].item()] = 0.4  # e34

        R = alg.exp(B)
        # Closed-form is approximate for non-simple, but produces a unit rotor
        R_rev = alg.reverse(R)
        RR = alg.geometric_product(R, R_rev)
        assert abs(RR[0, 0].item() - 1.0) < 0.01, \
            f"exp() should produce near-unit rotor, got scalar part {RR[0, 0].item()}"

    def test_non_simple_bivector_n4_decomposed_inference(self):
        """exp_decomposed() is exact for non-simple bivectors at inference."""
        alg = CliffordAlgebra(4, 0, device=DEVICE)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(1, 16, dtype=torch.float64)
        B[0, bv_indices[0].item()] = 0.3  # e12
        B[0, bv_indices[5].item()] = 0.4  # e34

        from core.decomposition import ExpPolicy
        alg.exp_policy = ExpPolicy.EXACT
        with torch.no_grad():
            R = alg.exp(B)
        alg.exp_policy = ExpPolicy.AUTO
        R_taylor = alg._exp_taylor(B, order=20)
        assert torch.allclose(R, R_taylor, atol=1e-3), \
            f"Decomposed exp at inference: max diff {(R - R_taylor).abs().max()}"

    def test_adaptive_unit_rotor(self):
        """Adaptive exp should produce unit rotors in Cl(4,0)."""
        alg = CliffordAlgebra(4, 0, device=DEVICE)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(1, 16, dtype=torch.float64)
        B[0, bv_indices[0].item()] = 0.3
        R = alg.exp(-0.5 * B)
        R_rev = alg.reverse(R)
        norm = alg.geometric_product(R, R_rev)
        assert abs(norm[0, 0].item() - 1.0) < 1e-6

    def test_adaptive_batch(self):
        """Adaptive exp should work with batched inputs."""
        alg = CliffordAlgebra(4, 0, device=DEVICE)
        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        B = torch.zeros(4, 16, dtype=torch.float64)
        for i in range(4):
            B[i, bv_indices[i % len(bv_indices)].item()] = (i + 1) * 0.2
        R = alg.exp(B)
        assert R.shape == (4, 16)

    def test_degenerate_n4_adaptive(self):
        """Cl(3,0,1) n=4: adaptive exp with null dimension."""
        alg = CliffordAlgebra(3, 0, 1, device=DEVICE)
        assert alg.n == 4

        bv_mask = alg.grade_masks[2]
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1)

        # Simple bivector in positive subspace (e12)
        B = torch.zeros(1, 16, dtype=torch.float64)
        B[0, bv_indices[0].item()] = 0.5

        R = alg.exp(B)
        # Should still produce valid result
        assert not torch.isnan(R).any()
        assert not torch.isinf(R).any()
