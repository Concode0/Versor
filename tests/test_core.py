# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import math
import pytest
import torch
from core.algebra import CliffordAlgebra

pytestmark = pytest.mark.unit


class TestCliffordAlgebra:
    def test_euclidean_2d_cayley(self):
        # E2: e1*e1=1, e2*e2=1
        # Basis: 1, e1, e2, e12
        # e1*e2 = e12
        # e2*e1 = -e12
        alg = CliffordAlgebra(p=2, q=0, device='cpu')

        # indices: 0(1), 1(e1), 2(e2), 3(e12)
        # 1 * 2 (e1 * e2) -> 3 (e12), sign +
        target_idx = alg.cayley_indices[1, 2]
        sign = alg.cayley_signs[1, 2]
        assert target_idx.item() == 3
        assert sign.item() == 1.0

        # 2 * 1 (e2 * e1) -> 3 (e12), sign -
        target_idx = alg.cayley_indices[2, 1]
        sign = alg.cayley_signs[2, 1]
        assert target_idx.item() == 3
        assert sign.item() == -1.0

    def test_geometric_product_simple(self):
        # E2
        alg = CliffordAlgebra(p=2, q=0, device='cpu')

        # A = 2*e1
        A = torch.zeros(1, 4)
        A[0, 1] = 2.0

        # B = 3*e2
        B = torch.zeros(1, 4)
        B[0, 2] = 3.0

        # C = A*B = 6*e12
        C = alg.geometric_product(A, B)
        assert C[0, 3].item() == 6.0

    def test_rotor_exp(self):
        # Rotation in 2D plane by 90 degrees
        # R = exp(-theta/2 * e12)
        # theta = pi/2 -> -pi/4 * e12
        alg = CliffordAlgebra(p=2, q=0, device='cpu')

        B = torch.zeros(1, 4)
        B[0, 3] = 1.0  # unit bivector

        theta = math.pi / 2
        R = alg.exp(-0.5 * theta * B)

        # R = cos(pi/4) - sin(pi/4)e12
        val = math.cos(math.pi / 4)
        assert abs(R[0, 0].item() - val) < 1e-5
        assert abs(R[0, 3].item() - (-val)) < 1e-5

        # Rotate e1 -> e2
        # v = e1
        v = torch.zeros(1, 4)
        v[0, 1] = 1.0

        R_rev = alg.reverse(R)

        # Rv
        Rv = alg.geometric_product(R, v)
        # v' = RvR~
        v_prime = alg.geometric_product(Rv, R_rev)

        # Expected e2
        assert abs(v_prime[0, 1].item() - 0.0) < 1e-5
        assert abs(v_prime[0, 2].item() - 1.0) < 1e-5

    def test_grade_involution_scalar_unchanged(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        mv = torch.zeros(1, 8)
        mv[0, 0] = 3.14
        result = alg.grade_involution(mv)
        assert torch.allclose(result, mv)

    def test_grade_involution_vector_negated(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = torch.zeros(1, 8)
        v[0, 1] = 2.0  # e1
        result = alg.grade_involution(v)
        assert torch.allclose(result[0, 1], torch.tensor(-2.0))
        assert result[0, 0].item() == 0.0

    def test_grade_involution_bivector_unchanged(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        bv = torch.zeros(1, 8)
        bv[0, 3] = 1.5  # e12
        result = alg.grade_involution(bv)
        assert torch.allclose(result[0, 3], torch.tensor(1.5))

    def test_grade_involution_double_identity(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        mv = torch.randn(5, 8)
        result = alg.grade_involution(alg.grade_involution(mv))
        assert torch.allclose(result, mv, atol=1e-6)

    def test_grade_involution_automorphism(self):
        """hat(AB) = hat(A) hat(B)."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        A = torch.randn(3, 8)
        B = torch.randn(3, 8)
        AB = alg.geometric_product(A, B)
        lhs = alg.grade_involution(AB)
        rhs = alg.geometric_product(
            alg.grade_involution(A), alg.grade_involution(B))
        assert torch.allclose(lhs, rhs, atol=1e-5)

    def test_clifford_conjugation_sign_pattern(self):
        """Grades 0,1,2,3 get signs +1, -1, -1, +1."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        for k, expected_sign in enumerate([1, -1, -1, 1]):
            mv = torch.zeros(1, 8)
            mask = alg.grade_masks[k]
            mv[0, mask] = 1.0
            result = alg.clifford_conjugation(mv)
            assert torch.allclose(
                result[0, mask],
                expected_sign * torch.ones(mask.sum()),
                atol=1e-6), f"Failed at grade {k}"

    def test_clifford_conjugation_equals_reverse_then_involution(self):
        """bar(x) = hat(~x)."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        mv = torch.randn(4, 8)
        conj = alg.clifford_conjugation(mv)
        rev_then_inv = alg.grade_involution(alg.reverse(mv))
        assert torch.allclose(conj, rev_then_inv, atol=1e-6)

    def test_clifford_conjugation_anti_automorphism(self):
        """bar(AB) = bar(B) bar(A)."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        A = torch.randn(3, 8)
        B = torch.randn(3, 8)
        AB = alg.geometric_product(A, B)
        lhs = alg.clifford_conjugation(AB)
        rhs = alg.geometric_product(
            alg.clifford_conjugation(B), alg.clifford_conjugation(A))
        assert torch.allclose(lhs, rhs, atol=1e-5)

    def test_norm_sq_scalar(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        mv = torch.zeros(1, 8)
        mv[0, 0] = 3.0
        assert torch.allclose(alg.norm_sq(mv), torch.tensor([[9.0]]), atol=1e-6)

    def test_norm_sq_vector_euclidean(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = alg.embed_vector(torch.tensor([[1.0, 2.0, 3.0]]))
        assert torch.allclose(alg.norm_sq(v), torch.tensor([[14.0]]), atol=1e-5)

    def test_norm_sq_null_vector_minkowski(self):
        """In Cl(2,1), e1+e3 is null: norm_sq = 1 + (-1) = 0."""
        alg = CliffordAlgebra(p=2, q=1, device='cpu')
        v = torch.zeros(1, alg.dim)
        v[0, 1] = 1.0  # e1 (positive)
        v[0, 4] = 1.0  # e3 (negative, index 2^2=4)
        assert torch.allclose(alg.norm_sq(v), torch.tensor([[0.0]]), atol=1e-6)

    def test_norm_sq_matches_full_gp(self):
        """norm_sq matches <x * ~x>_0 via full geometric product."""
        for p, q in [(3, 0), (2, 1), (3, 1), (2, 0)]:
            alg = CliffordAlgebra(p, q, device='cpu')
            mv = torch.randn(5, alg.dim)
            fast = alg.norm_sq(mv)
            rev = alg.reverse(mv)
            full = alg.geometric_product(mv, rev)
            ref = full[..., 0:1]
            assert torch.allclose(fast, ref, atol=1e-5), f"Failed for Cl({p},{q})"

    def test_norm_sq_rotor_unit(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        B = torch.zeros(1, 8)
        B[0, 3] = 0.5  # e12 bivector
        R = alg.exp(B)
        assert torch.allclose(alg.norm_sq(R), torch.tensor([[1.0]]), atol=1e-5)

    def test_left_contraction_vector_bivector(self):
        """e1 _| e12 = e2, e2 _| e12 = -e1 in Cl(3,0)."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')

        e1 = torch.zeros(1, 8); e1[0, 1] = 1.0
        e2 = torch.zeros(1, 8); e2[0, 2] = 1.0
        e12 = torch.zeros(1, 8); e12[0, 3] = 1.0

        r1 = alg.left_contraction(e1, e12)
        assert abs(r1[0, 2].item() - 1.0) < 1e-6
        r1[0, 2] = 0.0
        assert torch.allclose(r1, torch.zeros_like(r1), atol=1e-6)

        r2 = alg.left_contraction(e2, e12)
        assert abs(r2[0, 1].item() - (-1.0)) < 1e-6
        r2[0, 1] = 0.0
        assert torch.allclose(r2, torch.zeros_like(r2), atol=1e-6)

    def test_left_contraction_scalar(self):
        """Scalar _| X = scalar * X."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        s = torch.zeros(1, 8); s[0, 0] = 3.0
        X = torch.randn(1, 8)
        result = alg.left_contraction(s, X)
        assert torch.allclose(result, 3.0 * X, atol=1e-5)

    def test_left_contraction_higher_grade_vanishes(self):
        """e12 _| e1 = 0 (grade(A) > grade(B) -> zero)."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        e12 = torch.zeros(1, 8); e12[0, 3] = 1.0
        e1 = torch.zeros(1, 8); e1[0, 1] = 1.0
        result = alg.left_contraction(e12, e1)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_left_contraction_vector_vector_gives_scalar(self):
        """v _| w = v . w (inner product) for vectors."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = alg.embed_vector(torch.tensor([[1.0, 2.0, 3.0]]))
        w = alg.embed_vector(torch.tensor([[4.0, 5.0, 6.0]]))
        result = alg.left_contraction(v, w)
        # 1*4 + 2*5 + 3*6 = 32
        assert abs(result[0, 0].item() - 32.0) < 1e-5
        assert torch.allclose(result[0, 1:], torch.zeros(7), atol=1e-5)

    def test_dual_grade_shift(self):
        """Dual maps grade-k to grade-(n-k)."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        for k in range(4):
            mv = torch.zeros(1, 8)
            mask_k = alg.grade_masks[k]
            mv[0, mask_k] = torch.randn(mask_k.sum())
            result = alg.dual(mv)
            target_mask = alg.grade_masks[3 - k]
            target_energy = result[0, target_mask].abs().sum()
            total_energy = result[0].abs().sum()
            if total_energy > 1e-8:
                assert target_energy / total_energy > 0.99, \
                    f"Dual of grade-{k} not concentrated in grade-{3 - k}"

    def test_dual_scalar_to_pseudoscalar(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        s = torch.zeros(1, 8); s[0, 0] = 1.0
        result = alg.dual(s)
        g3_mask = alg.grade_masks[3]
        assert result[0, g3_mask].abs().sum() > 0.5
        assert result[0, ~g3_mask].abs().sum() < 1e-6

    def test_embed_vector_shape(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = torch.randn(5, 3)
        mv = alg.embed_vector(v)
        assert mv.shape == (5, 8)

    def test_embed_vector_only_grade1(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = torch.randn(3, 3)
        mv = alg.embed_vector(v)
        g1_mask = alg.grade_masks[1]
        assert torch.allclose(mv[:, g1_mask], v, atol=1e-6)
        assert torch.allclose(mv[:, ~g1_mask], torch.zeros_like(mv[:, ~g1_mask]))

    def test_embed_vector_norm_consistency(self):
        """norm_sq(embed(v)) = ||v||^2 in Euclidean signature."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = torch.randn(5, 3)
        mv = alg.embed_vector(v)
        ns = alg.norm_sq(mv)
        expected = (v * v).sum(dim=-1, keepdim=True)
        assert torch.allclose(ns, expected, atol=1e-5)

    def test_reflect_parallel_component_flips(self):
        """Reflecting e1 through e1-normal plane gives -e1."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        e1 = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        n = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        result = alg.reflect(e1, n)
        expected = alg.embed_vector(torch.tensor([[-1.0, 0.0, 0.0]]))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_reflect_perpendicular_component_stays(self):
        """Reflecting e2 through e1-normal plane gives e2."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        e2 = alg.embed_vector(torch.tensor([[0.0, 1.0, 0.0]]))
        n = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        result = alg.reflect(e2, n)
        expected = alg.embed_vector(torch.tensor([[0.0, 1.0, 0.0]]))
        assert torch.allclose(result, expected, atol=1e-5)

    def test_reflect_preserves_norm(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = alg.embed_vector(torch.randn(1, 3))
        n = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        result = alg.reflect(v, n)
        assert torch.allclose(alg.norm_sq(v), alg.norm_sq(result), atol=1e-4)

    def test_reflect_double_is_identity(self):
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = alg.embed_vector(torch.tensor([[1.0, 2.0, 3.0]]))
        n = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        r1 = alg.reflect(v, n)
        r2 = alg.reflect(r1, n)
        assert torch.allclose(r2, v, atol=1e-5)

    def test_two_reflections_equal_rotation(self):
        """Two reflections through orthogonal planes = 180 degrees rotation."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        v = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        n1 = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        n2 = alg.embed_vector(torch.tensor([[0.0, 1.0, 0.0]]))
        r1 = alg.reflect(v, n1)
        r2 = alg.reflect(r1, n2)
        expected = alg.embed_vector(torch.tensor([[-1.0, 0.0, 0.0]]))
        assert torch.allclose(r2, expected, atol=1e-5)

    def test_reflect_minkowski(self):
        """Reflection in Cl(2,1)."""
        alg = CliffordAlgebra(p=2, q=1, device='cpu')
        v = alg.embed_vector(torch.tensor([[1.0, 1.0, 1.0]]))
        n = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        result = alg.reflect(v, n)
        g1_mask = alg.grade_masks[1]
        vec = result[0, g1_mask]
        assert abs(vec[0].item() - (-1.0)) < 1e-5
        assert abs(vec[1].item() - 1.0) < 1e-5
        assert abs(vec[2].item() - 1.0) < 1e-5

    def test_versor_product_rotor_equals_sandwich(self):
        """For a rotor, versor_product = RxR~."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        B = torch.zeros(1, 8); B[0, 3] = 0.7
        R = alg.exp(B)
        x = alg.embed_vector(torch.tensor([[1.0, 2.0, 3.0]]))
        vp = alg.versor_product(R, x)
        R_rev = alg.reverse(R)
        sandwich = alg.geometric_product(
            alg.geometric_product(R, x), R_rev)
        assert torch.allclose(vp, sandwich, atol=1e-5)

    def test_versor_product_vector_equals_reflection(self):
        """For a vector, versor_product = reflection."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        n = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        x = alg.embed_vector(torch.tensor([[1.0, 2.0, 3.0]]))
        vp = alg.versor_product(n, x)
        refl = alg.reflect(x, n)
        assert torch.allclose(vp, refl, atol=1e-5)

    def test_versor_product_preserves_grade(self):
        """Versor product of a vector remains a vector."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        B = torch.zeros(1, 8); B[0, 5] = 0.3
        R = alg.exp(B)
        v = alg.embed_vector(torch.tensor([[1.0, 2.0, 0.0]]))
        result = alg.versor_product(R, v)
        g1_mask = alg.grade_masks[1]
        other_energy = result[0, ~g1_mask].abs().sum()
        assert other_energy < 1e-5

    def test_versor_product_composition(self):
        """V2(V1 x V1^-1)V2^-1 = (V2 V1) x (V2 V1)^-1."""
        alg = CliffordAlgebra(p=3, q=0, device='cpu')
        B1 = torch.zeros(1, 8); B1[0, 3] = 0.3
        B2 = torch.zeros(1, 8); B2[0, 5] = 0.5
        R1 = alg.exp(B1)
        R2 = alg.exp(B2)
        x = alg.embed_vector(torch.tensor([[1.0, 0.0, 0.0]]))
        step1 = alg.versor_product(R1, x)
        step2 = alg.versor_product(R2, step1)
        R12 = alg.geometric_product(R2, R1)
        composed = alg.versor_product(R12, x)
        assert torch.allclose(step2, composed, atol=1e-5)
