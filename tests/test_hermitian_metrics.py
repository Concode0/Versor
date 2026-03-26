# Tests for Hermitian metrics in core/metric.py

import torch
import pytest
from core.algebra import CliffordAlgebra

pytestmark = pytest.mark.unit
from core.metric import (
    inner_product, induced_norm, geometric_distance,
    clifford_conjugate, hermitian_inner_product, hermitian_norm,
    hermitian_distance, hermitian_angle, grade_hermitian_norm,
    hermitian_grade_spectrum, signature_trace_form, signature_norm_squared,
    _hermitian_signs,
)


class TestHermitianSigns:
    def test_scalar_positive(self, algebra_3d):
        """Scalar component (grade 0) always has sign +1."""
        signs = _hermitian_signs(algebra_3d)
        assert signs[0] == 1.0

    def test_shape(self, algebra_minkowski):
        signs = _hermitian_signs(algebra_minkowski)
        assert signs.shape == (algebra_minkowski.dim,)

    def test_buffer_registered(self, algebra_minkowski):
        """Hermitian signs are precomputed as a buffer on the algebra."""
        s1 = _hermitian_signs(algebra_minkowski)
        assert hasattr(algebra_minkowski, '_hermitian_signs')
        assert torch.allclose(s1, algebra_minkowski._hermitian_signs)

    def test_values_are_pm1(self, algebra_conformal):
        signs = _hermitian_signs(algebra_conformal)
        assert torch.all(torch.abs(signs) == 1.0)

    def test_matches_geometric_product(self, algebra_minkowski):
        """Verify signed IP matches <bar{A} B>_0 via full geometric product."""
        torch.manual_seed(42)
        A = torch.randn(algebra_minkowski.dim)
        B = torch.randn(algebra_minkowski.dim)

        # Method 1: signed coefficient formula
        signs = _hermitian_signs(algebra_minkowski)
        ip_signed = (signs * A * B).sum()

        # Method 2: full geometric product of conjugate with B
        A_bar = clifford_conjugate(algebra_minkowski, A)
        prod = algebra_minkowski.geometric_product(A_bar.unsqueeze(0), B.unsqueeze(0))
        ip_gp = prod[0, 0]  # scalar part

        assert torch.allclose(ip_signed, ip_gp, atol=1e-5), \
            f"Signed IP {ip_signed.item():.6f} != GP {ip_gp.item():.6f}"

    def test_matches_gp_conformal(self, algebra_conformal):
        """Same verification for Cl(4,1)."""
        torch.manual_seed(123)
        A = torch.randn(algebra_conformal.dim)
        B = torch.randn(algebra_conformal.dim)

        signs = _hermitian_signs(algebra_conformal)
        ip_signed = (signs * A * B).sum()

        A_bar = clifford_conjugate(algebra_conformal, A)
        prod = algebra_conformal.geometric_product(A_bar.unsqueeze(0), B.unsqueeze(0))
        ip_gp = prod[0, 0]

        assert torch.allclose(ip_signed, ip_gp, atol=1e-4), \
            f"Signed IP {ip_signed.item():.6f} != GP {ip_gp.item():.6f}"


class TestCliffordConjugate:
    def test_scalar_unchanged(self, algebra_3d):
        mv = torch.zeros(algebra_3d.dim)
        mv[0] = 3.0
        conj = clifford_conjugate(algebra_3d, mv)
        assert torch.allclose(conj[0], mv[0])

    def test_double_conjugate_is_identity(self, algebra_minkowski):
        mv = torch.randn(algebra_minkowski.dim)
        conj2 = clifford_conjugate(algebra_minkowski, clifford_conjugate(algebra_minkowski, mv))
        assert torch.allclose(conj2, mv, atol=1e-6)

    def test_batch(self, algebra_3d):
        mv = torch.randn(5, algebra_3d.dim)
        conj = clifford_conjugate(algebra_3d, mv)
        assert conj.shape == mv.shape


class TestHermitianInnerProduct:
    def test_matches_bar_gp_euclidean(self, algebra_3d):
        """Hermitian IP should match <bar{A}B>_0 via geometric product."""
        torch.manual_seed(42)
        A = torch.randn(algebra_3d.dim)
        B = torch.randn(algebra_3d.dim)
        ip = hermitian_inner_product(algebra_3d, A, B)
        A_bar = clifford_conjugate(algebra_3d, A)
        prod = algebra_3d.geometric_product(A_bar.unsqueeze(0), B.unsqueeze(0))
        assert torch.allclose(ip.squeeze(), prod[0, 0], atol=1e-5)

    def test_positive_for_pure_scalars(self, algebra_3d):
        """For pure scalars, <bar{s}s>_0 = s^2 >= 0."""
        mv = torch.zeros(algebra_3d.dim)
        mv[0] = 5.0
        ip = hermitian_inner_product(algebra_3d, mv, mv)
        assert ip > 0

    def test_zero_for_zero(self, algebra_minkowski):
        mv = torch.zeros(algebra_minkowski.dim)
        ip = hermitian_inner_product(algebra_minkowski, mv, mv)
        assert torch.allclose(ip, torch.tensor([0.0]))

    def test_symmetry(self, algebra_minkowski):
        a = torch.randn(algebra_minkowski.dim)
        b = torch.randn(algebra_minkowski.dim)
        assert torch.allclose(
            hermitian_inner_product(algebra_minkowski, a, b),
            hermitian_inner_product(algebra_minkowski, b, a),
            atol=1e-6
        )

    def test_linearity(self, algebra_minkowski):
        a = torch.randn(algebra_minkowski.dim)
        b = torch.randn(algebra_minkowski.dim)
        c = torch.randn(algebra_minkowski.dim)
        alpha = 2.5
        lhs = hermitian_inner_product(algebra_minkowski, alpha * a + b, c)
        rhs = alpha * hermitian_inner_product(algebra_minkowski, a, c) + hermitian_inner_product(algebra_minkowski, b, c)
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_has_negative_signs(self, algebra_minkowski):
        """Cl(2,1) should have some negative signs in the Hermitian form."""
        signs = _hermitian_signs(algebra_minkowski)
        has_negative = (signs < 0).any()
        assert has_negative, "Cl(2,1) should have negative signs"


class TestHermitianNorm:
    def test_non_negative(self, algebra_minkowski):
        mv = torch.randn(20, algebra_minkowski.dim)
        norms = hermitian_norm(algebra_minkowski, mv)
        assert (norms >= -1e-6).all()

    def test_zero_for_zero(self, algebra_3d):
        mv = torch.zeros(algebra_3d.dim)
        n = hermitian_norm(algebra_3d, mv)
        assert torch.allclose(n, torch.tensor([0.0]))

    def test_positive_for_nonzero(self, algebra_3d):
        mv = torch.randn(algebra_3d.dim)
        assert hermitian_norm(algebra_3d, mv) > 0

    def test_homogeneity(self, algebra_3d):
        """||alpha*A||_H = |alpha| * ||A||_H for Euclidean."""
        mv = torch.randn(algebra_3d.dim)
        alpha = 3.0
        n1 = hermitian_norm(algebra_3d, alpha * mv)
        n2 = alpha * hermitian_norm(algebra_3d, mv)
        assert torch.allclose(n1, n2, atol=1e-4)


class TestHermitianDistance:
    def test_zero_self_distance(self, algebra_minkowski):
        mv = torch.randn(algebra_minkowski.dim)
        d = hermitian_distance(algebra_minkowski, mv, mv)
        assert torch.allclose(d, torch.tensor([0.0]), atol=1e-6)

    def test_symmetry(self, algebra_minkowski):
        a = torch.randn(algebra_minkowski.dim)
        b = torch.randn(algebra_minkowski.dim)
        d1 = hermitian_distance(algebra_minkowski, a, b)
        d2 = hermitian_distance(algebra_minkowski, b, a)
        assert torch.allclose(d1, d2, atol=1e-5)

    def test_triangle_inequality(self, algebra_3d):
        """Triangle inequality holds for Euclidean (all signs +1)."""
        a = torch.randn(algebra_3d.dim)
        b = torch.randn(algebra_3d.dim)
        c = torch.randn(algebra_3d.dim)
        d_ab = hermitian_distance(algebra_3d, a, b)
        d_bc = hermitian_distance(algebra_3d, b, c)
        d_ac = hermitian_distance(algebra_3d, a, c)
        assert d_ac <= d_ab + d_bc + 1e-5

    def test_positive_for_different(self, algebra_3d):
        a = torch.randn(algebra_3d.dim)
        b = a + torch.randn(algebra_3d.dim) * 0.1
        d = hermitian_distance(algebra_3d, a, b)
        assert d > 0


class TestHermitianAngle:
    def test_zero_angle_same(self, algebra_3d):
        torch.manual_seed(42)
        mv = torch.randn(algebra_3d.dim)
        angle = hermitian_angle(algebra_3d, mv, mv)
        # float32 acos near cos=1 has ~sqrt(2*eps_machine) ~= 5e-4 rad error;
        # use atol=1e-3 to be robust across platforms
        assert torch.allclose(angle, torch.tensor([0.0]), atol=1e-3)

    def test_angle_range(self, algebra_3d):
        a = torch.randn(algebra_3d.dim)
        b = torch.randn(algebra_3d.dim)
        angle = hermitian_angle(algebra_3d, a, b)
        assert angle >= 0 and angle <= torch.pi + 1e-5

    def test_orthogonal(self, algebra_3d):
        a = torch.zeros(algebra_3d.dim)
        b = torch.zeros(algebra_3d.dim)
        a[0] = 1.0  # scalar
        b[1] = 1.0  # e1
        angle = hermitian_angle(algebra_3d, a, b)
        assert torch.allclose(angle, torch.tensor([torch.pi / 2]), atol=1e-5)


class TestGradeHermitianNorm:
    def test_scalar_grade(self, algebra_3d):
        mv = torch.zeros(algebra_3d.dim)
        mv[0] = 5.0
        n = grade_hermitian_norm(algebra_3d, mv, grade=0)
        assert torch.allclose(n, torch.tensor([5.0]), atol=1e-6)

    def test_zero_for_wrong_grade(self, algebra_3d):
        mv = torch.zeros(algebra_3d.dim)
        mv[0] = 5.0  # Only scalar
        n = grade_hermitian_norm(algebra_3d, mv, grade=1)
        assert torch.allclose(n, torch.tensor([0.0]), atol=1e-6)

    def test_grade_decomposition(self, algebra_3d):
        """Grade spectrum elements should correspond to per-grade Hermitian IPs."""
        mv = torch.randn(algebra_3d.dim)
        spec = hermitian_grade_spectrum(algebra_3d, mv)
        for k in range(algebra_3d.n + 1):
            mk = algebra_3d.grade_projection(mv, k)
            expected = torch.abs(hermitian_inner_product(algebra_3d, mk, mk).squeeze())
            assert torch.allclose(spec[k], expected, atol=1e-5)


class TestHermitianGradeSpectrum:
    def test_shape(self, algebra_3d):
        mv = torch.randn(algebra_3d.dim)
        spec = hermitian_grade_spectrum(algebra_3d, mv)
        assert spec.shape == (algebra_3d.n + 1,)

    def test_all_non_negative(self, algebra_minkowski):
        mv = torch.randn(algebra_minkowski.dim)
        spec = hermitian_grade_spectrum(algebra_minkowski, mv)
        assert (spec >= -1e-6).all()

    def test_scalar_only(self, algebra_3d):
        mv = torch.zeros(algebra_3d.dim)
        mv[0] = 3.0
        spec = hermitian_grade_spectrum(algebra_3d, mv)
        assert torch.allclose(spec[0], torch.tensor(9.0), atol=1e-5)
        assert torch.allclose(spec[1:], torch.zeros(algebra_3d.n), atol=1e-6)

    def test_sums_to_total_abs(self, algebra_3d):
        """Spectrum entries are |<A_k, A_k>_H|, sum should be consistent."""
        mv = torch.randn(algebra_3d.dim)
        spec = hermitian_grade_spectrum(algebra_3d, mv)
        # Each entry is abs of per-grade signed IP
        for k in range(algebra_3d.n + 1):
            mk = algebra_3d.grade_projection(mv, k)
            ip_k = hermitian_inner_product(algebra_3d, mk, mk)
            assert torch.allclose(spec[k], torch.abs(ip_k).squeeze(), atol=1e-5)

    def test_conformal_spectrum(self, algebra_conformal):
        mv = torch.randn(algebra_conformal.dim)
        spec = hermitian_grade_spectrum(algebra_conformal, mv)
        assert spec.shape == (algebra_conformal.n + 1,)
        assert (spec >= -1e-6).all()


class TestSignatureTraceForm:
    def test_matches_standard_for_euclidean(self, algebra_3d):
        a = torch.randn(algebra_3d.dim)
        trace = signature_trace_form(algebra_3d, a, a)
        std = inner_product(algebra_3d, a, algebra_3d.reverse(a))
        assert torch.allclose(trace, std, atol=1e-5)

    def test_can_be_negative_in_mixed(self, algebra_minkowski):
        """Trace form can go negative in Cl(2,1)."""
        found_negative = False
        for _ in range(100):
            mv = torch.randn(algebra_minkowski.dim)
            val = signature_trace_form(algebra_minkowski, mv, mv)
            if val < -1e-6:
                found_negative = True
                break
        assert found_negative, "Expected negative trace form values in Cl(2,1)"

    def test_signature_norm_squared(self, algebra_minkowski):
        mv = torch.randn(algebra_minkowski.dim)
        sn = signature_norm_squared(algebra_minkowski, mv)
        tf = signature_trace_form(algebra_minkowski, mv, mv)
        assert torch.allclose(sn, tf, atol=1e-6)


class TestComparisonHermitianVsSignature:
    def test_euclidean_norms_agree_for_scalars(self, algebra_3d):
        """For pure scalars, both norms should agree."""
        mv = torch.zeros(algebra_3d.dim)
        mv[0] = 7.0
        h = hermitian_norm(algebra_3d, mv)
        s = induced_norm(algebra_3d, mv)
        assert torch.allclose(h, s, atol=1e-5)

    def test_hermitian_ip_matches_bar_gp(self, algebra_3d):
        """Hermitian IP should match <bar{A}B>_0 via geometric product."""
        torch.manual_seed(99)
        mv = torch.randn(algebra_3d.dim)
        h_ip = hermitian_inner_product(algebra_3d, mv, mv)
        mv_bar = clifford_conjugate(algebra_3d, mv)
        prod = algebra_3d.geometric_product(mv_bar.unsqueeze(0), mv.unsqueeze(0))
        gp_scalar = prod[0, 0]
        assert torch.allclose(h_ip.squeeze(), gp_scalar, atol=1e-5)
