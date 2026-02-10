# Tests for Hermitian metrics in core/metric.py

import torch
import pytest
from core.algebra import CliffordAlgebra
from core.metric import (
    inner_product, induced_norm, geometric_distance,
    clifford_conjugate, hermitian_inner_product, hermitian_norm,
    hermitian_distance, hermitian_angle, grade_hermitian_norm,
    hermitian_grade_spectrum, signature_trace_form, signature_norm_squared,
    _hermitian_signs,
)


@pytest.fixture
def euclidean():
    return CliffordAlgebra(p=3, q=0, device='cpu')


@pytest.fixture
def spacetime():
    return CliffordAlgebra(p=2, q=1, device='cpu')


@pytest.fixture
def conformal():
    return CliffordAlgebra(p=4, q=1, device='cpu')


class TestHermitianSigns:
    def test_scalar_positive(self, euclidean):
        """Scalar component (grade 0) always has sign +1."""
        signs = _hermitian_signs(euclidean)
        assert signs[0] == 1.0

    def test_shape(self, spacetime):
        signs = _hermitian_signs(spacetime)
        assert signs.shape == (spacetime.dim,)

    def test_caching(self, spacetime):
        s1 = _hermitian_signs(spacetime)
        # Clear cache and recompute to test we get the same values
        if hasattr(spacetime, '_cached_hermitian_signs'):
            s2 = spacetime._cached_hermitian_signs
            assert torch.allclose(s1, s2)

    def test_values_are_pm1(self, conformal):
        signs = _hermitian_signs(conformal)
        assert torch.all(torch.abs(signs) == 1.0)

    def test_matches_geometric_product(self, spacetime):
        """Verify signed IP matches <bar{A} B>_0 via full geometric product."""
        torch.manual_seed(42)
        A = torch.randn(spacetime.dim)
        B = torch.randn(spacetime.dim)

        # Method 1: signed coefficient formula
        signs = _hermitian_signs(spacetime)
        ip_signed = (signs * A * B).sum()

        # Method 2: full geometric product of conjugate with B
        A_bar = clifford_conjugate(spacetime, A)
        prod = spacetime.geometric_product(A_bar.unsqueeze(0), B.unsqueeze(0))
        ip_gp = prod[0, 0]  # scalar part

        assert torch.allclose(ip_signed, ip_gp, atol=1e-5), \
            f"Signed IP {ip_signed.item():.6f} != GP {ip_gp.item():.6f}"

    def test_matches_gp_conformal(self, conformal):
        """Same verification for Cl(4,1)."""
        torch.manual_seed(123)
        A = torch.randn(conformal.dim)
        B = torch.randn(conformal.dim)

        signs = _hermitian_signs(conformal)
        ip_signed = (signs * A * B).sum()

        A_bar = clifford_conjugate(conformal, A)
        prod = conformal.geometric_product(A_bar.unsqueeze(0), B.unsqueeze(0))
        ip_gp = prod[0, 0]

        assert torch.allclose(ip_signed, ip_gp, atol=1e-4), \
            f"Signed IP {ip_signed.item():.6f} != GP {ip_gp.item():.6f}"


class TestCliffordConjugate:
    def test_scalar_unchanged(self, euclidean):
        mv = torch.zeros(euclidean.dim)
        mv[0] = 3.0
        conj = clifford_conjugate(euclidean, mv)
        assert torch.allclose(conj[0], mv[0])

    def test_double_conjugate_is_identity(self, spacetime):
        mv = torch.randn(spacetime.dim)
        conj2 = clifford_conjugate(spacetime, clifford_conjugate(spacetime, mv))
        assert torch.allclose(conj2, mv, atol=1e-6)

    def test_batch(self, euclidean):
        mv = torch.randn(5, euclidean.dim)
        conj = clifford_conjugate(euclidean, mv)
        assert conj.shape == mv.shape


class TestHermitianInnerProduct:
    def test_matches_bar_gp_euclidean(self, euclidean):
        """Hermitian IP should match <bar{A}B>_0 via geometric product."""
        torch.manual_seed(42)
        A = torch.randn(euclidean.dim)
        B = torch.randn(euclidean.dim)
        ip = hermitian_inner_product(euclidean, A, B)
        A_bar = clifford_conjugate(euclidean, A)
        prod = euclidean.geometric_product(A_bar.unsqueeze(0), B.unsqueeze(0))
        assert torch.allclose(ip.squeeze(), prod[0, 0], atol=1e-5)

    def test_positive_for_pure_scalars(self, euclidean):
        """For pure scalars, <bar{s}s>_0 = s^2 >= 0."""
        mv = torch.zeros(euclidean.dim)
        mv[0] = 5.0
        ip = hermitian_inner_product(euclidean, mv, mv)
        assert ip > 0

    def test_zero_for_zero(self, spacetime):
        mv = torch.zeros(spacetime.dim)
        ip = hermitian_inner_product(spacetime, mv, mv)
        assert torch.allclose(ip, torch.tensor([0.0]))

    def test_symmetry(self, spacetime):
        a = torch.randn(spacetime.dim)
        b = torch.randn(spacetime.dim)
        assert torch.allclose(
            hermitian_inner_product(spacetime, a, b),
            hermitian_inner_product(spacetime, b, a),
            atol=1e-6
        )

    def test_linearity(self, spacetime):
        a = torch.randn(spacetime.dim)
        b = torch.randn(spacetime.dim)
        c = torch.randn(spacetime.dim)
        alpha = 2.5
        lhs = hermitian_inner_product(spacetime, alpha * a + b, c)
        rhs = alpha * hermitian_inner_product(spacetime, a, c) + hermitian_inner_product(spacetime, b, c)
        assert torch.allclose(lhs, rhs, atol=1e-4)

    def test_has_negative_signs(self, spacetime):
        """Cl(2,1) should have some negative signs in the Hermitian form."""
        signs = _hermitian_signs(spacetime)
        has_negative = (signs < 0).any()
        assert has_negative, "Cl(2,1) should have negative signs"


class TestHermitianNorm:
    def test_non_negative(self, spacetime):
        mv = torch.randn(20, spacetime.dim)
        norms = hermitian_norm(spacetime, mv)
        assert (norms >= -1e-6).all()

    def test_zero_for_zero(self, euclidean):
        mv = torch.zeros(euclidean.dim)
        n = hermitian_norm(euclidean, mv)
        assert torch.allclose(n, torch.tensor([0.0]))

    def test_positive_for_nonzero(self, euclidean):
        mv = torch.randn(euclidean.dim)
        assert hermitian_norm(euclidean, mv) > 0

    def test_homogeneity(self, euclidean):
        """||alpha*A||_H = |alpha| * ||A||_H for Euclidean."""
        mv = torch.randn(euclidean.dim)
        alpha = 3.0
        n1 = hermitian_norm(euclidean, alpha * mv)
        n2 = alpha * hermitian_norm(euclidean, mv)
        assert torch.allclose(n1, n2, atol=1e-4)


class TestHermitianDistance:
    def test_zero_self_distance(self, spacetime):
        mv = torch.randn(spacetime.dim)
        d = hermitian_distance(spacetime, mv, mv)
        assert torch.allclose(d, torch.tensor([0.0]), atol=1e-6)

    def test_symmetry(self, spacetime):
        a = torch.randn(spacetime.dim)
        b = torch.randn(spacetime.dim)
        d1 = hermitian_distance(spacetime, a, b)
        d2 = hermitian_distance(spacetime, b, a)
        assert torch.allclose(d1, d2, atol=1e-5)

    def test_triangle_inequality(self, euclidean):
        """Triangle inequality holds for Euclidean (all signs +1)."""
        a = torch.randn(euclidean.dim)
        b = torch.randn(euclidean.dim)
        c = torch.randn(euclidean.dim)
        d_ab = hermitian_distance(euclidean, a, b)
        d_bc = hermitian_distance(euclidean, b, c)
        d_ac = hermitian_distance(euclidean, a, c)
        assert d_ac <= d_ab + d_bc + 1e-5

    def test_positive_for_different(self, euclidean):
        a = torch.randn(euclidean.dim)
        b = a + torch.randn(euclidean.dim) * 0.1
        d = hermitian_distance(euclidean, a, b)
        assert d > 0


class TestHermitianAngle:
    def test_zero_angle_same(self, euclidean):
        mv = torch.randn(euclidean.dim)
        angle = hermitian_angle(euclidean, mv, mv)
        assert torch.allclose(angle, torch.tensor([0.0]), atol=1e-5)

    def test_angle_range(self, euclidean):
        a = torch.randn(euclidean.dim)
        b = torch.randn(euclidean.dim)
        angle = hermitian_angle(euclidean, a, b)
        assert angle >= 0 and angle <= torch.pi + 1e-5

    def test_orthogonal(self, euclidean):
        a = torch.zeros(euclidean.dim)
        b = torch.zeros(euclidean.dim)
        a[0] = 1.0  # scalar
        b[1] = 1.0  # e1
        angle = hermitian_angle(euclidean, a, b)
        assert torch.allclose(angle, torch.tensor([torch.pi / 2]), atol=1e-5)


class TestGradeHermitianNorm:
    def test_scalar_grade(self, euclidean):
        mv = torch.zeros(euclidean.dim)
        mv[0] = 5.0
        n = grade_hermitian_norm(euclidean, mv, grade=0)
        assert torch.allclose(n, torch.tensor([5.0]), atol=1e-6)

    def test_zero_for_wrong_grade(self, euclidean):
        mv = torch.zeros(euclidean.dim)
        mv[0] = 5.0  # Only scalar
        n = grade_hermitian_norm(euclidean, mv, grade=1)
        assert torch.allclose(n, torch.tensor([0.0]), atol=1e-6)

    def test_grade_decomposition(self, euclidean):
        """Grade spectrum elements should correspond to per-grade Hermitian IPs."""
        mv = torch.randn(euclidean.dim)
        spec = hermitian_grade_spectrum(euclidean, mv)
        for k in range(euclidean.n + 1):
            mk = euclidean.grade_projection(mv, k)
            expected = torch.abs(hermitian_inner_product(euclidean, mk, mk).squeeze())
            assert torch.allclose(spec[k], expected, atol=1e-5)


class TestHermitianGradeSpectrum:
    def test_shape(self, euclidean):
        mv = torch.randn(euclidean.dim)
        spec = hermitian_grade_spectrum(euclidean, mv)
        assert spec.shape == (euclidean.n + 1,)

    def test_all_non_negative(self, spacetime):
        mv = torch.randn(spacetime.dim)
        spec = hermitian_grade_spectrum(spacetime, mv)
        assert (spec >= -1e-6).all()

    def test_scalar_only(self, euclidean):
        mv = torch.zeros(euclidean.dim)
        mv[0] = 3.0
        spec = hermitian_grade_spectrum(euclidean, mv)
        assert torch.allclose(spec[0], torch.tensor(9.0), atol=1e-5)
        assert torch.allclose(spec[1:], torch.zeros(euclidean.n), atol=1e-6)

    def test_sums_to_total_abs(self, euclidean):
        """Spectrum entries are |<A_k, A_k>_H|, sum should be consistent."""
        mv = torch.randn(euclidean.dim)
        spec = hermitian_grade_spectrum(euclidean, mv)
        # Each entry is abs of per-grade signed IP
        for k in range(euclidean.n + 1):
            mk = euclidean.grade_projection(mv, k)
            ip_k = hermitian_inner_product(euclidean, mk, mk)
            assert torch.allclose(spec[k], torch.abs(ip_k).squeeze(), atol=1e-5)

    def test_conformal_spectrum(self, conformal):
        mv = torch.randn(conformal.dim)
        spec = hermitian_grade_spectrum(conformal, mv)
        assert spec.shape == (conformal.n + 1,)
        assert (spec >= -1e-6).all()


class TestSignatureTraceForm:
    def test_matches_standard_for_euclidean(self, euclidean):
        a = torch.randn(euclidean.dim)
        trace = signature_trace_form(euclidean, a, a)
        std = inner_product(euclidean, a, euclidean.reverse(a))
        assert torch.allclose(trace, std, atol=1e-5)

    def test_can_be_negative_in_mixed(self, spacetime):
        """Trace form can go negative in Cl(2,1)."""
        found_negative = False
        for _ in range(100):
            mv = torch.randn(spacetime.dim)
            val = signature_trace_form(spacetime, mv, mv)
            if val < -1e-6:
                found_negative = True
                break
        assert found_negative, "Expected negative trace form values in Cl(2,1)"

    def test_signature_norm_squared(self, spacetime):
        mv = torch.randn(spacetime.dim)
        sn = signature_norm_squared(spacetime, mv)
        tf = signature_trace_form(spacetime, mv, mv)
        assert torch.allclose(sn, tf, atol=1e-6)


class TestComparisonHermitianVsSignature:
    def test_euclidean_norms_agree_for_scalars(self, euclidean):
        """For pure scalars, both norms should agree."""
        mv = torch.zeros(euclidean.dim)
        mv[0] = 7.0
        h = hermitian_norm(euclidean, mv)
        s = induced_norm(euclidean, mv)
        assert torch.allclose(h, s, atol=1e-5)

    def test_hermitian_ip_matches_bar_gp(self, euclidean):
        """Hermitian IP should match <bar{A}B>_0 via geometric product."""
        torch.manual_seed(99)
        mv = torch.randn(euclidean.dim)
        h_ip = hermitian_inner_product(euclidean, mv, mv)
        mv_bar = clifford_conjugate(euclidean, mv)
        prod = euclidean.geometric_product(mv_bar.unsqueeze(0), mv.unsqueeze(0))
        gp_scalar = prod[0, 0]
        assert torch.allclose(h_ip.squeeze(), gp_scalar, atol=1e-5)
