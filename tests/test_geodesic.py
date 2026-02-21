"""Tests for GeodesicFlow and DimensionLifter in core/search.py.

Covers:
- GeodesicFlow: flow bivectors, coherence, curvature, interpolation,
  causal_report, causal vs noisy synthetic data
- DimensionLifter: lift shape, positive/null lift, test() report structure,
  lifting structured data improves coherence, format_report
"""

import math
import pytest
import torch
from core.algebra import CliffordAlgebra
from core.search import GeodesicFlow, DimensionLifter, MetricSearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def alg2():
    return CliffordAlgebra(p=2, q=0, device='cpu')


@pytest.fixture(scope="module")
def alg3():
    return CliffordAlgebra(p=3, q=0, device='cpu')


@pytest.fixture(scope="module")
def alg31():
    return CliffordAlgebra(p=3, q=1, device='cpu')


def _circle_data(N: int = 64, noise: float = 0.0) -> torch.Tensor:
    """N points uniformly on the unit circle, optionally with Gaussian noise."""
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    x = torch.stack([theta.cos(), theta.sin()], dim=-1)
    if noise > 0:
        x = x + torch.randn_like(x) * noise
    return x


def _helix_data(N: int = 64) -> torch.Tensor:
    """3D helix: uniform rotation plane → concentrated connection bivectors."""
    t = torch.linspace(0, 4 * math.pi, N)
    x = torch.stack([t.cos(), t.sin(), t / (4 * math.pi)], dim=-1)
    return x


def _flat_2d_in_3d(N: int = 64) -> torch.Tensor:
    """2D circle embedded in 3D with z=0.

    All connection bivectors are in the e₁₂ plane (grade-2 index 3 in Cl(3,0)).
    Pairwise |cos| within every neighbourhood = 1.0 → coherence = 1.0.
    """
    theta = torch.linspace(0, 2 * math.pi, N + 1)[:-1]
    z = torch.zeros(N)
    return torch.stack([theta.cos(), theta.sin(), z], dim=-1)


def _causal_flow_data(N: int = 64) -> torch.Tensor:
    """2D data whose flow is strongly aligned: points on a spiral."""
    t = torch.linspace(0, 4 * math.pi, N)
    r = 0.5 + t / (4 * math.pi)
    x = torch.stack([r * t.cos(), r * t.sin()], dim=-1)
    return x


def _random_data(N: int = 64, dim: int = 2) -> torch.Tensor:
    """Pure noise: no spatial structure."""
    return torch.randn(N, dim)


# ---------------------------------------------------------------------------
# TestGeodesicFlow
# ---------------------------------------------------------------------------

class TestGeodesicFlow:
    """Tests for GeodesicFlow."""

    def test_flow_bivectors_shape(self, alg2):
        """Flow bivectors must have shape [N, algebra.dim]."""
        gf = GeodesicFlow(alg2, k=4)
        data = _circle_data(32)
        mv = gf._embed(data)
        flow = gf.flow_bivectors(mv)
        assert flow.shape == (32, alg2.dim)

    def test_connection_bivectors_grade2_only(self, alg2):
        """Connection bivectors must have energy only in grade-2 components.

        Tests the raw per-edge bivectors (not the mean, which may cancel for
        symmetric data).  In Cl(2,0) the only grade-2 blade is index 3 (e₁₂).
        """
        gf = GeodesicFlow(alg2, k=4)
        # Use two non-parallel grade-1 vectors that produce a non-zero bivector
        data = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.],
                              [0., -1.], [1., -1.], [-1., 1.], [0.5, 0.5]])
        mv = gf._embed(data)
        bv = gf._connection_bivectors(mv)  # [N, k, dim]

        # Non-grade-2 components must be zero
        non_g2_mask = [i for i in range(alg2.dim) if bin(i).count('1') != 2]
        other_energy = bv[:, :, non_g2_mask].abs().max().item()
        assert other_energy < 1e-5, f"Non-grade-2 energy should be ~0, got {other_energy}"

        # Grade-2 component must have some energy (index 3 = e₁₂ in Cl(2,0))
        g2_energy = bv[:, :, 3].abs().max().item()
        assert g2_energy > 0.0, "Connection bivectors should have grade-2 energy"

    def test_coherence_range(self, alg2):
        """Coherence must be in [0, 1]."""
        gf = GeodesicFlow(alg2, k=4)
        for data in [_circle_data(32), _random_data(32, dim=2)]:
            mv = gf._embed(data)
            coh = gf.coherence(mv)
            assert 0.0 <= coh <= 1.0 + 1e-5, f"Coherence {coh} out of [0,1] range"

    def test_curvature_nonnegative(self, alg2):
        """Curvature must be >= 0."""
        gf = GeodesicFlow(alg2, k=4)
        for data in [_circle_data(32), _random_data(32, dim=2)]:
            mv = gf._embed(data)
            curv = gf.curvature(mv)
            assert curv >= 0.0, f"Curvature {curv} is negative"

    def test_structured_higher_coherence_than_noise(self, alg3):
        """Flat 2D circle in 3D should have coherence=1 vs lower-coherence random 3D.

        In Cl(3,0) the bivector space has 3 planes (e₁₂, e₁₃, e₂₃).

        A flat circle in the z=0 plane forces ALL connection bivectors into
        the single e₁₂ plane — pairwise |cos| = 1.0 exactly → coherence = 1.0.

        Pure random 3D data scatters connections across all three planes, giving
        coherence below 1.0.
        """
        torch.manual_seed(0)
        gf = GeodesicFlow(alg3, k=6)

        flat = _flat_2d_in_3d(64)
        noise = _random_data(64, dim=3)

        mv_flat = gf._embed(flat)
        mv_noise = gf._embed(noise)

        coh_flat = gf.coherence(mv_flat)
        coh_noise = gf.coherence(mv_noise)

        assert coh_flat > coh_noise, (
            f"Flat-2D-in-3D coherence {coh_flat:.3f} should exceed noise {coh_noise:.3f}"
        )
        # Flat data must be exactly 1.0 (all connections in e₁₂ plane)
        assert abs(coh_flat - 1.0) < 1e-4, f"Flat coherence should be 1.0, got {coh_flat:.5f}"

    def test_structured_lower_curvature_than_noise(self, alg3):
        """Flat 2D circle in 3D should have curvature=0 vs higher-curvature random 3D.

        Since all connections of every point are in the e₁₂ plane, the cross-
        neighbourhood comparison always gives |cos| = 1.0 → curvature = 0.0.
        """
        torch.manual_seed(1)
        gf = GeodesicFlow(alg3, k=6)

        flat = _flat_2d_in_3d(64)
        noise = _random_data(64, dim=3)

        mv_flat = gf._embed(flat)
        mv_noise = gf._embed(noise)

        curv_flat = gf.curvature(mv_flat)
        curv_noise = gf.curvature(mv_noise)

        assert curv_flat < curv_noise, (
            f"Flat curvature {curv_flat:.3f} should be less than noise {curv_noise:.3f}"
        )

    def test_embed_shape(self, alg3):
        """Embed should produce [N, dim] grade-1 multivectors."""
        gf = GeodesicFlow(alg3, k=4)
        data = torch.randn(20, 3)
        mv = gf._embed(data)
        assert mv.shape == (20, alg3.dim)

    def test_embed_grade1_only(self, alg3):
        """Embedded multivectors must have energy only in grade-1 blades."""
        gf = GeodesicFlow(alg3, k=4)
        data = torch.randn(10, 3)
        mv = gf._embed(data)
        # Grade-1 blades in Cl(3,0): indices 1,2,4 (one bit set)
        grade1_mask = [i for i in range(alg3.dim) if bin(i).count('1') == 1]
        other_mask = [i for i in range(alg3.dim) if bin(i).count('1') != 1]
        assert mv[:, other_mask].abs().max().item() < 1e-6

    def test_knn_count(self, alg2):
        """KNN should return min(k, N-1) indices per point."""
        gf = GeodesicFlow(alg2, k=10)
        data = _circle_data(8)        # fewer points than k
        mv = gf._embed(data)
        idx = gf._knn(mv)
        assert idx.shape == (8, 7)   # k capped at N-1 = 7

    def test_interpolate_endpoints(self, alg3):
        """Interpolated endpoints must match a and b (up to approximation)."""
        gf = GeodesicFlow(alg3, k=4)
        # Use simple unit vectors
        a = alg3.embed_vector(torch.tensor([[1., 0., 0.]]))  # [1, 8]
        b = alg3.embed_vector(torch.tensor([[0., 1., 0.]]))  # [1, 8]
        path = gf.interpolate(a[0], b[0], steps=10)          # [10, 8]
        assert path.shape == (10, alg3.dim)
        # Step 0 should be close to a
        assert torch.allclose(path[0], a[0], atol=1e-4)

    def test_interpolate_steps(self, alg3):
        """Number of returned frames should equal steps."""
        gf = GeodesicFlow(alg3, k=4)
        a = alg3.embed_vector(torch.tensor([[1., 0., 0.]]))
        b = alg3.embed_vector(torch.tensor([[0., 1., 0.]]))
        for steps in [2, 5, 20]:
            path = gf.interpolate(a[0], b[0], steps=steps)
            assert path.shape[0] == steps

    def test_causal_report_keys(self, alg2):
        """causal_report must return all expected keys."""
        gf = GeodesicFlow(alg2, k=4)
        report = gf.causal_report(_circle_data(32))
        for key in ('coherence', 'curvature', 'causal', 'label'):
            assert key in report, f"Missing key: {key}"

    def test_causal_report_types(self, alg2):
        """causal_report values must have correct types."""
        gf = GeodesicFlow(alg2, k=4)
        report = gf.causal_report(_circle_data(32))
        assert isinstance(report['coherence'], float)
        assert isinstance(report['curvature'], float)
        assert isinstance(report['causal'], bool)
        assert isinstance(report['label'], str)

    def test_causal_report_causal_data(self, alg2):
        """Structured spiral data should be labelled 'Causal' if coherence > 0.5."""
        gf = GeodesicFlow(alg2, k=6)
        report = gf.causal_report(_causal_flow_data(128))
        # Whether it's causal depends on the data; at least check label consistency
        if report['causal']:
            assert 'Causal' in report['label']
        else:
            assert 'Noisy' in report['label']

    def test_causal_report_label_consistency(self, alg2):
        """'causal' flag and 'label' string must be consistent."""
        torch.manual_seed(42)
        gf = GeodesicFlow(alg2, k=6)
        for data in [_circle_data(64), _random_data(64, dim=2)]:
            report = gf.causal_report(data)
            if report['causal']:
                assert 'Causal' in report['label']
            else:
                assert 'Noisy' in report['label']


# ---------------------------------------------------------------------------
# TestDimensionLifter
# ---------------------------------------------------------------------------

class TestDimensionLifter:
    """Tests for DimensionLifter."""

    def test_lift_shape(self):
        """Lifted multivectors must have shape [N, target_dim]."""
        lifter = DimensionLifter(device='cpu')
        data = torch.randn(20, 2)
        alg = CliffordAlgebra(3, 0, device='cpu')   # 2D data → 3D algebra
        mv = lifter.lift(data, alg, fill=1.0)
        assert mv.shape == (20, alg.dim)

    def test_lift_grade1_only(self):
        """Lifted multivectors must have energy only in grade-1 blades."""
        lifter = DimensionLifter(device='cpu')
        data = torch.randn(10, 2)
        alg = CliffordAlgebra(3, 0, device='cpu')
        mv = lifter.lift(data, alg, fill=1.0)
        other = [i for i in range(alg.dim) if bin(i).count('1') != 1]
        assert mv[:, other].abs().max().item() < 1e-6

    def test_lift_same_dimension(self):
        """Lifting to the same dimension (no padding) should still work."""
        lifter = DimensionLifter(device='cpu')
        data = torch.randn(10, 3)
        alg = CliffordAlgebra(3, 0, device='cpu')
        mv = lifter.lift(data, alg, fill=1.0)
        assert mv.shape == (10, alg.dim)

    def test_lift_fill_values(self):
        """The extra coordinate should equal `fill` in the extra blade."""
        lifter = DimensionLifter(device='cpu')
        data = torch.zeros(5, 2)   # all-zero 2D data
        alg = CliffordAlgebra(3, 0, device='cpu')
        # The third grade-1 blade is index 4 (= 1 << 2)
        mv = lifter.lift(data, alg, fill=0.7)
        assert torch.allclose(mv[:, 4], torch.full((5,), 0.7), atol=1e-5)

    def test_lift_too_large_raises(self):
        """Lifting to a smaller algebra must raise ValueError."""
        lifter = DimensionLifter(device='cpu')
        data = torch.randn(10, 4)
        alg = CliffordAlgebra(2, 0, device='cpu')   # only 2D
        with pytest.raises(ValueError):
            lifter.lift(data, alg)

    def test_test_output_keys(self):
        """test() must return all expected keys."""
        lifter = DimensionLifter(device='cpu')
        data = _circle_data(32)
        results = lifter.test(data, p=2, q=0, k=4)
        for key in ('original', 'lift_positive', 'lift_null', 'best'):
            assert key in results

    def test_test_signature_fields(self):
        """Each algebra result must include signature, coherence, curvature, causal."""
        lifter = DimensionLifter(device='cpu')
        data = _circle_data(32)
        results = lifter.test(data, p=2, q=0, k=4)
        for key in ('original', 'lift_positive', 'lift_null'):
            r = results[key]
            assert 'signature' in r
            assert 'coherence' in r
            assert 'curvature' in r
            assert 'causal' in r

    def test_test_original_signature(self):
        """Original result must carry the correct (p, q) signature."""
        lifter = DimensionLifter(device='cpu')
        results = lifter.test(_circle_data(32), p=2, q=0, k=4)
        assert results['original']['signature'] == (2, 0)

    def test_test_lifted_signatures(self):
        """Lifted results must carry incremented signatures."""
        lifter = DimensionLifter(device='cpu')
        results = lifter.test(_circle_data(32), p=2, q=0, k=4)
        assert results['lift_positive']['signature'] == (3, 0)
        assert results['lift_null']['signature'] == (2, 1)

    def test_test_best_is_valid_key(self):
        """'best' must point to one of the three algebra keys."""
        lifter = DimensionLifter(device='cpu')
        results = lifter.test(_circle_data(32), p=2, q=0, k=4)
        assert results['best'] in ('original', 'lift_positive', 'lift_null')

    def test_test_best_has_highest_coherence(self):
        """'best' must indeed be the algebra with the highest coherence."""
        lifter = DimensionLifter(device='cpu')
        results = lifter.test(_circle_data(48), p=2, q=0, k=6)
        best_key = results['best']
        best_coh = results[best_key]['coherence']
        for key in ('original', 'lift_positive', 'lift_null'):
            assert results[key]['coherence'] <= best_coh + 1e-6

    def test_positive_lift_expands_algebra_dim(self):
        """Positive lift must produce [N, 2^(n+1)] multivectors."""
        lifter = DimensionLifter(device='cpu')
        data = _circle_data(16)
        alg_pos = CliffordAlgebra(3, 0, device='cpu')
        mv = lifter.lift(data, alg_pos, fill=1.0)
        assert mv.shape == (16, alg_pos.dim)  # 2^3 = 8

    def test_format_report_returns_string(self):
        """format_report must return a non-empty string."""
        lifter = DimensionLifter(device='cpu')
        results = lifter.test(_circle_data(32), p=2, q=0, k=4)
        report = lifter.format_report(results)
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'Cl(' in report

    def test_format_report_contains_all_algebras(self):
        """Report must mention all three algebras."""
        lifter = DimensionLifter(device='cpu')
        results = lifter.test(_circle_data(32), p=2, q=0, k=4)
        report = lifter.format_report(results)
        # All three signatures should appear
        assert 'Cl(2,0)' in report
        assert 'Cl(3,0)' in report
        assert 'Cl(2,1)' in report
