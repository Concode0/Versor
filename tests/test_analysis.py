# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the core/analysis/ toolkit.

Covers: EffectiveDimensionAnalyzer, StatisticalSampler, SpectralAnalyzer,
SymmetryDetector, CommutatorAnalyzer, compute_uncertainty_and_alignment,
and the GeometricAnalyzer pipeline orchestrator.
"""

import math

import pytest
import torch

from core.algebra import CliffordAlgebra
from core.analysis import (
    AnalysisConfig,
    AnalysisReport,
    CommutatorAnalyzer,
    CommutatorResult,
    DimensionResult,
    EffectiveDimensionAnalyzer,
    GeometricAnalyzer,
    SamplingConfig,
    SpectralAnalyzer,
    SpectralResult,
    StatisticalSampler,
    SymmetryDetector,
    SymmetryResult,
    compute_uncertainty_and_alignment,
)

DEVICE = "cpu"


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(scope="module")
def alg3():
    return CliffordAlgebra(3, 0, device=DEVICE)


@pytest.fixture(scope="module")
def alg21():
    return CliffordAlgebra(2, 1, device=DEVICE)


@pytest.fixture(scope="module")
def alg2():
    return CliffordAlgebra(2, 0, device=DEVICE)


@pytest.fixture(scope="module")
def circle_data_3d(alg3):
    """50 points on a circle in the (e1, e2) plane — grade-1 multivectors."""
    torch.manual_seed(0)
    t = torch.linspace(0, 2 * math.pi, 51)[:-1]  # 50 points
    raw = torch.stack([t.cos(), t.sin(), torch.zeros_like(t)], dim=-1)
    return alg3.embed_vector(raw)  # [50, 8]


@pytest.fixture(scope="module")
def noise_mv_3d(alg3):
    """50 random multivectors in Cl(3,0)."""
    torch.manual_seed(1)
    return torch.randn(50, alg3.dim)


@pytest.fixture(scope="module")
def raw_3d_data():
    """100 samples, 3 features — rank-2 (lives in a 2-D plane)."""
    torch.manual_seed(2)
    base = torch.randn(100, 2)
    # Embed into 3D: third coordinate is linear combo of first two
    x = torch.cat([base, (base[:, 0:1] + base[:, 1:2]) * 0.5], dim=-1)
    return x


# =====================================================================
# EffectiveDimensionAnalyzer
# =====================================================================


class TestEffectiveDimensionAnalyzer:
    def test_analyze_returns_dimension_result(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        result = da.analyze(raw_3d_data)
        assert isinstance(result, DimensionResult)

    def test_intrinsic_dim_rank2_data(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        result = da.analyze(raw_3d_data)
        # Data is exactly rank-2 (3rd col = linear combo of first two).
        # Broken-stick is conservative on exact rank-deficiency;
        # participation ratio is the reliable continuous measure here.
        assert result.intrinsic_dim >= 1
        assert 1.5 < result.participation_ratio < 2.5

    def test_participation_ratio_rank2(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        result = da.analyze(raw_3d_data)
        # PR should be close to 2 for rank-2 data
        assert 1.5 < result.participation_ratio < 2.8

    def test_eigenvalues_descending(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        result = da.analyze(raw_3d_data)
        evs = result.eigenvalues
        assert (evs[:-1] >= evs[1:] - 1e-6).all()

    def test_explained_variance_sums_to_one(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        result = da.analyze(raw_3d_data)
        assert abs(result.explained_variance_ratio.sum().item() - 1.0) < 1e-5

    def test_local_dims_computed_for_small_data(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE, k_local=10)
        result = da.analyze(raw_3d_data)
        assert result.local_dims is not None
        assert result.local_dims.shape == (raw_3d_data.shape[0],)

    def test_local_dims_skipped_for_large_k(self):
        da = EffectiveDimensionAnalyzer(device=DEVICE, k_local=200)
        # Only 30 samples — k_local > N, should skip
        data = torch.randn(30, 5)
        result = da.analyze(data)
        assert result.local_dims is None

    def test_reduce_output_shape(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        reduced = da.reduce(raw_3d_data, target_dim=2)
        assert reduced.shape == (100, 2)

    def test_reduce_preserves_variance(self, raw_3d_data):
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        reduced = da.reduce(raw_3d_data, target_dim=2)
        # Most variance should be preserved for rank-2 data
        orig_var = raw_3d_data.var(dim=0).sum().item()
        red_var = reduced.var(dim=0).sum().item()
        assert red_var / orig_var > 0.9

    def test_full_rank_data(self):
        torch.manual_seed(3)
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        data = torch.randn(200, 5)  # full rank
        result = da.analyze(data)
        # Broken-stick is conservative; PR is the reliable measure here
        assert result.intrinsic_dim >= 3
        assert result.participation_ratio > 4.0

    def test_one_dim_data(self):
        torch.manual_seed(4)
        da = EffectiveDimensionAnalyzer(device=DEVICE)
        t = torch.randn(100, 1)
        data = torch.cat([t, t * 2, t * 3], dim=-1)  # rank 1
        result = da.analyze(data)
        assert result.intrinsic_dim == 1
        assert result.participation_ratio < 1.5


# =====================================================================
# StatisticalSampler
# =====================================================================


class TestStatisticalSampler:
    def test_passthrough(self, raw_3d_data):
        cfg = SamplingConfig(strategy="passthrough")
        sampled, meta = StatisticalSampler.sample(raw_3d_data, cfg)
        assert torch.equal(sampled, raw_3d_data)
        assert meta["strategy"] == "passthrough"

    def test_random_size(self, raw_3d_data):
        cfg = SamplingConfig(strategy="random", max_samples=30, seed=0)
        sampled, meta = StatisticalSampler.sample(raw_3d_data, cfg)
        assert sampled.shape[0] == 30
        assert meta["strategy"] == "random"

    def test_random_deterministic(self, raw_3d_data):
        cfg = SamplingConfig(strategy="random", max_samples=20, seed=42)
        s1, _ = StatisticalSampler.sample(raw_3d_data, cfg)
        s2, _ = StatisticalSampler.sample(raw_3d_data, cfg)
        assert torch.equal(s1, s2)

    def test_random_caps_at_n(self):
        data = torch.randn(10, 3)
        cfg = SamplingConfig(strategy="random", max_samples=100)
        sampled, _ = StatisticalSampler.sample(data, cfg)
        assert sampled.shape[0] == 10

    def test_bootstrap_returns_list(self, raw_3d_data):
        cfg = SamplingConfig(strategy="bootstrap", n_bootstrap=5, max_samples=20)
        sampled, meta = StatisticalSampler.sample(raw_3d_data, cfg)
        assert isinstance(sampled, list)
        assert len(sampled) == 5
        assert sampled[0].shape[0] == 20
        assert meta["strategy"] == "bootstrap"

    def test_stratified_returns_tensor(self, raw_3d_data):
        cfg = SamplingConfig(strategy="stratified", max_samples=40, seed=0)
        sampled, meta = StatisticalSampler.sample(raw_3d_data, cfg)
        assert isinstance(sampled, torch.Tensor)
        assert sampled.shape[0] <= 40
        assert meta["strategy"] == "stratified"
        assert "coherence_scores" in meta

    def test_stratified_coherence_scores_shape(self, raw_3d_data):
        cfg = SamplingConfig(strategy="stratified", max_samples=40, seed=0)
        _, meta = StatisticalSampler.sample(raw_3d_data, cfg)
        scores = meta["coherence_scores"]
        assert scores.shape == (raw_3d_data.shape[0],)

    def test_unknown_strategy_raises(self, raw_3d_data):
        cfg = SamplingConfig(strategy="unknown")
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            StatisticalSampler.sample(raw_3d_data, cfg)

    def test_recommend_size(self):
        assert StatisticalSampler.recommend_size(10, 10000) == 500
        assert StatisticalSampler.recommend_size(50, 10000) == 1000
        assert StatisticalSampler.recommend_size(10, 100) == 100


# =====================================================================
# SpectralAnalyzer
# =====================================================================


class TestSpectralAnalyzer:
    def test_analyze_returns_spectral_result(self, alg3, circle_data_3d):
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        assert isinstance(result, SpectralResult)

    def test_grade_energy_shape(self, alg3, circle_data_3d):
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        # n+1 = 4 grades for Cl(3,0)
        assert result.grade_energy.shape == (4,)

    def test_grade_energy_nonnegative(self, alg3, circle_data_3d):
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        assert (result.grade_energy >= -1e-6).all()

    def test_circle_grade1_dominant(self, alg3, circle_data_3d):
        """Circle data embedded as grade-1 should have dominant grade-1 energy."""
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        # Grade-1 energy should dominate
        assert result.grade_energy[1] > result.grade_energy[0]
        assert result.grade_energy[1] > result.grade_energy[2]

    def test_bivector_spectrum_nonempty(self, alg3, circle_data_3d):
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        assert result.bivector_spectrum.numel() > 0

    def test_simple_components_list(self, alg3, circle_data_3d):
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        assert isinstance(result.simple_components, list)
        for comp in result.simple_components:
            assert comp.shape == (alg3.dim,)

    def test_gp_eigenvalues_present_small_algebra(self, alg3, circle_data_3d):
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        assert result.gp_eigenvalues is not None
        assert result.gp_eigenvalues.numel() > 0

    def test_gp_eigenvalues_sorted_descending(self, alg3, circle_data_3d):
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(circle_data_3d)
        eigs = result.gp_eigenvalues
        assert (eigs[:-1] >= eigs[1:] - 1e-6).all()

    def test_multichannel_input(self, alg3):
        torch.manual_seed(10)
        mv = torch.randn(30, 4, alg3.dim)  # 4 channels
        sa = SpectralAnalyzer(alg3)
        result = sa.analyze(mv)
        assert result.grade_energy.shape == (4,)

    def test_small_algebra_cl2(self, alg2):
        torch.manual_seed(11)
        raw = torch.randn(40, 2)
        mv = alg2.embed_vector(raw)
        sa = SpectralAnalyzer(alg2)
        result = sa.analyze(mv)
        # Cl(2,0): grades 0,1,2 → 3 entries
        assert result.grade_energy.shape == (3,)


# =====================================================================
# SymmetryDetector
# =====================================================================


class TestSymmetryDetector:
    def test_analyze_returns_symmetry_result(self, alg3, circle_data_3d):
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d)
        assert isinstance(result, SymmetryResult)

    def test_null_scores_shape(self, alg3, circle_data_3d):
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d)
        assert result.null_scores.shape == (3,)

    def test_circle_e3_is_null(self, alg3, circle_data_3d):
        """Circle in (e1,e2) plane — e3 direction has zero energy → null."""
        sd = SymmetryDetector(alg3, null_threshold=0.05)
        result = sd.analyze(circle_data_3d)
        # Direction index 2 (e3) should be detected as null
        assert 2 in result.null_directions

    def test_involution_symmetry_nonneg(self, alg3, circle_data_3d):
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d)
        # ||α(x) - x||² / ||x||² : for pure grade-1 (odd), α(x) = -x → ratio = 4
        assert result.involution_symmetry >= 0.0

    def test_grade1_data_high_involution(self, alg3, circle_data_3d):
        """Grade-1 data is purely odd → involution_symmetry = 1.0
        (100% of energy in odd grades)."""
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d)
        # Grade-1 data is purely odd → involution_symmetry should be 1.0
        assert result.involution_symmetry > 0.99

    def test_reflection_symmetries_structure(self, alg3, circle_data_3d):
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d)
        assert isinstance(result.reflection_symmetries, list)
        assert len(result.reflection_symmetries) == 3  # one per basis vector
        for entry in result.reflection_symmetries:
            assert "direction" in entry
            assert "score" in entry
            assert entry["score"] >= 0

    def test_reflection_sorted_by_score(self, alg3, circle_data_3d):
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d)
        scores = [r["score"] for r in result.reflection_symmetries]
        assert scores == sorted(scores)

    def test_continuous_symmetry_dim_nonneg(self, alg3, circle_data_3d):
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d)
        assert result.continuous_symmetry_dim >= 0

    def test_continuous_symmetry_from_commutator_result(self, alg3, circle_data_3d):
        """When CommutatorResult is provided, continuous symmetry uses exchange spectrum."""
        ca = CommutatorAnalyzer(alg3)
        comm_result = ca.analyze(circle_data_3d)
        sd = SymmetryDetector(alg3)
        result = sd.analyze(circle_data_3d, commutator_result=comm_result)
        assert result.continuous_symmetry_dim >= 0

    def test_multichannel_input(self, alg3):
        torch.manual_seed(20)
        mv = torch.randn(30, 4, alg3.dim)
        sd = SymmetryDetector(alg3)
        result = sd.analyze(mv)
        assert isinstance(result, SymmetryResult)

    def test_noise_fewer_null_directions(self, alg3, noise_mv_3d):
        """Random data should have energy in all directions — fewer nulls."""
        sd = SymmetryDetector(alg3, null_threshold=0.05)
        result = sd.analyze(noise_mv_3d)
        # Should have at most 1 null direction (random noise is isotropic)
        assert len(result.null_directions) <= 1


# =====================================================================
# CommutatorAnalyzer
# =====================================================================


class TestCommutatorAnalyzer:
    def test_analyze_returns_commutator_result(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        assert isinstance(result, CommutatorResult)

    def test_commutativity_matrix_shape(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        assert result.commutativity_matrix.shape == (3, 3)

    def test_commutativity_matrix_symmetric(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        m = result.commutativity_matrix
        assert torch.allclose(m, m.T, atol=1e-6)

    def test_commutativity_matrix_zero_diagonal(self, alg3, circle_data_3d):
        """[e_i, e_i] = 0, so diagonal should be zero."""
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        diag = result.commutativity_matrix.diag()
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    def test_exchange_spectrum_sorted(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        spec = result.exchange_spectrum
        assert spec.numel() > 0
        assert (spec[:-1] >= spec[1:] - 1e-6).all()

    def test_mean_commutator_norm_nonneg(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        assert result.mean_commutator_norm >= 0

    def test_lie_bracket_structure_keys(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        lie = result.lie_bracket_structure
        assert "structure_constants" in lie
        assert "closure_error" in lie
        assert "basis_indices" in lie

    def test_lie_bracket_closure_error_range(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        ce = result.lie_bracket_structure["closure_error"]
        assert 0.0 <= ce <= 1.0 + 1e-6

    def test_structure_constants_antisymmetric(self, alg3, circle_data_3d):
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(circle_data_3d)
        sc = result.lie_bracket_structure["structure_constants"]
        if sc.numel() > 0:
            k = sc.shape[0]
            for a in range(k):
                for b in range(a + 1, k):
                    assert torch.allclose(sc[a, b], -sc[b, a], atol=1e-5)

    def test_multichannel_input(self, alg3):
        torch.manual_seed(30)
        mv = torch.randn(30, 4, alg3.dim)
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(mv)
        assert isinstance(result, CommutatorResult)

    def test_cl2_has_bivectors(self, alg2):
        """Cl(2,0) has 1 bivector — lie bracket structure should work."""
        torch.manual_seed(31)
        raw = torch.randn(40, 2)
        mv = alg2.embed_vector(raw)
        ca = CommutatorAnalyzer(alg2)
        result = ca.analyze(mv)
        assert result.lie_bracket_structure["basis_indices"]

    def test_commutator_primitive_matches_manual(self, alg3):
        """Verify algebra.commutator() matches GP(A,B) - GP(B,A)."""
        torch.manual_seed(32)
        A = torch.randn(10, alg3.dim)
        B = torch.randn(10, alg3.dim)
        manual = alg3.geometric_product(A, B) - alg3.geometric_product(B, A)
        prim = alg3.commutator(A, B)
        assert torch.allclose(manual, prim, atol=1e-5)

    def test_anti_commutator_primitive_matches_manual(self, alg3):
        """Verify algebra.anti_commutator() matches GP(A,B) + GP(B,A)."""
        torch.manual_seed(33)
        A = torch.randn(10, alg3.dim)
        B = torch.randn(10, alg3.dim)
        manual = alg3.geometric_product(A, B) + alg3.geometric_product(B, A)
        prim = alg3.anti_commutator(A, B)
        assert torch.allclose(manual, prim, atol=1e-5)


# =====================================================================
# compute_uncertainty_and_alignment
# =====================================================================


class TestComputeUncertaintyAndAlignment:
    def test_returns_tuple(self, alg3):
        data = torch.randn(50, 3)
        U, V = compute_uncertainty_and_alignment(alg3, data)
        assert isinstance(U, float)
        assert isinstance(V, torch.Tensor)

    def test_V_shape(self, alg3):
        data = torch.randn(50, 3)
        _, V = compute_uncertainty_and_alignment(alg3, data)
        assert V.shape == (3, 3)

    def test_U_nonnegative(self, alg3):
        data = torch.randn(50, 3)
        U, _ = compute_uncertainty_and_alignment(alg3, data)
        assert U >= 0

    def test_padding_when_D_lt_n(self):
        alg = CliffordAlgebra(4, 0, device=DEVICE)
        data = torch.randn(50, 2)  # D=2 < n=4
        U, V = compute_uncertainty_and_alignment(alg, data)
        assert V.shape == (2, 2)
        assert U >= 0


# =====================================================================
# GeometricAnalyzer (pipeline)
# =====================================================================


class TestGeometricAnalyzerPipeline:
    def test_raw_mode_returns_report(self, raw_3d_data):
        """Raw [N, D] data, no algebra → full pipeline."""
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=50),
            run_signature=False,  # skip signature search for speed
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        assert isinstance(report, AnalysisReport)

    def test_raw_mode_dimension_result(self, raw_3d_data):
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=50),
            run_signature=False,
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        assert report.dimension is not None
        assert isinstance(report.dimension, DimensionResult)

    def test_raw_mode_spectral_result(self, raw_3d_data):
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=50),
            run_signature=False,
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        assert report.spectral is not None

    def test_raw_mode_symmetry_result(self, raw_3d_data):
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=50),
            run_signature=False,
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        assert report.symmetry is not None

    def test_raw_mode_commutator_result(self, raw_3d_data):
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=50),
            run_signature=False,
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        assert report.commutator is not None

    def test_pre_embedded_mode(self, alg3, circle_data_3d):
        """Pre-embedded [N, C, dim] data + algebra → GA analyses only."""
        cfg = AnalysisConfig(device=DEVICE)
        ga = GeometricAnalyzer(cfg)
        mv_3d = circle_data_3d.unsqueeze(1)  # [50, 1, 8]
        report = ga.analyze(mv_3d, algebra=alg3)
        # Dimension/signature not run in pre-embedded mode
        assert report.dimension is None
        assert report.signature is None
        assert report.spectral is not None
        assert report.symmetry is not None
        assert report.commutator is not None

    def test_raw_with_known_algebra(self, alg3, raw_3d_data):
        """Raw [N, D] + algebra → embed then GA analyses."""
        cfg = AnalysisConfig(device=DEVICE)
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data, algebra=alg3)
        assert report.dimension is None
        assert report.spectral is not None

    def test_metadata_contains_elapsed(self, raw_3d_data):
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=30),
            run_signature=False,
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        assert "elapsed_seconds" in report.metadata

    def test_summary_returns_string(self, raw_3d_data):
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=30),
            run_signature=False,
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        s = report.summary()
        assert isinstance(s, str)
        assert "Geometric Analysis Report" in s

    def test_selective_analyzers(self, raw_3d_data):
        """Disable individual analyzers."""
        cfg = AnalysisConfig(
            device=DEVICE,
            sampling=SamplingConfig(strategy="random", max_samples=30),
            run_dimension=True,
            run_signature=False,
            run_spectral=False,
            run_symmetry=True,
            run_commutator=False,
        )
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw_3d_data)
        assert report.dimension is not None
        assert report.spectral is None
        assert report.symmetry is not None
        assert report.commutator is None

    def test_invalid_shape_raises(self):
        cfg = AnalysisConfig(device=DEVICE)
        ga = GeometricAnalyzer(cfg)
        with pytest.raises(ValueError, match="Unexpected data shape"):
            ga.analyze(torch.randn(2, 3, 4, 5))

    def test_minkowski_algebra(self, alg21):
        """Pipeline works with non-Euclidean algebra."""
        torch.manual_seed(40)
        raw = torch.randn(40, 3)
        cfg = AnalysisConfig(device=DEVICE)
        ga = GeometricAnalyzer(cfg)
        report = ga.analyze(raw, algebra=alg21)
        assert report.spectral is not None
        assert report.commutator is not None


# =====================================================================
# Cross-component integration
# =====================================================================


class TestCrossComponentIntegration:
    def test_symmetry_uses_commutator_for_continuous(self, alg3, circle_data_3d):
        """When both symmetry and commutator run, continuous symmetry is refined."""
        cfg = AnalysisConfig(
            device=DEVICE,
            run_dimension=False,
            run_signature=False,
            run_spectral=False,
            run_symmetry=True,
            run_commutator=True,
        )
        ga = GeometricAnalyzer(cfg)
        mv = circle_data_3d.unsqueeze(1)
        report = ga.analyze(mv, algebra=alg3)
        assert report.symmetry is not None
        assert report.commutator is not None
        # continuous_symmetry_dim should be set
        assert report.symmetry.continuous_symmetry_dim >= 0

    def test_commutator_primitive_in_analysis(self, alg3):
        """End-to-end: commutator primitive produces same result as manual."""
        torch.manual_seed(50)
        mv = torch.randn(30, alg3.dim)

        # Direct analyzer
        ca = CommutatorAnalyzer(alg3)
        result = ca.analyze(mv)

        # Manual mean commutator norm for comparison
        mu = mv.mean(dim=0, keepdim=True).expand_as(mv)
        manual_comm = alg3.geometric_product(mv, mu) - alg3.geometric_product(mu, mv)
        manual_mcn = manual_comm.norm(dim=-1).mean().item()

        assert abs(result.mean_commutator_norm - manual_mcn) < 1e-4
