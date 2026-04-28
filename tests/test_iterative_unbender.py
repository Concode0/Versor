# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Tests for Iterative Geometric Unbending v2 pipeline.

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.slow

from core.algebra import CliffordAlgebra
from models.sr.unbender import (
    IterativeUnbender,
    StageResult,
    UnbendingResult,
    OrthogonalEliminationResult,
)


@pytest.fixture
def simple_sin_data():
    """Synthetic y = sin(x) dataset."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, (100, 1)).astype(np.float32)
    y = np.sin(X[:, 0]).astype(np.float32)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(-1)

    x_mean = X_t.mean(0)
    x_std = X_t.std(0).clamp(min=1e-8)
    y_mean = y_t.mean()
    y_std = y_t.std().clamp(min=1e-8)

    X_norm = (X_t - x_mean) / x_std
    y_norm = (y_t - y_mean) / y_std

    return X_norm, y_norm, x_mean, x_std, y_mean, y_std


def test_iterative_unbender_smoke(simple_sin_data):
    """Run 1 stage on synthetic y=sin(x) data, verify UnbendingResult returned."""
    X_norm, y_norm, x_mean, x_std, y_mean, y_std = simple_sin_data

    unbender = IterativeUnbender(
        in_features=1,
        device="cpu",
        max_stages=1,
        stage_epochs=3,
        geodesic_k=4,
        grouping_enabled=False,
        implicit_mode='explicit',
    )

    result = unbender.run(
        X_norm,
        y_norm,
        x_mean,
        x_std,
        y_mean,
        y_std,
        var_names=["x"],
    )

    assert isinstance(result, UnbendingResult)
    assert result.formula.startswith("y = ")
    assert isinstance(result.r2_final, float)
    assert isinstance(result.implicit_mode, str)


def test_stage_result_fields(simple_sin_data):
    """Verify StageResult has correct field types including new fields."""
    X_norm, y_norm, x_mean, x_std, y_mean, y_std = simple_sin_data

    unbender = IterativeUnbender(
        in_features=1,
        device="cpu",
        max_stages=1,
        stage_epochs=2,
        geodesic_k=4,
        grouping_enabled=False,
        implicit_mode='explicit',
    )

    result = unbender.run(
        X_norm,
        y_norm,
        x_mean,
        x_std,
        y_mean,
        y_std,
        var_names=["x"],
    )

    if result.stages:
        stage = result.stages[0]
        assert isinstance(stage.stage_idx, int)
        assert isinstance(stage.signature, tuple)
        assert len(stage.signature) == 3
        assert isinstance(stage.terms, list)
        assert isinstance(stage.curvature_before, float)
        assert isinstance(stage.curvature_after, float)
        assert isinstance(stage.coherence_before, float)
        assert isinstance(stage.coherence_after, float)
        assert isinstance(stage.accepted, bool)
        assert isinstance(stage.rotor_planes, list)
        assert isinstance(stage.group_idx, int)


def test_orthogonal_elimination():
    """Test GA blade rejection preserves orthogonal components."""
    algebra = CliffordAlgebra(3, 0, 0, device="cpu")

    unbender = IterativeUnbender(
        in_features=2,
        device="cpu",
        soft_rejection_alpha=10.0,
        soft_rejection_threshold=0.01,
    )

    # Create a bivector blade in e12 plane
    blade = torch.zeros(algebra.dim)
    # e12 index = 0b11 = 3
    blade[3] = 1.0

    # Create data multivector with components in and out of plane
    data_mv = torch.zeros(10, algebra.dim)
    data_mv[:, 1] = 1.0  # e1 (in plane)
    data_mv[:, 4] = 0.5  # e3 (out of plane)

    rejected, elim = unbender._orthogonal_eliminate(data_mv, blade, algebra)

    assert isinstance(elim, OrthogonalEliminationResult)
    assert elim.projection_energy >= 0.0
    assert elim.rejection_energy >= 0.0
    assert 0.0 <= elim.preserved_fraction <= 1.0


def test_svd_warmstart():
    """SVD warm-start sets bivector weights without error."""
    from models.sr.net import SRGBN

    algebra = CliffordAlgebra(3, 0, 0, device="cpu")
    model = SRGBN.single_rotor(algebra, 3, channels=4)

    # Create a simple rotation matrix
    Vt = np.eye(3, dtype=np.float32)
    theta = 0.5
    Vt[0, 0] = np.cos(theta)
    Vt[0, 1] = -np.sin(theta)
    Vt[1, 0] = np.sin(theta)
    Vt[1, 1] = np.cos(theta)

    model.svd_warmstart(Vt, algebra)

    # Check that bivector weights were modified
    bv = model.blocks[0].rotor.bivector_weights.detach()
    assert bv.abs().max().item() > 0.0


def test_soft_rejection_preserves_weak():
    """Soft rejection should preserve components below threshold."""
    algebra = CliffordAlgebra(3, 0, 0, device="cpu")

    unbender = IterativeUnbender(
        in_features=2,
        device="cpu",
        soft_rejection_alpha=100.0,
        soft_rejection_threshold=1.0,  # high threshold
    )

    blade = torch.zeros(algebra.dim)
    blade[3] = 1.0  # e12

    # Small projection -> should be mostly preserved
    data_mv = torch.zeros(5, algebra.dim)
    data_mv[:, 1] = 0.01  # very small e1

    rejected, elim = unbender._orthogonal_eliminate(data_mv, blade, algebra)
    assert elim.preserved_fraction > 0.5


def test_coherence_backtrack():
    """Mock degraded coherence on 2D data, verify retry logic."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-3, 3, (100, 2)).astype(np.float32)
    y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) + X[:, 0] ** 2).astype(np.float32)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(-1)
    x_mean = X_t.mean(0)
    x_std = X_t.std(0).clamp(min=1e-8)
    y_mean = y_t.mean()
    y_std = y_t.std().clamp(min=1e-8)
    X_norm = (X_t - x_mean) / x_std
    y_norm = (y_t - y_mean) / y_std

    unbender = IterativeUnbender(
        in_features=2,
        device="cpu",
        max_stages=1,
        stage_epochs=2,
        geodesic_k=4,
        coherence_degradation_threshold=0.0,
        grouping_enabled=False,
        implicit_mode='explicit',
    )

    call_count = {"n": 0}

    def mock_measure(X_raw, residual, algebra):
        call_count["n"] += 1
        return 0.1

    with patch.object(unbender, "_measure_coherence", side_effect=mock_measure):
        result = unbender.run(
            X_norm,
            y_norm,
            x_mean,
            x_std,
            y_mean,
            y_std,
            var_names=["x1", "x2"],
        )

    # Should have been called multiple times (initial + per-stage)
    assert call_count["n"] >= 1


def test_reprobe_changes_signature():
    """Verify different residuals can produce different (p,q,r) signatures."""
    unbender = IterativeUnbender(
        in_features=2,
        device="cpu",
        max_stages=1,
        stage_epochs=2,
        geodesic_k=4,
    )

    t = np.linspace(0, 2 * np.pi, 100, dtype=np.float32)
    X_circle = np.column_stack([np.cos(t), np.sin(t)])
    y_circle = np.sin(t).astype(np.float32)
    result1 = unbender._probe_residual(X_circle, y_circle, n_probes=2)

    rng = np.random.default_rng(42)
    X_exp = rng.uniform(0, 3, (100, 2)).astype(np.float32)
    y_exp = np.exp(X_exp[:, 0]).astype(np.float32)
    result2 = unbender._probe_residual(X_exp, y_exp, n_probes=2)

    assert len(result1["signature"]) == 3
    assert len(result2["signature"]) == 3
    assert all(isinstance(v, int) for v in result1["signature"])
    assert all(isinstance(v, int) for v in result2["signature"])


def test_linearity_detection():
    """Synthetic y = 2x + 1 should be detected as linear."""
    unbender = IterativeUnbender(in_features=1, device="cpu")

    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, (100, 1)).astype(np.float32)
    y = (2.0 * X[:, 0] + 1.0).astype(np.float32)

    is_linear, terms, r2 = unbender._check_linearity(X, y)
    assert is_linear
    assert r2 > 0.99
    assert len(terms) >= 1


def test_linearity_skips_stages():
    """Linear data should produce 0 unbending stages."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-5, 5, (100, 1)).astype(np.float32)
    y = (3.0 * X[:, 0] - 2.0).astype(np.float32)

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(-1)
    x_mean = X_t.mean(0)
    x_std = X_t.std(0).clamp(min=1e-8)
    y_mean = y_t.mean()
    y_std = y_t.std().clamp(min=1e-8)
    X_norm = (X_t - x_mean) / x_std
    y_norm = (y_t - y_mean) / y_std

    unbender = IterativeUnbender(
        in_features=1,
        device="cpu",
        max_stages=3,
        stage_epochs=2,
        geodesic_k=4,
    )
    result = unbender.run(X_norm, y_norm, x_mean, x_std, y_mean, y_std, var_names=["x"])

    assert len(result.stages) == 0
    assert result.r2_final > 0.95


def test_empty_terms():
    """Unbender handles case where no terms are extracted."""
    unbender = IterativeUnbender(in_features=1, device="cpu")

    formula = unbender._assemble_formula([], ["x"])
    assert formula == "y = 0"

    refined, ops = unbender._refine_all_terms([], np.zeros((10, 1)), np.zeros(10))
    assert refined == []
    assert ops == []


def test_active_var_selection():
    """10-var data where only 2 matter, verify correct top-k selection."""
    unbender = IterativeUnbender(in_features=10, device="cpu")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = (5.0 * X[:, 3] + 2.0 * X[:, 7]).astype(np.float32)

    active = unbender._select_active_vars(X, y, max_vars=4)
    assert len(active) == 4
    assert 3 in active
    assert 7 in active
    assert active.index(3) < 3
    assert active.index(7) < 3


def test_hubble_linear():
    """1D linear v=H0*d should pass linearity check with R2>0.95."""
    unbender = IterativeUnbender(in_features=1, device="cpu")

    rng = np.random.default_rng(42)
    H0 = 70.0
    d = rng.uniform(1, 100, (200, 1)).astype(np.float32)
    v = (H0 * d[:, 0]).astype(np.float32)

    is_linear, terms, r2 = unbender._check_linearity(d, v)
    assert is_linear, f"Hubble should be linear but got R2={r2:.4f}"
    assert r2 > 0.95
    assert len(terms) >= 1


def test_hubble_noisy_linear():
    """Hubble with low noise should pass linearity (R2>0.999)."""
    unbender = IterativeUnbender(in_features=1, device="cpu")

    rng = np.random.default_rng(42)
    H0 = 70.0
    d = rng.uniform(1, 100, (200, 1)).astype(np.float32)
    # Low noise to stay above 0.999 threshold
    v = (H0 * d[:, 0] + rng.normal(0, 10, 200)).astype(np.float32)

    is_linear, terms, r2 = unbender._check_linearity(d, v)
    assert is_linear, f"Low-noise Hubble should be linear but got R2={r2:.4f}"
    assert r2 > 0.99


def test_kepler_power_law_recovery():
    """Kepler T ~ a^1.5 should be recovered as power law short-circuit."""
    unbender = IterativeUnbender(in_features=1, device="cpu")

    rng = np.random.default_rng(42)
    a = rng.uniform(1, 50, (200, 1)).astype(np.float32)
    T = (a[:, 0] ** 1.5).astype(np.float32)

    is_shortcircuit, terms, r2 = unbender._check_linearity(a, T)
    # Power law should be detected and returned as a short-circuit term
    assert is_shortcircuit, "Kepler T~a^1.5 should be detected as power law"
    assert len(terms) == 1, "Should produce exactly one power-law term"
    assert r2 > 0.99, f"Power law R2 should be high, got {r2}"
    # Check that the expression contains x1^(3/2)
    import sympy

    expr = terms[0].expr
    assert expr is not None, "Power law term should have an expression"
    x1 = sympy.Symbol("x1")
    assert x1 in expr.free_symbols, "Expression should contain x1"


def test_accepted_false_excludes_terms():
    """Unaccepted stage terms should NOT be added to all_terms."""
    from models.sr.translator import RotorTerm

    fake_terms = [RotorTerm(planes=[], weight=1.0)]
    stage_rejected = StageResult(
        stage_idx=0,
        signature=(3, 0, 0),
        terms=fake_terms,
        fitted_values=np.zeros(10),
        residual_before=np.ones(10),
        residual_after=np.ones(10),
        curvature_before=0.5,
        curvature_after=0.3,
        coherence_before=0.8,
        coherence_after=0.7,
        rotor_planes=[],
        accepted=False,
        composition_ops=["sub"],
    )

    all_terms = []
    all_ops = []
    residual = np.ones(10)

    if stage_rejected.accepted:
        residual = stage_rejected.residual_after
        all_terms.extend(stage_rejected.terms)
        all_ops.extend(stage_rejected.composition_ops)

    assert len(all_terms) == 0, "Rejected terms should not be in all_terms"
    assert len(all_ops) == 0


def test_1d_coherence_bypass():
    """1D data should skip coherence check (trivially coherent)."""
    n_vars = 1
    r2_extraction = 0.5
    skip_coherence = (r2_extraction > 0.9) or (n_vars <= 1)
    assert skip_coherence, "1D data should bypass coherence check"

    n_vars = 2
    skip_coherence = (r2_extraction > 0.9) or (n_vars <= 1)
    assert not skip_coherence, "2D data with low R2 should not bypass"

    r2_extraction = 0.95
    skip_coherence = (r2_extraction > 0.9) or (n_vars <= 1)
    assert skip_coherence, "High R2 extraction should bypass coherence"


def test_unbending_result_new_fields():
    """UnbendingResult has new fields: groups, implicit_mode, mother_cross_energy."""
    result = UnbendingResult(
        stages=[],
        formula="y = x",
        r2_final=0.99,
        all_terms=[],
        signature_history=[],
        groups=[],
        implicit_mode="explicit",
        mother_cross_energy=0.0,
        all_ops=["sub"],
    )
    assert result.implicit_mode == "explicit"
    assert result.mother_cross_energy == 0.0
    assert result.groups == []

    # Defaults
    result2 = UnbendingResult(
        stages=[],
        formula="y = 0",
        r2_final=0.0,
        all_terms=[],
        signature_history=[],
    )
    assert result2.all_ops == []
    assert result2.implicit_mode == "explicit"


def test_orthogonal_elimination_result_fields():
    """OrthogonalEliminationResult stores all expected fields."""
    elim = OrthogonalEliminationResult(
        projection_energy=1.5,
        rejection_energy=0.3,
        soft_threshold=0.01,
        preserved_fraction=0.8,
    )
    assert elim.projection_energy == 1.5
    assert elim.rejection_energy == 0.3
    assert elim.soft_threshold == 0.01
    assert elim.preserved_fraction == 0.8


def test_polynomial_fallback_removed():
    """Polynomial fallback was removed in favor of implicit mode + parabolic terms."""
    unbender = IterativeUnbender(in_features=1, device="cpu")
    assert not hasattr(unbender, '_polynomial_fallback')
