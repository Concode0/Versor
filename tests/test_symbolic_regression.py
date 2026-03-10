# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Tests for Symbolic Regression (SRBench / PMLB) Task.

import pytest
import numpy as np
import torch
from omegaconf import OmegaConf

from core.algebra import CliffordAlgebra
from datalib.symbolic_regression import (
    SRDataset,
    get_sr_loaders,
    get_dataset_ids,
    _fetch_pmlb_data,
    FIRST_PRINCIPLES_DATASETS,
    BLACKBOX_DATASETS,
    SRBENCH_DATASETS,
)
from models.sr.net import (
    SRMultiGradeEmbedding,
    SRGBN,
    _blade_name,
    blade_names_for_algebra,
)


@pytest.fixture(scope="module")
def algebra():
    return CliffordAlgebra(p=4, q=0, device="cpu")


@pytest.fixture(scope="module")
def small_algebra():
    return CliffordAlgebra(p=3, q=0, device="cpu")


# Use a blackbox dataset that's in the PyPI index for fast offline tests
_TEST_DATASET = "192_vineyard"
_TEST_CACHE = "./data/pmlb_cache"


def _make_cfg(dataset_name=_TEST_DATASET, hidden_channels=4, num_layers=1,
              n_samples=200, auto=False):
    return OmegaConf.create({
        "name": "sr",
        "algebra": {"p": 4, "q": 0, "r": 0, "device": "cpu", "auto": auto},
        "dataset": {
            "dataset_name": dataset_name,
            "category": "blackbox",
            "n_samples": n_samples,
            "noise": 0.0,
            "cache_dir": _TEST_CACHE,
        },
        "model": {
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "use_decomposition": False,
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "batch_size": 16,
            "optimizer_type": "riemannian_adam",
            "max_bivector_norm": 10.0,
            "sparsity_weight": 0.01,
            "seed": 0,
        },
        "checkpoint": None,
    })


def test_dataset_categories():
    """Built-in dataset lists are non-empty."""
    assert len(FIRST_PRINCIPLES_DATASETS) > 0
    assert len(BLACKBOX_DATASETS) > 0
    assert all(d.startswith("first_principles_") for d in FIRST_PRINCIPLES_DATASETS)


def test_get_dataset_ids():
    """get_dataset_ids returns correct lists for all categories."""
    fp_ids = get_dataset_ids("first_principles")
    assert len(fp_ids) == 12

    bb_ids = get_dataset_ids("blackbox")
    assert len(bb_ids) == 12

    all_ids = get_dataset_ids("all")
    assert len(all_ids) == 24
    assert all_ids == SRBENCH_DATASETS


@pytest.mark.slow
@pytest.mark.integration
def test_pmlb_dataset_load():
    """Verify _fetch_pmlb_data returns valid data for a blackbox dataset."""
    df = _fetch_pmlb_data(_TEST_DATASET, _TEST_CACHE)
    assert "target" in df.columns
    assert len(df) > 0
    assert df.shape[1] >= 2


def test_sr_loaders():
    """get_sr_loaders returns normalized data with correct shapes."""
    train_loader, test_loader, x_mean, x_std, y_mean, y_std, var_names = \
        get_sr_loaders(
            dataset_name=_TEST_DATASET,
            n_samples=200,
            batch_size=32,
            cache_dir=_TEST_CACHE,
            seed=42,
            num_workers=0,
        )

    assert len(var_names) > 0
    assert x_mean.shape[0] == len(var_names)
    assert x_std.shape[0] == len(var_names)

    # Check normalization on train split
    x_all, y_all = [], []
    for x_b, y_b in train_loader:
        x_all.append(x_b)
        y_all.append(y_b)
    x_all = torch.cat(x_all)
    y_all = torch.cat(y_all)

    assert x_all.mean(0).abs().max().item() < 0.15, \
        "Normalised x mean should be near zero"
    assert y_all.mean().abs().item() < 0.15, \
        "Normalised y mean should be near zero"


def test_embedding_shape(algebra):
    """SRMultiGradeEmbedding outputs [B, C, 2^p]."""
    B, k, C = 8, 3, 6
    emb = SRMultiGradeEmbedding(algebra, in_features=k, channels=C)
    x = torch.randn(B, k)
    out = emb(x)
    assert out.shape == (B, C, algebra.dim), \
        f"Expected ({B}, {C}, {algebra.dim}), got {out.shape}"


def test_embedding_grade1_nonzero(algebra):
    """Grade-1 blade components are populated from non-trivial inputs."""
    B, k, C = 4, 3, 4
    emb = SRMultiGradeEmbedding(algebra, in_features=k, channels=C)
    x = torch.ones(B, k) * 10.0
    out = emb(x)

    g1_idx = [i for i in range(algebra.dim) if bin(i).count("1") == 1]
    g1_components = out[:, :, g1_idx]
    assert g1_components.abs().max().item() > 1e-7, \
        "Grade-1 components should be non-zero for non-zero input"


def test_embedding_grade2_zero(algebra):
    """Grade-2 components are zero (no LUT embedding)."""
    B, k, C = 4, 3, 4
    x = torch.ones(B, k) * 5.0

    g2_idx = [i for i in range(algebra.dim) if bin(i).count("1") == 2]
    emb = SRMultiGradeEmbedding(algebra, in_features=k, channels=C)
    out = emb(x)
    assert out[:, :, g2_idx].abs().max().item() == 0.0, \
        "Grade-2 components should be zero (no LUT)"


def test_model_forward_shape(algebra):
    """SRGBN returns [B, 1]."""
    B, k = 8, 3
    model = SRGBN(algebra, in_features=k, channels=4, num_layers=1)
    x = torch.randn(B, k)
    out = model(x)
    assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"


def test_model_gradient_flow(algebra):
    """loss.backward() completes without NaN gradients."""
    B, k = 6, 4
    model = SRGBN(algebra, in_features=k, channels=4, num_layers=1)
    x = torch.randn(B, k)
    y = torch.randn(B, 1)

    criterion = torch.nn.MSELoss()
    loss = criterion(model(x), y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), \
                f"NaN/Inf gradient in {name}"


def test_sparsity_loss(algebra):
    """total_sparsity_loss() returns a positive scalar tensor."""
    model = SRGBN(algebra, in_features=3, channels=4, num_layers=2)
    x = torch.randn(4, 3)
    _ = model(x)

    spl = model.total_sparsity_loss()
    assert spl.ndim == 0, "sparsity_loss should be a scalar tensor"
    assert spl.item() > 0.0, "sparsity_loss should be positive"
    assert torch.isfinite(spl), "sparsity_loss should be finite"


@pytest.mark.slow
def test_task_train_step():
    """Single train_step returns finite (loss, logs) dict."""
    from tasks.symbolic_regression import SRTask

    cfg = _make_cfg(n_samples=200, hidden_channels=4, num_layers=1)
    task = SRTask(cfg)
    train_loader, _ = task.get_data()

    task.model.train()
    batch = next(iter(train_loader))
    loss, logs = task.train_step(batch)

    assert np.isfinite(loss), f"Loss is not finite: {loss}"
    for key, val in logs.items():
        assert np.isfinite(val), f"Log '{key}' is not finite: {val}"
    assert "MSE" in logs
    assert "Sparsity" in logs
    assert "MAE" in logs


def test_blade_names(algebra):
    """blade_names_for_algebra returns correct names for Cl(4,0)."""
    names = blade_names_for_algebra(algebra)

    assert len(names) == 16, f"Expected 16 blade names, got {len(names)}"
    assert names[0] == "1", f"Scalar blade should be '1', got {names[0]!r}"
    assert names[3] == "e12", f"idx=3 should be 'e12', got {names[3]!r}"
    assert names[15] == "e1234", f"idx=15 should be 'e1234', got {names[15]!r}"

    g1_names = [names[i] for i in range(16) if bin(i).count("1") == 1]
    assert all(len(n) == 2 and n.startswith("e") for n in g1_names), \
        "Grade-1 blades should be 'e1', 'e2', 'e3', 'e4'"


def test_get_rotor_analysis(algebra):
    """get_rotor_analysis() returns one dict per layer with expected keys."""
    num_layers = 2
    model = SRGBN(
        algebra, in_features=3, channels=4,
        num_layers=num_layers,
    )
    x = torch.randn(8, 3)
    _ = model(x)

    analysis = model.get_rotor_analysis()

    assert len(analysis) == num_layers, \
        f"Expected {num_layers} layer dicts, got {len(analysis)}"

    expected_keys = {"layer", "bivectors", "plane_names", "dominant_plane"}
    for item in analysis:
        assert expected_keys.issubset(item.keys()), \
            f"Missing keys: {expected_keys - item.keys()}"


def test_variable_importance_shape():
    """SRTask.variable_importance() returns Tensor of shape [n_vars]."""
    from tasks.symbolic_regression import SRTask

    cfg = _make_cfg(n_samples=200, hidden_channels=4, num_layers=1)
    task = SRTask(cfg)

    x_batch = torch.randn(8, task.n_vars)
    imp = task.variable_importance(x_batch)

    assert imp.shape == (task.n_vars,), \
        f"Expected shape ({task.n_vars},), got {imp.shape}"
    assert torch.isfinite(imp).all(), "variable_importance contains NaN/Inf"


def test_structural_analysis(algebra):
    """structural_analysis returns valid max_degree, grade_energy, and active_vars."""
    model = SRGBN(algebra, in_features=3, channels=4, num_layers=1)
    x = torch.randn(32, 3)

    max_degree, grade_fracs, active_vars = model.structural_analysis(x)

    assert max_degree >= 1, "max_degree should be at least 1"
    assert max_degree <= 4, "max_degree should be capped at 4"
    assert abs(sum(grade_fracs) - 1.0) < 0.01, "grade energy fractions should sum to ~1.0"
    assert len(active_vars) >= 1, "should have at least one active variable"
    assert all(0 <= v < 3 for v in active_vars), "active vars should be valid indices"


def test_single_rotor_factory(algebra):
    """SRGBN.single_rotor creates a 1-block model."""
    model = SRGBN.single_rotor(algebra, in_features=3, channels=8)
    assert len(model.blocks) == 1
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 1)


def test_auto_config():
    """auto_config returns appropriate channels for different dataset sizes."""
    small = SRGBN.auto_config(10, 3, 16)
    assert small["channels"] == 4

    medium = SRGBN.auto_config(100, 3, 16)
    assert medium["channels"] >= 8

    large = SRGBN.auto_config(500, 3, 16)
    assert large["channels"] >= 16


def test_extract_formula_smoke():
    """extract_formula returns an UnbendingResult with formula string."""
    from tasks.symbolic_regression import SRTask

    cfg = _make_cfg(n_samples=50, hidden_channels=4, num_layers=1)
    # Add iterative config for extract_formula
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "iterative": {
            "max_stages": 1,
            "stage_epochs": 5,
            "r2_target": 0.999,
        },
        "implicit": {"mode": "explicit"},
        "grouping": {"enabled": False},
        "svd": {"warmstart": False},
        "rejection": {"soft_alpha": 10.0, "soft_threshold": 0.01},
        "mother_algebra": {"cross_term_threshold": 0.01},
    }))
    task = SRTask(cfg)
    train_loader, test_loader = task.get_data()

    task.model.train()
    batch = next(iter(train_loader))
    task.train_step(batch)

    result = task.extract_formula(test_loader)

    assert hasattr(result, 'formula')
    assert isinstance(result.formula, str)
    assert hasattr(result, 'r2_final')
    assert np.isfinite(result.r2_final)
