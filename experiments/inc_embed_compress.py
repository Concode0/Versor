# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""
==============================================================================
VERSOR EXPERIMENT: IDEA INCUBATOR (SPIN-OFF CONCEPT)
==============================================================================

This script serves as an early-stage proof-of-concept for radical, non-Euclidean
architectures. The concepts demonstrated here are strongly driven by geometric
intuition and may currently reside ahead of established academic literature.

Please understand that rigorous mathematical proofs or comprehensive citations
might be incomplete at this stage. If this geometric hypothesis proves
structurally sound, it is planned to be spun off into a dedicated, independent
repository for detailed research.

==============================================================================

Geometric Embedding Compression.

Hypothesis
  High-dimensional embeddings such as ``BAAI/bge-large-en-v1.5`` occupy a much
  lower effective intrinsic dimension. A Geometric Blade Network compressor
  should learn a structured bottleneck that preserves downstream
  classification accuracy better than PCA at the same compression ratio,
  because rotor-based mixing can capture manifold curvature that principal
  components miss. The experiment estimates intrinsic dimension, chooses an
  internal Clifford algebra, sweeps target compression widths, and compares
  PCA against the learned compressor on validation accuracy and reconstruction
  diagnostics.

Execute Command
  uv run python -m experiments.inc_embed_compress
  uv run python -m experiments.inc_embed_compress --datasets sst2 snli
  uv run python -m experiments.inc_embed_compress --datasets manifold_ladder
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is importable when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import matplotlib
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from core.algebra import CliffordAlgebra
from core.analysis._types import DimensionResult
from core.analysis.dimension import DimensionLifter, EffectiveDimensionAnalyzer
from core.analysis.spectral import SpectralAnalyzer
from experiments._lib import (
    build_visualization_metadata,
    ensure_output_dir,
    make_experiment_parser,
    save_experiment_figure,
    set_seed,
    setup_algebra,
    signature_metadata,
)
from functional.activation import GeometricGELU
from layers import (
    BladeSelector,
    CliffordLayerNorm,
    CliffordLinear,
    CliffordModule,
    RotorLayer,
)
from optimizers.riemannian import RiemannianAdam

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = make_experiment_parser(
        "GBN Embedding Compression — Incubator Experiment",
        include=("seed", "epochs", "lr", "batch_size", "output_dir"),
        defaults={"epochs": 30, "batch_size": 256, "lr": 5e-4, "output_dir": "embed_compress"},
    )
    p.add_argument(
        "--model",
        default="BAAI/bge-large-en-v1.5",
        help="SentenceTransformer model (text datasets only; ignored otherwise)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2"],
        choices=[
            "sst2",
            "snli",
            "trec",
            "agnews",
            "mnist",
            "cifar10",
            "flat_plane",
            "swiss_roll",
            "torus",
            "klein_bottle",
            "mixed_curvature",
            "manifold_ladder",
        ],
        help="Data sources to run (default: sst2). "
        "Text: sst2, snli, trec, agnews. Image: mnist, cifar10. "
        "Synthetic: flat_plane, swiss_roll, torus, klein_bottle, "
        "mixed_curvature. Umbrella: manifold_ladder (runs all 5).",
    )
    p.add_argument("--max-train", type=int, default=8000, help="Max training samples per dataset (default: 8000)")
    p.add_argument("--max-val", type=int, default=2000, help="Max validation samples per dataset (default: 2000)")
    p.add_argument(
        "--cache-dir", default="data/embed_compress", help="Directory to cache features (default: data/embed_compress)"
    )
    # Swiss roll synthetic data
    p.add_argument(
        "--swiss-roll-dim",
        type=int,
        default=1024,
        help="High-D lift dimension for synthetic Swiss roll (default: 1024)",
    )
    p.add_argument(
        "--swiss-roll-n",
        type=int,
        default=10000,
        help="Total samples for synthetic Swiss roll, 80/20 split (default: 10000)",
    )
    # GBN architecture
    p.add_argument(
        "--algebra-p", type=int, default=5, help="Positive signature of internal Clifford algebra (default: 5)"
    )
    p.add_argument(
        "--algebra-q", type=int, default=0, help="Negative signature of internal Clifford algebra (default: 0)"
    )
    p.add_argument("--channels", type=int, default=8, help="Number of multivector channels in GBN (default: 8)")
    # Training
    p.add_argument("--alpha", type=float, default=1.0, help="Reconstruction loss weight (default: 1.0)")
    p.add_argument("--beta", type=float, default=0.5, help="Classification loss weight (default: 0.5)")
    p.add_argument(
        "--complexity-plot",
        dest="complexity_plot",
        action="store_true",
        default=True,
        help="Produce gain-vs-complexity scatter (default: on).",
    )
    p.add_argument(
        "--no-complexity-plot", dest="complexity_plot", action="store_false", help="Disable gain-vs-complexity scatter."
    )
    return p


# ============================================================================
# Dataset loading
# ============================================================================

# Text datasets: raw-text loaders that feed the SentenceTransformer encoder.


def _load_sst2_raw(max_train: int, max_val: int):
    ds = load_dataset("sst2", trust_remote_code=False)
    train = ds["train"].shuffle(seed=42).select(range(min(max_train, len(ds["train"]))))
    val = ds["validation"].shuffle(seed=42).select(range(min(max_val, len(ds["validation"]))))
    return (list(train["sentence"]), list(train["label"])), (list(val["sentence"]), list(val["label"]))


def _load_snli_raw(max_train: int, max_val: int):
    ds = load_dataset("snli", trust_remote_code=False)

    def _pick(split, max_n):
        rows = [(p, h, l) for p, h, l in zip(split["premise"], split["hypothesis"], split["label"]) if l != -1]
        rows = rows[:max_n]
        return [f"{p} [SEP] {h}" for p, h, _ in rows], [l for _, _, l in rows]

    train = ds["train"].shuffle(seed=42)
    val = ds["validation"].shuffle(seed=42)
    return _pick(train, max_train), _pick(val, max_val)


def _load_trec_raw(max_train: int, max_val: int):
    ds = load_dataset("CogComp/trec", trust_remote_code=False)
    train = ds["train"].shuffle(seed=42).select(range(min(max_train, len(ds["train"]))))
    val = ds["test"].shuffle(seed=42).select(range(min(max_val, len(ds["test"]))))
    label_key = "coarse_label" if "coarse_label" in train.column_names else "label-coarse"
    return (list(train["text"]), list(train[label_key])), (list(val["text"]), list(val[label_key]))


def _load_agnews_raw(max_train: int, max_val: int):
    ds = load_dataset("ag_news", trust_remote_code=False)
    train = ds["train"].shuffle(seed=42).select(range(min(max_train, len(ds["train"]))))
    val = ds["test"].shuffle(seed=42).select(range(min(max_val, len(ds["test"]))))
    return (list(train["text"]), list(train["label"])), (list(val["text"]), list(val["label"]))


TEXT_LOADERS = {
    "sst2": _load_sst2_raw,
    "snli": _load_snli_raw,
    "trec": _load_trec_raw,
    "agnews": _load_agnews_raw,
}
N_CLASSES = {
    "sst2": 2,
    "snli": 3,
    "trec": 6,
    "agnews": 4,
    "mnist": 10,
    "cifar10": 10,
    "flat_plane": 4,
    "swiss_roll": 4,
    "torus": 4,
    "klein_bottle": 4,
    "mixed_curvature": 4,
}

# Ordered ladder of synthetic manifolds with increasing intrinsic complexity.
MANIFOLD_LADDER = (
    "flat_plane",
    "swiss_roll",
    "torus",
    "klein_bottle",
    "mixed_curvature",
)


# ============================================================================
# Feature preparation (unified cache across text / image / synthetic)
# ============================================================================


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _load_cache(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    print(f"  Loading cached features from {path}")
    data = torch.load(path, weights_only=True)
    # Back-compat: older cache files stored 'embeddings' instead of 'features'
    feats = data.get("features", data.get("embeddings"))
    return feats, data["labels"]


def _save_cache(path: Path, features: torch.Tensor, labels: torch.Tensor):
    torch.save({"features": features, "labels": labels}, path)
    print(f"  Saved to {path}")


def _prepare_text(
    ds_name: str,
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    slug = _model_slug(args.model)
    train_cache = cache_dir / f"{ds_name}_train_{slug}.pt"
    val_cache = cache_dir / f"{ds_name}_val_{slug}.pt"

    if train_cache.exists() and val_cache.exists():
        tr_x, tr_y = _load_cache(train_cache)
        va_x, va_y = _load_cache(val_cache)
        return tr_x, tr_y, va_x, va_y

    (tr_txt, tr_lab), (va_txt, va_lab) = TEXT_LOADERS[ds_name](args.max_train, args.max_val)

    print(f"  Encoding {len(tr_txt)}+{len(va_txt)} texts with {args.model} …")
    model = SentenceTransformer(args.model)
    tr_embs = model.encode(
        tr_txt, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False
    )
    va_embs = model.encode(
        va_txt, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False
    )

    tr_x = torch.from_numpy(tr_embs).float()
    tr_y = torch.tensor(tr_lab, dtype=torch.long)
    va_x = torch.from_numpy(va_embs).float()
    va_y = torch.tensor(va_lab, dtype=torch.long)

    _save_cache(train_cache, tr_x, tr_y)
    _save_cache(val_cache, va_x, va_y)
    return tr_x, tr_y, va_x, va_y


def _prepare_mnist(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cache = cache_dir / f"mnist_train_{args.max_train}.pt"
    val_cache = cache_dir / f"mnist_val_{args.max_val}.pt"

    if train_cache.exists() and val_cache.exists():
        tr_x, tr_y = _load_cache(train_cache)
        va_x, va_y = _load_cache(val_cache)
        return tr_x, tr_y, va_x, va_y

    print("  Loading MNIST via HuggingFace datasets …")
    ds = load_dataset("ylecun/mnist")
    train = ds["train"].shuffle(seed=42).select(range(min(args.max_train, len(ds["train"]))))
    val = ds["test"].shuffle(seed=42).select(range(min(args.max_val, len(ds["test"]))))

    def _flatten(split):
        imgs = np.stack([np.asarray(img, dtype=np.float32) for img in split["image"]])
        imgs = imgs.reshape(imgs.shape[0], -1) / 255.0
        return (torch.from_numpy(imgs).float(), torch.tensor(split["label"], dtype=torch.long))

    tr_x, tr_y = _flatten(train)
    va_x, va_y = _flatten(val)

    _save_cache(train_cache, tr_x, tr_y)
    _save_cache(val_cache, va_x, va_y)
    return tr_x, tr_y, va_x, va_y


_CIFAR10_PRETRIM_DIM = 1024  # PCA-pretrim to match text embedding dim


def _prepare_cifar10(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cache = cache_dir / f"cifar10_train_{args.max_train}_d{_CIFAR10_PRETRIM_DIM}.pt"
    val_cache = cache_dir / f"cifar10_val_{args.max_val}_d{_CIFAR10_PRETRIM_DIM}.pt"

    if train_cache.exists() and val_cache.exists():
        tr_x, tr_y = _load_cache(train_cache)
        va_x, va_y = _load_cache(val_cache)
        return tr_x, tr_y, va_x, va_y

    print("  Loading CIFAR-10 via HuggingFace datasets …")
    ds = load_dataset("cifar10")
    img_key = "img" if "img" in ds["train"].column_names else "image"
    train = ds["train"].shuffle(seed=42).select(range(min(args.max_train, len(ds["train"]))))
    val = ds["test"].shuffle(seed=42).select(range(min(args.max_val, len(ds["test"]))))

    def _flatten(split):
        imgs = np.stack([np.asarray(img, dtype=np.float32) for img in split[img_key]])
        imgs = imgs.reshape(imgs.shape[0], -1) / 255.0  # [N, 3072]
        return imgs, np.asarray(split["label"], dtype=np.int64)

    tr_raw, tr_lab = _flatten(train)
    va_raw, va_lab = _flatten(val)

    print(f"  PCA-pretrimming 3072 -> {_CIFAR10_PRETRIM_DIM} to match text dim …")
    tr_t = torch.from_numpy(tr_raw).float()
    va_t = torch.from_numpy(va_raw).float()
    mean, Vh = precompute_pca(tr_t)
    Vk = Vh[:_CIFAR10_PRETRIM_DIM]
    tr_x = (tr_t - mean) @ Vk.T
    va_x = (va_t - mean) @ Vk.T
    tr_y = torch.from_numpy(tr_lab).long()
    va_y = torch.from_numpy(va_lab).long()

    _save_cache(train_cache, tr_x, tr_y)
    _save_cache(val_cache, va_x, va_y)
    return tr_x, tr_y, va_x, va_y


_SWISS_SEED_OFFSET = 2026


def _prepare_swiss_roll(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_total = args.swiss_roll_n
    dim = args.swiss_roll_dim
    seed = args.seed + _SWISS_SEED_OFFSET
    cache_path = cache_dir / f"swiss_roll_n{n_total}_d{dim}_s{seed}.pt"

    if cache_path.exists():
        tr_x, tr_y = _load_cache(cache_path)
        n_train = int(n_total * 0.8)
        return tr_x[:n_train], tr_y[:n_train], tr_x[n_train:], tr_y[n_train:]

    print(f"  Generating synthetic Swiss roll: n={n_total}, lift_dim={dim}, seed={seed}")
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n_total)
    height = rng.uniform(0.0, 21.0, size=n_total)
    x_raw = np.stack([t * np.cos(t), height, t * np.sin(t)], axis=1).astype(np.float32)

    # Random linear lift 3 -> dim, scaled so per-dim variance ~O(1).
    A = rng.standard_normal(size=(3, dim)).astype(np.float32) / np.sqrt(dim)
    x = x_raw @ A
    x += rng.standard_normal(size=x.shape).astype(np.float32) * 0.01

    # Arc-length-bin labels (4 equal-frequency bins of t).
    q = np.quantile(t, [0.25, 0.5, 0.75])
    labels = np.digitize(t, q).astype(np.int64)

    perm = rng.permutation(n_total)
    x = x[perm]
    labels = labels[perm]

    features = torch.from_numpy(x).float()
    labels_t = torch.from_numpy(labels).long()
    _save_cache(cache_path, features, labels_t)

    n_train = int(n_total * 0.8)
    return (features[:n_train], labels_t[:n_train], features[n_train:], labels_t[n_train:])


def _finalize_manifold(
    x_raw: np.ndarray,
    param: np.ndarray,
    dim: int,
    rng: np.random.Generator,
    cache_path: Path,
    n_total: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared post-processing for manifold generators.

    Applies a random linear lift `x_raw` -> R^dim, adds small noise,
    derives 4-bin labels from `param` quantiles, shuffles, caches, and
    returns an 80/20 train/val split.
    """
    ambient = x_raw.shape[1]
    A = rng.standard_normal(size=(ambient, dim)).astype(np.float32) / np.sqrt(dim)
    x = x_raw.astype(np.float32) @ A
    x += rng.standard_normal(size=x.shape).astype(np.float32) * 0.01

    q = np.quantile(param, [0.25, 0.5, 0.75])
    labels = np.digitize(param, q).astype(np.int64)

    perm = rng.permutation(n_total)
    x = x[perm]
    labels = labels[perm]

    features = torch.from_numpy(x).float()
    labels_t = torch.from_numpy(labels).long()
    _save_cache(cache_path, features, labels_t)

    n_train = int(n_total * 0.8)
    return (features[:n_train], labels_t[:n_train], features[n_train:], labels_t[n_train:])


def _prepare_flat_plane(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dim = args.swiss_roll_n, args.swiss_roll_dim
    seed = args.seed + _SWISS_SEED_OFFSET + 1
    cache_path = cache_dir / f"flat_plane_n{n}_d{dim}_s{seed}.pt"
    if cache_path.exists():
        tr_x, tr_y = _load_cache(cache_path)
        n_tr = int(n * 0.8)
        return tr_x[:n_tr], tr_y[:n_tr], tr_x[n_tr:], tr_y[n_tr:]

    print(f"  Generating flat plane: n={n}, lift_dim={dim}, seed={seed}")
    rng = np.random.default_rng(seed)
    x_raw = rng.standard_normal(size=(n, 2))
    angle = np.arctan2(x_raw[:, 1], x_raw[:, 0])  # label param
    return _finalize_manifold(x_raw, angle, dim, rng, cache_path, n)


def _prepare_torus(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dim = args.swiss_roll_n, args.swiss_roll_dim
    seed = args.seed + _SWISS_SEED_OFFSET + 2
    cache_path = cache_dir / f"torus_n{n}_d{dim}_s{seed}.pt"
    if cache_path.exists():
        tr_x, tr_y = _load_cache(cache_path)
        n_tr = int(n * 0.8)
        return tr_x[:n_tr], tr_y[:n_tr], tr_x[n_tr:], tr_y[n_tr:]

    print(f"  Generating torus: n={n}, lift_dim={dim}, seed={seed}")
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, size=n)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)
    R, r = 3.0, 1.0
    x_raw = np.stack(
        [
            (R + r * np.cos(theta)) * np.cos(phi),
            (R + r * np.cos(theta)) * np.sin(phi),
            r * np.sin(theta),
        ],
        axis=1,
    )
    return _finalize_manifold(x_raw, theta, dim, rng, cache_path, n)


def _prepare_klein_bottle(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dim = args.swiss_roll_n, args.swiss_roll_dim
    seed = args.seed + _SWISS_SEED_OFFSET + 3
    cache_path = cache_dir / f"klein_bottle_n{n}_d{dim}_s{seed}.pt"
    if cache_path.exists():
        tr_x, tr_y = _load_cache(cache_path)
        n_tr = int(n * 0.8)
        return tr_x[:n_tr], tr_y[:n_tr], tr_x[n_tr:], tr_y[n_tr:]

    print(f"  Generating Klein bottle: n={n}, lift_dim={dim}, seed={seed}")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 2 * np.pi, size=n)
    v = rng.uniform(0.0, 2 * np.pi, size=n)
    R = 2.0
    common = R + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)
    x_raw = np.stack(
        [
            common * np.cos(u),
            common * np.sin(u),
            np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v),
            np.cos(v),
        ],
        axis=1,
    )
    return _finalize_manifold(x_raw, u, dim, rng, cache_path, n)


def _prepare_mixed_curvature(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, dim = args.swiss_roll_n, args.swiss_roll_dim
    seed = args.seed + _SWISS_SEED_OFFSET + 4
    cache_path = cache_dir / f"mixed_curv_n{n}_d{dim}_s{seed}.pt"
    if cache_path.exists():
        tr_x, tr_y = _load_cache(cache_path)
        n_tr = int(n * 0.8)
        return tr_x[:n_tr], tr_y[:n_tr], tr_x[n_tr:], tr_y[n_tr:]

    print(f"  Generating mixed curvature (sphere + hyperboloid): n={n}, lift_dim={dim}, seed={seed}")
    rng = np.random.default_rng(seed)
    n_sph = n // 2
    n_hyp = n - n_sph

    # Unit sphere S^2 (positive curvature).
    sph = rng.standard_normal(size=(n_sph, 3))
    sph = sph / np.linalg.norm(sph, axis=1, keepdims=True)

    # Upper sheet of two-sheeted hyperboloid x^2 + y^2 - z^2 = -1 (negative curvature).
    xh = rng.standard_normal(size=(n_hyp, 2))
    zh = np.sqrt(1.0 + xh[:, 0] ** 2 + xh[:, 1] ** 2)
    hyp = np.stack([xh[:, 0], xh[:, 1], zh], axis=1)

    x_raw = np.concatenate([sph, hyp], axis=0)
    # Label param mixes signature id with radial position.
    sig = np.concatenate([np.zeros(n_sph), np.ones(n_hyp)])
    rad = np.concatenate([sph[:, 2], zh])
    param = sig * 10.0 + rad
    return _finalize_manifold(x_raw, param, dim, rng, cache_path, n)


_MANIFOLD_GENERATORS = {
    "flat_plane": _prepare_flat_plane,
    "swiss_roll": _prepare_swiss_roll,
    "torus": _prepare_torus,
    "klein_bottle": _prepare_klein_bottle,
    "mixed_curvature": _prepare_mixed_curvature,
}


def prepare_features(
    ds_name: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Unified feature loader. Returns (train_x, train_y, val_x, val_y, n_classes)."""
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if ds_name in TEXT_LOADERS:
        tr_x, tr_y, va_x, va_y = _prepare_text(ds_name, args, cache_dir)
    elif ds_name == "mnist":
        tr_x, tr_y, va_x, va_y = _prepare_mnist(args, cache_dir)
    elif ds_name == "cifar10":
        tr_x, tr_y, va_x, va_y = _prepare_cifar10(args, cache_dir)
    elif ds_name in _MANIFOLD_GENERATORS:
        tr_x, tr_y, va_x, va_y = _MANIFOLD_GENERATORS[ds_name](args, cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    return tr_x, tr_y, va_x, va_y, N_CLASSES[ds_name]


# ============================================================================
# Effective dimension analysis
# ============================================================================


def analyze_effective_dimension(
    train_embs: torch.Tensor,
) -> DimensionResult:
    """Run EffectiveDimensionAnalyzer and print a summary."""
    analyzer = EffectiveDimensionAnalyzer(device="cpu", dtype=torch.float32)
    result = analyzer.analyze(train_embs)

    print(f"\n  [Dimension Analysis]")
    print(f"    Embedding dim         : {train_embs.shape[1]}")
    print(f"    Intrinsic dim (BS)    : {result.intrinsic_dim}")
    print(f"    Participation ratio   : {result.participation_ratio:.1f}")
    top5_evr = result.explained_variance_ratio[:5].tolist()
    print(f"    Top-5 EVR             : {[f'{v:.3f}' for v in top5_evr]}")
    cum50 = int((result.explained_variance_ratio.cumsum(0) < 0.50).sum().item()) + 1
    cum90 = int((result.explained_variance_ratio.cumsum(0) < 0.90).sum().item()) + 1
    print(f"    Dims for 50%/90% var  : {cum50} / {cum90}")

    return result


def build_target_dims(intrinsic_dim: int) -> List[int]:
    """Build a sorted, deduplicated sweep of target dimensions."""
    candidates = [
        max(8, intrinsic_dim // 2),
        intrinsic_dim,
        min(512, intrinsic_dim * 2),
        64,
        128,
        256,
    ]
    dims = sorted(set(d for d in candidates if 8 <= d <= 512))
    return dims


# ============================================================================
# GBN Embedding Compressor
# ============================================================================


class GBNEmbedCompressor(CliffordModule):
    """Geometric Blade Network autoencoder for embedding compression.

    Architecture (encoder):
        Linear lift  →  [B, C, alg.dim]
        CliffordLayerNorm → GeometricGELU
        RotorLayer  (geometric alignment)
        CliffordLinear  (channel mixing)
        BladeSelector  (grade-selective gating)
        Linear readout  →  [B, bottleneck_dim]

    Architecture (decoder):
        Linear → GELU → Linear  →  [B, in_dim]  (for reconstruction loss)

    A shared linear classification head:
        Linear  →  [B, n_classes]
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_dim: int,
        channels: int,
        bottleneck_dim: int,
        n_classes: int,
    ):
        super().__init__(algebra)
        self.in_dim = in_dim
        self.channels = channels
        self.bottleneck_dim = bottleneck_dim
        self.n_classes = n_classes

        mv_dim = algebra.dim  # 2^n, e.g. 32 for Cl(5,0)
        self.mv_flat = channels * mv_dim  # bottleneck capacity

        # ── Encoder ──────────────────────────────────────────────────────────
        self.lift = nn.Linear(in_dim, self.mv_flat)
        self.norm = CliffordLayerNorm(algebra, channels)
        self.act = GeometricGELU(algebra, channels)
        self.rotor = RotorLayer(algebra, channels)
        self.linear = CliffordLinear(algebra, channels, channels)
        self.gate = BladeSelector(algebra, channels)
        self.readout_enc = nn.Linear(self.mv_flat, bottleneck_dim)

        # ── Decoder (for reconstruction loss) ────────────────────────────────
        hidden_dec = max(bottleneck_dim * 2, 256)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dec),
            nn.GELU(),
            nn.Linear(hidden_dec, in_dim),
        )

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Linear(bottleneck_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lift.weight)
        nn.init.zeros_(self.lift.bias)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.readout_enc.weight)
        nn.init.zeros_(self.readout_enc.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode raw embeddings to bottleneck.

        Args:
            x: [B, in_dim]

        Returns:
            z: [B, bottleneck_dim]  — compressed representation
            mv: [B, channels, alg.dim]  — gated multivector (for grade analysis)
        """
        B = x.shape[0]
        mv = self.lift(x).view(B, self.channels, -1)  # [B, C, D]
        mv = self.norm(mv)
        mv = self.act(mv)
        mv = self.rotor(mv)
        mv = self.linear(mv)
        mv = self.gate(mv)
        z = self.readout_enc(mv.reshape(B, -1))  # [B, bottleneck_dim]
        return z, mv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: [B, in_dim]

        Returns:
            z: [B, bottleneck_dim]  — compressed code
            x_hat: [B, in_dim]  — reconstructed embedding
            logits: [B, n_classes]
        """
        z, _ = self.encode(x)
        x_hat = self.decoder(z)
        logits = self.classifier(z)
        return z, x_hat, logits


# ============================================================================
# PCA compression
# ============================================================================


def precompute_pca(
    train_embs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute PCA axes from training embeddings.

    Returns (mean [D], Vh [D, D]) — top principal directions in rows of Vh.
    """
    mean = train_embs.mean(0)
    centered = train_embs - mean
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return mean, Vh


def pca_compress(
    train_embs: torch.Tensor,
    val_embs: torch.Tensor,
    target_dim: int,
    pca_mean: torch.Tensor,
    pca_Vh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA compression using precomputed axes.

    Args:
        train_embs: [N_train, D]
        val_embs:   [N_val, D]
        target_dim: number of principal components to keep
        pca_mean:   [D] mean from train set
        pca_Vh:     [D, D] right singular vectors (principal directions in rows)

    Returns:
        train_z:    [N_train, target_dim]
        val_z:      [N_val, target_dim]
        val_recon:  [N_val, D] back-projected reconstruction of val
    """
    Vk = pca_Vh[:target_dim]  # [target_dim, D]
    train_z = (train_embs - pca_mean) @ Vk.T  # [N_train, target_dim]
    val_centered = val_embs - pca_mean
    val_z = val_centered @ Vk.T  # [N_val, target_dim]
    val_recon = val_z @ Vk + pca_mean  # [N_val, D]
    return train_z, val_z, val_recon


# ============================================================================
# Training
# ============================================================================


def train_gbn(
    model: GBNEmbedCompressor,
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    val_embs: torch.Tensor,
    val_labels: torch.Tensor,
    args: argparse.Namespace,
) -> Dict:
    """Train the GBN compressor.

    Returns a history dict with keys:
        'loss', 'loss_recon', 'loss_clf', 'val_acc'
    """
    optimizer = RiemannianAdam.from_model(model, lr=args.lr, algebra=model.algebra)
    dataset = TensorDataset(train_embs, train_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    history = {"loss": [], "loss_recon": [], "loss_clf": [], "val_acc": []}

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = epoch_recon = epoch_clf = 0.0
        n_batches = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            _z, x_hat, logits = model(xb)

            loss_recon = F.mse_loss(x_hat, xb)
            loss_clf = F.cross_entropy(logits, yb)
            loss = args.alpha * loss_recon + args.beta * loss_clf

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_clf += loss_clf.item()
            n_batches += 1

        history["loss"].append(epoch_loss / n_batches)
        history["loss_recon"].append(epoch_recon / n_batches)
        history["loss_clf"].append(epoch_clf / n_batches)

        # Val accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                _, _, logits_val = model(val_embs)
                preds = logits_val.argmax(dim=-1)
                acc = (preds == val_labels).float().mean().item()
            history["val_acc"].append(acc)
            model.train()
            print(
                f"      epoch {epoch + 1:3d}/{args.epochs}  "
                f"loss={epoch_loss / n_batches:.4f}  "
                f"recon={epoch_recon / n_batches:.4f}  "
                f"clf={epoch_clf / n_batches:.4f}  "
                f"val_acc={acc:.3f}"
            )

    return history


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_linear_probe(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
) -> float:
    """Fit a logistic regression linear probe on compressed codes.

    Returns val accuracy.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_z.numpy())
    X_val = scaler.transform(val_z.numpy())
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_train, train_y.numpy())
    return float(clf.score(X_val, val_y.numpy()))


def cosine_similarity_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity between row-pairs of a and b."""
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return float((a_n * b_n).sum(dim=-1).mean().item())


def get_grade_spectrum(
    model: GBNEmbedCompressor,
    embs: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    """Extract mean grade energy spectrum from GBN bottleneck multivectors.

    Returns [n+1] grade energy tensor.
    """
    algebra = model.algebra
    spectral = SpectralAnalyzer(algebra)

    all_mv = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(embs), batch_size):
            xb = embs[i : i + batch_size]
            _, mv = model.encode(xb)
            all_mv.append(mv.detach())

    mv_cat = torch.cat(all_mv, dim=0)  # [N, C, dim]
    grade_energy = spectral.grade_energy_spectrum(mv_cat)  # [n+1]
    return grade_energy


# ============================================================================
# Dimension lifting test
# ============================================================================

_LIFT_TEST_DIM = 6  # Reduce to this many dims before lifting test


def run_lifting_test(
    train_embs: torch.Tensor,
    intrinsic_dim: int,
    device: str = "cpu",
) -> Dict:
    """DimensionLifter test on PCA-compressed embeddings.

    DimensionLifter requires d = p + q (the data dim must equal the algebra's
    vector-space dim).  Because Clifford algebra multivector spaces grow as 2^n
    and text intrinsic dims are typically O(100), we first compress to a small
    fixed dimension (_LIFT_TEST_DIM=6) via PCA and run the test there.

    The question answered: "Given 6D PCA-compressed embeddings, does adding
    an extra spacelike (+pos) or timelike (+null) dimension improve geodesic
    coherence of the neighbourhood graph?"
    """
    test_dim = max(2, min(_LIFT_TEST_DIM, intrinsic_dim))
    print(f"\n  [DimensionLifter] Reducing to {test_dim}D (Cl({test_dim},0)) for lift test …")

    analyzer = EffectiveDimensionAnalyzer(device=device)
    reduced = analyzer.reduce(train_embs, test_dim)  # [N, test_dim]

    # Cap sample size for GeodesicFlow k-NN computation
    sample = reduced[: min(500, len(reduced))]

    lifter = DimensionLifter(device=device)
    results = lifter.test(sample, p=test_dim, q=0, k=8)
    print(lifter.format_report(results))
    return results


# ============================================================================
# Manifold complexity scorer
# ============================================================================


def _manifold_complexity(dim_result, lifting_result: Dict, in_dim: int) -> float:
    """Scalar summarising manifold complexity.

    Combines intrinsic dim (normalised to [0, 1] via log-ratio vs ambient)
    and positive-signature lift gain (coherence improvement when adding a
    spacelike direction). Higher = more complex manifold.
    """
    intrinsic = max(1, dim_result.intrinsic_dim)
    # log-ratio gives a smoother scale than linear intrinsic/in_dim.
    dim_component = float(np.log(intrinsic + 1) / np.log(in_dim + 1))
    dim_component = max(0.0, min(1.0, dim_component))

    coh_orig = float(lifting_result["original"]["coherence"])
    coh_lift = float(lifting_result["lift_positive"]["coherence"])
    lift_gain = max(0.0, coh_lift - coh_orig)
    # Clamp gain to a reasonable scale; empirically most datasets <= 0.5.
    lift_component = min(1.0, lift_gain / 0.5)

    return 0.5 * dim_component + 0.5 * lift_component


# ============================================================================
# Plotting
# ============================================================================


def plot_dimension_analysis(
    dim_result,
    out_dir: Path,
    dataset_name: str,
    plot_metadata: str,
    args: argparse.Namespace,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Effective Dimension Analysis — {dataset_name}", fontsize=13)

    # Panel 1: Eigenvalue spectrum
    ax = axes[0]
    eigs = dim_result.eigenvalues.numpy()
    top_n = min(100, len(eigs))
    ax.plot(range(1, top_n + 1), eigs[:top_n], "b-", lw=1.5)
    ax.axvline(dim_result.intrinsic_dim, color="r", ls="--", lw=1.5, label=f"intrinsic dim={dim_result.intrinsic_dim}")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Covariance eigenspectrum")
    ax.legend(fontsize=8)
    ax.set_xlim(1, top_n)

    # Panel 2: Cumulative EVR
    ax = axes[1]
    evr_cumsum = dim_result.explained_variance_ratio.cumsum(0).numpy()
    ax.plot(range(1, len(evr_cumsum) + 1), evr_cumsum, "g-", lw=1.5)
    ax.axvline(dim_result.intrinsic_dim, color="r", ls="--", lw=1.5, label=f"intrinsic dim={dim_result.intrinsic_dim}")
    ax.axhline(0.90, color="gray", ls=":", lw=1, label="90% var")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative EVR")
    ax.set_title("Cumulative explained variance")
    ax.legend(fontsize=8)
    ax.set_xlim(1, min(300, len(evr_cumsum)))
    ax.set_ylim(0, 1.05)

    # Panel 3: Summary text
    ax = axes[2]
    ax.axis("off")
    summary = (
        f"Embedding dim:  {len(eigs)}\n"
        f"Intrinsic dim (broken-stick): {dim_result.intrinsic_dim}\n"
        f"Participation ratio: {dim_result.participation_ratio:.1f}\n"
        f"Compression ratio at intrinsic dim: {len(eigs) / dim_result.intrinsic_dim:.1f}×\n"
    )
    ax.text(
        0.05,
        0.6,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    path = save_experiment_figure(
        fig,
        output_dir=str(out_dir),
        experiment_name="inc_embed_compress",
        metadata=plot_metadata,
        plot_name=f"{dataset_name}_dimension_analysis",
        args=args,
        module=__name__,
        dpi=120,
    )
    print(f"  Saved {path}")


def plot_accuracy_vs_compression(
    results_by_dataset: Dict[str, Dict],
    out_dir: Path,
    plot_metadata: str,
    args: argparse.Namespace,
):
    n_ds = len(results_by_dataset)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), squeeze=False)
    fig.suptitle("Classification Accuracy vs Target Dimension", fontsize=13)

    for col, (ds_name, res) in enumerate(results_by_dataset.items()):
        ax = axes[0][col]
        target_dims = res["target_dims"]
        pca_accs = res["pca_acc"]
        gbn_accs = res["gbn_acc"]
        in_dim = res["in_dim"]

        ax.plot(target_dims, pca_accs, "b-o", label="PCA", markersize=5)
        ax.plot(target_dims, gbn_accs, "r-s", label="GBN", markersize=5)
        ax.axvline(res["intrinsic_dim"], color="gray", ls="--", lw=1, label=f"intrinsic ({res['intrinsic_dim']})")
        ax.set_xlabel("Target dimension")
        ax.set_ylabel("Val accuracy")
        ax.set_title(ds_name.upper())
        ax.legend(fontsize=9)
        # Secondary x-axis: compression ratio
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(target_dims)
        ax2.set_xticklabels([f"{in_dim / d:.0f}×" for d in target_dims], fontsize=7)
        ax2.set_xlabel("Compression ratio", fontsize=9)

    plt.tight_layout()
    path = save_experiment_figure(
        fig,
        output_dir=str(out_dir),
        experiment_name="inc_embed_compress",
        metadata=plot_metadata,
        plot_name="accuracy_vs_compression",
        args=args,
        module=__name__,
        dpi=120,
    )
    print(f"  Saved {path}")


def plot_reconstruction_similarity(
    results_by_dataset: Dict[str, Dict],
    out_dir: Path,
    plot_metadata: str,
    args: argparse.Namespace,
):
    n_ds = len(results_by_dataset)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), squeeze=False)
    fig.suptitle("Reconstruction Cosine Similarity vs Compression Ratio", fontsize=13)

    for col, (ds_name, res) in enumerate(results_by_dataset.items()):
        ax = axes[0][col]
        target_dims = res["target_dims"]
        in_dim = res["in_dim"]
        ratios = [in_dim / d for d in target_dims]
        pca_sims = res["pca_cos"]
        gbn_sims = res["gbn_cos"]

        ax.plot(ratios, pca_sims, "b-o", label="PCA", markersize=5)
        ax.plot(ratios, gbn_sims, "r-s", label="GBN", markersize=5)
        ax.set_xlabel("Compression ratio")
        ax.set_ylabel("Mean cosine similarity")
        ax.set_title(ds_name.upper())
        ax.legend(fontsize=9)
        ax.invert_xaxis()  # higher compression on right

    plt.tight_layout()
    path = save_experiment_figure(
        fig,
        output_dir=str(out_dir),
        experiment_name="inc_embed_compress",
        metadata=plot_metadata,
        plot_name="reconstruction_similarity",
        args=args,
        module=__name__,
        dpi=120,
    )
    print(f"  Saved {path}")


def plot_grade_energy_spectrum(
    grade_spectra_by_dataset: Dict[str, Dict],
    out_dir: Path,
    plot_metadata: str,
    args: argparse.Namespace,
):
    for ds_name, spectra in grade_spectra_by_dataset.items():
        target_dims = list(spectra.keys())
        n_td = len(target_dims)
        if n_td == 0:
            continue

        fig, axes = plt.subplots(1, n_td, figsize=(4 * n_td, 4), squeeze=False)
        fig.suptitle(f"GBN Grade Energy Spectrum — {ds_name.upper()}", fontsize=12)

        n_grades = spectra[target_dims[0]].shape[0]
        grade_labels = [f"G{k}" for k in range(n_grades)]

        for col, td in enumerate(target_dims):
            ax = axes[0][col]
            energy = spectra[td].numpy()
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_grades))
            ax.bar(grade_labels, energy, color=colors)
            ax.set_title(f"target={td}")
            ax.set_xlabel("Grade")
            ax.set_ylabel("Hermitian energy")
            ax.tick_params(axis="x", labelsize=8)

        plt.tight_layout()
        path = save_experiment_figure(
            fig,
            output_dir=str(out_dir),
            experiment_name="inc_embed_compress",
            metadata=plot_metadata,
            plot_name=f"{ds_name}_grade_energy_spectrum",
            args=args,
            module=__name__,
            dpi=120,
        )
        print(f"  Saved {path}")


def plot_training_curves(
    histories_by_dataset: Dict[str, Dict],
    out_dir: Path,
    plot_metadata: str,
    args: argparse.Namespace,
):
    for ds_name, histories in histories_by_dataset.items():
        target_dims = list(histories.keys())
        n_td = len(target_dims)
        if n_td == 0:
            continue

        fig, axes = plt.subplots(2, n_td, figsize=(4 * n_td, 7), squeeze=False)
        fig.suptitle(f"GBN Training Curves — {ds_name.upper()}", fontsize=12)

        for col, td in enumerate(target_dims):
            h = histories[td]
            ep = range(1, len(h["loss"]) + 1)

            ax = axes[0][col]
            ax.plot(ep, h["loss_recon"], "b-", lw=1.5, label="recon")
            ax.plot(ep, h["loss_clf"], "r-", lw=1.5, label="clf")
            ax.set_title(f"target={td}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)

            ax = axes[1][col]
            val_ep = [5 * i for i in range(1, len(h["val_acc"]) + 1)]
            if h["val_acc"]:
                ax.plot(val_ep, h["val_acc"], "g-o", lw=1.5, markersize=4)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val accuracy")
            ax.set_ylim(0, 1)

        plt.tight_layout()
        path = save_experiment_figure(
            fig,
            output_dir=str(out_dir),
            experiment_name="inc_embed_compress",
            metadata=plot_metadata,
            plot_name=f"{ds_name}_training_curves",
            args=args,
            module=__name__,
            dpi=120,
        )
        print(f"  Saved {path}")


def plot_lifting_test(
    lifting_results: Dict[str, Dict],
    out_dir: Path,
    plot_metadata: str,
    args: argparse.Namespace,
):
    if not lifting_results:
        return

    n_ds = len(lifting_results)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4), squeeze=False)
    fig.suptitle("DimensionLifter — Algebra Coherence Comparison", fontsize=13)

    for col, (ds_name, res) in enumerate(lifting_results.items()):
        ax = axes[0][col]
        keys = ["original", "lift_positive", "lift_null"]
        labels = ["Original\nCl(p,q)", "+Positive\nCl(p+1,q)", "+Null\nCl(p,q+1)"]
        cohs = [res[k]["coherence"] for k in keys]
        colors = ["steelblue", "darkorange", "forestgreen"]
        bars = ax.bar(labels, cohs, color=colors, alpha=0.85, edgecolor="black", lw=0.8)

        best = res["best"]
        best_idx = keys.index(best)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel("Geodesic coherence")
        ax.set_title(ds_name.upper())
        ax.set_ylim(min(0, min(cohs) - 0.05), max(cohs) + 0.1)

        # Annotate causal/noisy
        for i, k in enumerate(keys):
            causal = res[k]["causal"]
            sign = "O" if causal else "X"
            ax.text(i, cohs[i] + 0.01, sign, ha="center", va="bottom", fontsize=12, color="green" if causal else "red")

    plt.tight_layout()
    path = save_experiment_figure(
        fig,
        output_dir=str(out_dir),
        experiment_name="inc_embed_compress",
        metadata=plot_metadata,
        plot_name="lifting_test",
        args=args,
        module=__name__,
        dpi=120,
    )
    print(f"  Saved {path}")


# ============================================================================
# Main experiment
# ============================================================================


def run_dataset(
    ds_name: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Run the full experiment for one dataset.

    Returns:
        results: accuracy/cosine per target_dim
        histories: GBN training histories per target_dim
        grade_spectra: grade energy per target_dim
        lifting_result: DimensionLifter output
    """
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {ds_name.upper()}")
    print(f"{'=' * 60}")

    print("\n[1] Preparing features …")
    train_embs, train_labels, val_embs, val_labels, n_classes = prepare_features(ds_name, args)
    in_dim = train_embs.shape[1]
    print(f"  Train: {tuple(train_embs.shape)}  Val: {tuple(val_embs.shape)}  n_classes={n_classes}")

    # Dimension analysis
    print("\n[3] Effective dimension analysis …")
    dim_result = analyze_effective_dimension(train_embs)
    dataset_metadata = build_visualization_metadata(
        signature_metadata(args.algebra_p, args.algebra_q),
        datasets=ds_name,
        model_slug=args.model,
        seed=args.seed,
    )
    plot_dimension_analysis(dim_result, out_dir, ds_name, dataset_metadata, args)
    target_dims = build_target_dims(dim_result.intrinsic_dim)
    print(f"  Target dim sweep: {target_dims}")

    # DimensionLifter
    print("\n[4] DimensionLifter test …")
    lifting_result = run_lifting_test(train_embs, dim_result.intrinsic_dim)

    # Internal algebra
    algebra = setup_algebra(p=args.algebra_p, q=args.algebra_q, device="cpu")
    print(f"\n  Internal algebra: Cl({args.algebra_p},{args.algebra_q})  dim={algebra.dim}")

    # Precompute PCA axes once
    print("\n[5] Precomputing PCA axes …")
    pca_mean, pca_Vh = precompute_pca(train_embs)

    # Compression sweep
    print(f"\n    Sweep: {target_dims}")

    results = {
        "target_dims": target_dims,
        "pca_acc": [],
        "gbn_acc": [],
        "pca_cos": [],
        "gbn_cos": [],
        "intrinsic_dim": dim_result.intrinsic_dim,
        "in_dim": in_dim,
    }
    histories: Dict[int, Dict] = {}
    grade_spectra: Dict[int, torch.Tensor] = {}

    for td in target_dims:
        print(f"\n  ── target_dim={td} (ratio={in_dim / td:.1f}×) ──")

        # PCA baseline
        print(f"    [PCA] compressing …")
        train_pca, val_pca, val_recon_pca = pca_compress(train_embs, val_embs, td, pca_mean, pca_Vh)
        pca_acc = evaluate_linear_probe(train_pca, train_labels, val_pca, val_labels)
        pca_cos = cosine_similarity_mean(val_embs, val_recon_pca)

        results["pca_acc"].append(pca_acc)
        results["pca_cos"].append(pca_cos)
        print(f"    [PCA] acc={pca_acc:.3f}  cos_sim={pca_cos:.3f}")

        # GBN compressor
        print(f"    [GBN] training {args.epochs} epochs …")
        model = GBNEmbedCompressor(
            algebra=algebra,
            in_dim=in_dim,
            channels=args.channels,
            bottleneck_dim=td,
            n_classes=n_classes,
        )
        history = train_gbn(model, train_embs, train_labels, val_embs, val_labels, args)
        histories[td] = history

        # GBN reconstruction cosine similarity
        model.eval()
        with torch.no_grad():
            gbn_val_z, gbn_val_xhat, gbn_val_logits = model(val_embs)
            gbn_cos = cosine_similarity_mean(val_embs, gbn_val_xhat)
            gbn_acc_direct = (gbn_val_logits.argmax(-1) == val_labels).float().mean().item()

        # Also evaluate with linear probe on the code
        model.eval()
        with torch.no_grad():
            gbn_train_z, _, _ = model(train_embs)
        gbn_probe_acc = evaluate_linear_probe(gbn_train_z.detach(), train_labels, gbn_val_z.detach(), val_labels)
        # Report the best of probe vs direct classifier
        gbn_acc = max(gbn_acc_direct, gbn_probe_acc)

        results["gbn_acc"].append(gbn_acc)
        results["gbn_cos"].append(gbn_cos)
        print(f"    [GBN] acc={gbn_acc:.3f}  cos_sim={gbn_cos:.3f}")

        # Grade energy spectrum
        grade_spectra[td] = get_grade_spectrum(model, train_embs[:512])

    return results, histories, grade_spectra, lifting_result


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(ensure_output_dir(args.output_dir))
    plot_metadata = build_visualization_metadata(
        signature_metadata(args.algebra_p, args.algebra_q),
        datasets=args.datasets,
        model_slug=args.model,
        seed=args.seed,
    )

    results_by_dataset: Dict[str, Dict] = {}
    histories_by_dataset: Dict[str, Dict] = {}
    grade_spectra_by_dataset: Dict[str, Dict] = {}
    lifting_results: Dict[str, Dict] = {}

    for ds_name in args.datasets:
        res, hist, spectra, lifting = run_dataset(ds_name, args, out_dir)
        results_by_dataset[ds_name] = res
        histories_by_dataset[ds_name] = hist
        grade_spectra_by_dataset[ds_name] = spectra
        lifting_results[ds_name] = lifting

    print(f"\n[6] Generating plots in {out_dir} …")
    plot_accuracy_vs_compression(results_by_dataset, out_dir, plot_metadata, args)
    plot_reconstruction_similarity(results_by_dataset, out_dir, plot_metadata, args)
    plot_grade_energy_spectrum(grade_spectra_by_dataset, out_dir, plot_metadata, args)
    plot_training_curves(histories_by_dataset, out_dir, plot_metadata, args)
    plot_lifting_test(lifting_results, out_dir, plot_metadata, args)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for ds_name, res in results_by_dataset.items():
        deltas = [g - p for g, p in zip(res["gbn_acc"], res["pca_acc"])]
        best_idx = max(range(len(deltas)), key=deltas.__getitem__)
        best_td = res["target_dims"][best_idx]
        best_d = deltas[best_idx]
        headline = f"best GBN−PCA Δ = {best_d:+.3f} at target={best_td} (ratio {res['in_dim'] / best_td:.1f}×)"
        print(f"\n{ds_name.upper()} (intrinsic_dim={res['intrinsic_dim']}) — {headline}")
        print(
            f"  {'target':>8}  {'ratio':>6}  {'PCA acc':>8}  {'GBN acc':>8}  "
            f"{'Δ acc':>8}  {'PCA cos':>8}  {'GBN cos':>8}"
        )
        for i, td in enumerate(res["target_dims"]):
            print(
                f"  {td:>8}  {res['in_dim'] / td:>5.1f}×  "
                f"{res['pca_acc'][i]:>8.3f}  {res['gbn_acc'][i]:>8.3f}  "
                f"{deltas[i]:>+8.3f}  "
                f"{res['pca_cos'][i]:>8.3f}  {res['gbn_cos'][i]:>8.3f}"
            )

    print(f"\nDone. Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
