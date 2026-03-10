# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""SRBench-aligned Symbolic Regression Dataset (PMLB).

Loads the 24 benchmark datasets from SRBench 2025 via PMLB:
  - 12 first-principles equations (first_principles_*)
  - 12 curated black-box regression datasets

The PMLB PyPI package (v1.x) index does not include first_principles
datasets. We fetch those directly from the PMLB GitHub repository and
cache locally.

Reference:
  PMLB: La Cava et al. (2021), arXiv:2107.14351
  SRBench: La Cava et al. (2025), arXiv:2505.03977
"""

import os
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# PMLB GitHub raw URL for datasets not yet in the PyPI index
_PMLB_GITHUB_RAW = "https://github.com/EpistasisLab/pmlb/raw/master/datasets"

# ---------------------------------------------------------------------------
# SRBench 2025 dataset lists (24 total)
# ---------------------------------------------------------------------------

# Phenomenological & first-principles track (12 datasets)
FIRST_PRINCIPLES_DATASETS = [
    'first_principles_absorption',
    'first_principles_bode',
    'first_principles_hubble',
    'first_principles_ideal_gas',
    'first_principles_kepler',
    'first_principles_leavitt',
    'first_principles_newton',
    'first_principles_planck',
    'first_principles_rydberg',
    'first_principles_schechter',
    'first_principles_supernovae_zr',
    'first_principles_tully_fisher',
]

# Black-box track (12 datasets)
BLACKBOX_DATASETS = [
    '1028_SWD',
    '1089_USCrime',
    '1193_BNG_lowbwt',
    '1199_BNG_echoMonths',
    '192_vineyard',
    '210_cloud',
    '522_pm10',
    '557_analcatdata_apnea1',
    '579_fri_c0_250_5',
    '606_fri_c2_1000_10',
    '650_fri_c0_500_50',
    '678_visualizing_environmental',
]

# All 24 SRBench datasets
SRBENCH_DATASETS = FIRST_PRINCIPLES_DATASETS + BLACKBOX_DATASETS


def get_dataset_ids(category: str = "all") -> list[str]:
    """Return PMLB dataset names for a category.

    Args:
        category: One of "first_principles", "blackbox", or "all".

    Returns:
        List of PMLB dataset name strings.
    """
    if category == "first_principles":
        return list(FIRST_PRINCIPLES_DATASETS)
    elif category == "blackbox":
        return list(BLACKBOX_DATASETS)
    elif category == "all":
        return list(SRBENCH_DATASETS)
    else:
        raise ValueError(f"Unknown category '{category}'. "
                         f"Choose from: first_principles, blackbox, all")


def _fetch_pmlb_data(dataset_name: str, cache_dir: str) -> pd.DataFrame:
    """Fetch a PMLB dataset, falling back to GitHub raw if not in local index.

    Args:
        dataset_name: PMLB dataset name.
        cache_dir: Local cache directory.

    Returns:
        DataFrame with feature columns and 'target' column.
    """
    # Try local cache first
    if cache_dir:
        cached_path = os.path.join(cache_dir, dataset_name,
                                   dataset_name + '.tsv.gz')
        if os.path.exists(cached_path):
            return pd.read_csv(cached_path, sep='\t', compression='gzip')

    # Try pmlb.fetch_data (works for datasets in the PyPI index)
    try:
        import pmlb
        if dataset_name in pmlb.dataset_names:
            return pmlb.fetch_data(dataset_name, local_cache_dir=cache_dir)
    except ImportError:
        pass

    # Fallback: fetch directly from PMLB GitHub
    url = f"{_PMLB_GITHUB_RAW}/{dataset_name}/{dataset_name}.tsv.gz"
    logger.info(f"Fetching {dataset_name} from PMLB GitHub...")
    df = pd.read_csv(url, sep='\t', compression='gzip')

    # Cache locally for future use
    if cache_dir:
        local_dir = os.path.join(cache_dir, dataset_name)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, dataset_name + '.tsv.gz')
        df.to_csv(local_path, sep='\t', compression='gzip', index=False)

    return df


def get_sr_raw_splits(
    dataset_name: str = "192_vineyard",
    n_samples: int | None = 10000,
    cache_dir: str = "./data/pmlb_cache",
    seed: int = 42,
) -> tuple:
    """Return raw train/test splits without normalization.

    Reuses the same loading/splitting logic as get_sr_loaders but skips
    normalization, for use by BasisExpander which needs raw-scale data.

    Args:
        dataset_name: PMLB dataset name.
        n_samples: Max samples to use; None or 0 for all available.
        cache_dir: Local cache directory for PMLB downloads.
        seed: Random seed for splitting and subsampling.

    Returns:
        (X_train, y_train, X_test, y_test, var_names) -- all numpy arrays.
    """
    df = _fetch_pmlb_data(dataset_name, cache_dir)

    X = df.drop("target", axis=1).values.astype(np.float64)
    y = df["target"].values.astype(np.float64)
    var_names = [c for c in df.columns if c != "target"]

    # Filter NaN/Inf
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    rng = np.random.default_rng(seed)
    if n_samples and n_samples > 0 and n_samples < len(X):
        perm = rng.permutation(len(X))[:n_samples]
        X = X[perm]
        y = y[perm]

    # 75/25 split (SRBench standard)
    N = len(X)
    n_train = int(0.75 * N)
    perm = rng.permutation(N)
    X = X[perm]
    y = y[perm]

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test, var_names


class SRDataset(Dataset):
    """Holds normalised (x, y) pairs for one symbolic regression problem.

    Attributes:
        x (Tensor): Inputs [N, k].
        y (Tensor): Targets [N, 1].
        training (bool): Whether to apply augmentation.
        aug_sigma (float): Gaussian noise std for input perturbation.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 training: bool = False, aug_sigma: float = 0.0):
        self.x = x
        self.y = y
        self.training = training
        self.aug_sigma = aug_sigma

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.training and self.aug_sigma > 0:
            x = x + torch.randn_like(x) * self.aug_sigma
        return x, self.y[idx]


def get_sr_loaders(
    dataset_name: str = "192_vineyard",
    n_samples: int | None = 10000,
    batch_size: int = 64,
    noise: float = 0.0,
    cache_dir: str = "./data/pmlb_cache",
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = False,
    aug_sigma: float = 0.0,
) -> tuple:
    """Build train / test DataLoaders for one PMLB dataset.

    Split: 75 / 25 (SRBench standard). Normalisation is computed on training split.

    Args:
        dataset_name: PMLB dataset name (e.g. "192_vineyard").
        n_samples: Max samples to use; None or 0 for all available.
        batch_size: DataLoader batch size.
        noise: Gaussian noise std fraction to add to targets.
        cache_dir: Local cache directory for PMLB downloads.
        seed: Random seed for splitting and subsampling.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for CUDA transfer.
        aug_sigma: Gaussian noise std for input augmentation (in normalised
            space). If <= 0, auto-determined by dataset size: 0.1 for
            n_train < 20, 0.05 for n_train < 50, else 0.0.

    Returns:
        train_loader, test_loader,
        x_mean [k], x_std [k], y_mean scalar, y_std scalar,
        var_names list[str]
    """
    df = _fetch_pmlb_data(dataset_name, cache_dir)

    # Extract features and target
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    var_names = [c for c in df.columns if c != "target"]

    # Filter NaN/Inf
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    # Subsample if requested
    rng = np.random.default_rng(seed)
    if n_samples and n_samples > 0 and n_samples < len(X):
        perm = rng.permutation(len(X))[:n_samples]
        X = X[perm]
        y = y[perm]

    # Add noise if requested
    if noise > 0:
        y_std_raw = y.std()
        y = y + rng.normal(0, noise * y_std_raw, len(y)).astype(np.float32)

    # 75/25 split (SRBench standard)
    N = len(X)
    n_train = int(0.75 * N)

    # Shuffle before splitting
    perm = rng.permutation(N)
    X = X[perm]
    y = y[perm]

    train_X, test_X = X[:n_train], X[n_train:]
    train_y, test_y = y[:n_train], y[n_train:]

    # Convert to tensors
    train_X = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_y).unsqueeze(-1)  # [N, 1]
    test_X = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_y).unsqueeze(-1)

    # Normalise using training statistics
    x_mean = train_X.mean(0)
    x_std = train_X.std(0).clamp(min=1e-6)
    y_mean = train_y.mean()
    y_std = train_y.std().clamp(min=1e-6)

    def _norm_x(t):
        return (t - x_mean) / x_std

    def _norm_y(t):
        return (t - y_mean) / y_std

    # Auto-determine augmentation sigma for small datasets.
    # aug_sigma < 0 disables augmentation entirely; 0 means auto.
    if aug_sigma < 0:
        aug_sigma = 0.0
    elif aug_sigma == 0:
        if n_train < 20:
            aug_sigma = 0.1
        elif n_train < 50:
            aug_sigma = 0.05

    train_ds = SRDataset(_norm_x(train_X), _norm_y(train_y),
                         training=True, aug_sigma=aug_sigma)
    test_ds = SRDataset(_norm_x(test_X), _norm_y(test_y))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader, x_mean, x_std, y_mean, y_std, var_names
