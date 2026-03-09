# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""DEAP EEG dataset loader with Differential Entropy band features.

The DEAP dataset (Koelstra et al., 2012) contains 32-channel EEG recordings
from 32 participants watching 40 music videos, with Valence/Arousal/Dominance/
Liking (VADL) self-assessment ratings.

Feature extraction follows the standard DEAP protocol: Differential Entropy (DE)
computed per frequency band (theta, alpha, beta, gamma) per channel per window.

Normalization is **subject-wise**: each subject's DE features are z-scored using
that subject's own mean/std. This accounts for large inter-subject variability
in EEG amplitude and is critical for cross-subject generalization (LOSO).
"""

import os
import pickle
import math
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from log import get_logger

logger = get_logger(__name__)

# ── DEAP 10-20 channel order (indices 0-31) ──────────────────────────────────
DEAP_CHANNELS = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8',
    'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fz', 'Cz',
]

# ── Brain region groups ───────────────────────────────────────────────────────
REGION_GROUPS = {
    'frontal':   [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 30],  # 13 channels
    'central':   [6, 8, 9, 22, 24, 25, 31],                        # 7 channels
    'temporal':  [7, 23],                                            # 2 channels
    'parietal':  [10, 11, 26, 27, 15],                              # 5 channels
    'occipital': [12, 13, 14, 28, 29],                              # 5 channels
}

# ── Frequency bands (Hz) ─────────────────────────────────────────────────────
BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 14.0),
    'beta':  (14.0, 31.0),
    'gamma': (31.0, 50.0),
}

SAMPLING_RATE = 128
BASELINE_SAMPLES = 384   # 3 seconds × 128 Hz
STIMULUS_SAMPLES = 7680  # 60 seconds × 128 Hz

NUM_BANDS = len(BANDS)


def _deap_cache_path(data_root, sid, window_size, stride, n_bands):
    """Cache path for one subject's pre-computed DE features."""
    return Path(data_root).parent / "deap_cache" / f"s{sid:02d}_w{window_size}_st{stride}_b{n_bands}.pt"


def _bandpass_filter(data, low, high, fs, order=5):
    """Apply Butterworth bandpass filter.

    Args:
        data: ndarray [..., time_samples]
        low: Lower cutoff frequency (Hz).
        high: Upper cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.

    Returns:
        Filtered ndarray, same shape.
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def _compute_de(signal):
    """Compute Differential Entropy assuming Gaussian distribution.

    DE = 0.5 * log(2 * pi * e * variance)

    Args:
        signal: ndarray [..., time_samples]

    Returns:
        DE values, shape [...] (last axis consumed).
    """
    var = np.var(signal, axis=-1)
    var = np.maximum(var, 1e-10)  # numerical floor
    return 0.5 * np.log(2.0 * math.pi * math.e * var)


def extract_de_features(eeg, fs=SAMPLING_RATE, bands=None):
    """Extract DE features per band for a single window.

    Args:
        eeg: ndarray [num_channels, window_samples]
        fs: Sampling rate.
        bands: Dict of band_name → (low_hz, high_hz). Defaults to BANDS.

    Returns:
        ndarray [num_channels, num_bands] of DE values.
    """
    if bands is None:
        bands = BANDS
    n_ch = eeg.shape[0]
    n_bands = len(bands)
    features = np.zeros((n_ch, n_bands), dtype=np.float32)
    for i, (_, (low, high)) in enumerate(bands.items()):
        filtered = _bandpass_filter(eeg, low, high, fs)
        features[:, i] = _compute_de(filtered)
    return features


def group_features(de_features, region_groups=None):
    """Group DE features by brain region.

    Args:
        de_features: ndarray [num_channels, num_bands]
        region_groups: Dict mapping region name to channel indices.

    Returns:
        Dict mapping region name to flattened feature vector (1D ndarray).
    """
    if region_groups is None:
        region_groups = REGION_GROUPS
    result = {}
    for name, indices in region_groups.items():
        result[name] = de_features[indices].flatten().astype(np.float32)
    return result


def get_group_sizes(region_groups=None, num_bands=NUM_BANDS):
    """Compute input dimension per group for model construction.

    Returns:
        Dict mapping region name to input_dim (num_channels_in_group × num_bands).
    """
    if region_groups is None:
        region_groups = REGION_GROUPS
    return {name: len(indices) * num_bands for name, indices in region_groups.items()}


# ── Subject-wise normalization ────────────────────────────────────────────────

def _normalize_subject_samples(samples):
    """Z-score normalize a single subject's samples using that subject's stats.

    Modifies samples in-place and returns (means, stds) per group.

    Args:
        samples: List of (group_dict, labels) for one subject.

    Returns:
        (means_dict, stds_dict) computed from this subject's data.
    """
    if not samples:
        return {}, {}

    group_names = list(samples[0][0].keys())
    group_arrays = {name: [] for name in group_names}

    for grouped, _ in samples:
        for name in group_names:
            group_arrays[name].append(grouped[name])

    means = {}
    stds = {}
    for name in group_names:
        arr = np.stack(group_arrays[name])
        means[name] = arr.mean(axis=0)
        stds[name] = arr.std(axis=0) + 1e-8

    for i in range(len(samples)):
        grouped, vadl = samples[i]
        normed = {}
        for name in group_names:
            normed[name] = ((grouped[name] - means[name]) / stds[name]).astype(np.float32)
        samples[i] = (normed, vadl)

    return means, stds


class DEAPDataset(Dataset):
    """DEAP EEG dataset with Differential Entropy band features.

    Loads preprocessed .dat files, segments trials into windows, and extracts
    DE features per frequency band per channel. Features are grouped by brain
    region for the MotherEmbedding architecture.

    Normalization is **subject-wise**: each subject's features are z-scored
    independently using that subject's own statistics, so cross-subject
    distribution shift is handled before pooling into a single dataset.

    Args:
        data_root: Path to data_preprocessed_python/ directory.
        subjects: List of subject IDs (1-32) to load.
        window_size: Window size in samples (default 512 = 4s at 128Hz).
        stride: Window stride in samples (default = window_size, non-overlapping).
        bands: Frequency band dict or None for defaults.
        normalize: Whether to apply subject-wise z-score normalization.
    """

    def __init__(self, data_root, subjects, window_size=512, stride=None,
                 bands=None, normalize=True):
        super().__init__()
        self.data_root = data_root
        self.window_size = window_size
        self.stride = stride or window_size
        self.bands = bands or BANDS
        self.normalize = normalize

        self.samples = []       # list of (group_features_dict, labels_array)
        self.subject_ids = []   # parallel list: subject id for each sample
        self._subject_stats = {}  # {sid: (means_dict, stds_dict)}

        self._load_subjects(subjects)

    def _load_subjects(self, subjects):
        """Load and window all trials for given subjects, normalize per-subject.

        Uses a per-subject cache to skip expensive bandpass filtering on
        subsequent runs. Cache stores raw (un-normalized) features so one
        cache file serves both normalize=True and normalize=False.
        """
        n_bands = len(self.bands)

        for sid in subjects:
            cache_path = _deap_cache_path(
                self.data_root, sid, self.window_size, self.stride, n_bands
            )

            if cache_path.exists():
                cached = torch.load(cache_path, weights_only=False)
                subject_samples = cached["samples"]
                logger.info("Subject %02d: loaded %d windows from cache", sid, len(subject_samples))
            else:
                path = os.path.join(self.data_root, f's{sid:02d}.dat')
                if not os.path.exists(path):
                    logger.warning("Subject file not found: %s", path)
                    continue

                with open(path, 'rb') as f:
                    dat = pickle.load(f, encoding='latin1')

                data = dat['data']     # (40, 40, 8064) — trials × channels × samples
                labels = dat['labels'] # (40, 4) — VADL

                subject_samples = []
                for trial_idx in range(data.shape[0]):
                    eeg = data[trial_idx, :32, :]  # 32 EEG channels only
                    vadl = labels[trial_idx]        # [4]

                    # Strip baseline, keep stimulus
                    stimulus = eeg[:, BASELINE_SAMPLES:BASELINE_SAMPLES + STIMULUS_SAMPLES]

                    # Window the stimulus
                    n_windows = (STIMULUS_SAMPLES - self.window_size) // self.stride + 1
                    for w in range(n_windows):
                        start = w * self.stride
                        end = start + self.window_size
                        window = stimulus[:, start:end]  # [32, window_size]

                        de = extract_de_features(window, SAMPLING_RATE, self.bands)
                        grouped = group_features(de)
                        subject_samples.append((grouped, vadl.astype(np.float32)))

                # Save cache (raw, before normalization)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"samples": subject_samples}, cache_path)
                logger.info("Subject %02d: %d windows (cached to %s)", sid, len(subject_samples), cache_path)

            # Subject-wise normalization (applied after cache load)
            if self.normalize and subject_samples:
                means, stds = _normalize_subject_samples(subject_samples)
                self._subject_stats[sid] = (means, stds)

            self.samples.extend(subject_samples)
            self.subject_ids.extend([sid] * len(subject_samples))

        logger.info("Total: %d windows from %d subjects", len(self.samples), len(subjects))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grouped, vadl = self.samples[idx]
        group_tensors = {name: torch.from_numpy(feat) for name, feat in grouped.items()}
        label_tensor = torch.from_numpy(vadl)
        label_tensor = (label_tensor - 1.0) / 8.0  # 1-9 → [0,1]
        return group_tensors, label_tensor


def _collate_deap(batch):
    """Custom collate for dict-based group data."""
    group_names = list(batch[0][0].keys())
    group_data = {
        name: torch.stack([item[0][name] for item in batch])
        for name in group_names
    }
    labels = torch.stack([item[1] for item in batch])
    return group_data, labels


def get_deap_loaders(data_root, subject_id=1, mode='cross_subject',
                     batch_size=32, window_size=512, stride=None,
                     fold=0, n_folds=5, num_workers=0):
    """Create DEAP train/val DataLoaders.

    Normalization is always **subject-wise**: each subject's DE features are
    z-scored using that subject's own statistics before merging into the
    combined dataset. This handles inter-subject EEG amplitude variability.

    Args:
        data_root: Path to data_preprocessed_python/.
        subject_id: Subject ID (1-32). For cross_subject: held-out test subject.
                     For within_subject: the subject to run CV on.
        mode: 'cross_subject' (LOSO, default) or 'within_subject' (5-fold CV).
        batch_size: Batch size.
        window_size: Window size in samples.
        stride: Window stride (None = non-overlapping).
        fold: Fold index for within_subject CV (0 to n_folds-1).
        n_folds: Number of CV folds for within_subject mode.
        num_workers: DataLoader workers.

    Returns:
        (train_loader, val_loader) tuple.
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"DEAP data not found at {data_root}. "
            "Download from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ "
            "and place preprocessed files in data/deap/data_preprocessed_python/"
        )

    if mode == 'cross_subject':
        # LOSO: train on 31 subjects, validate on held-out subject
        all_subjects = list(range(1, 33))
        train_subjects = [s for s in all_subjects if s != subject_id]
        train_ds = DEAPDataset(data_root, train_subjects, window_size, stride)
        val_ds = DEAPDataset(data_root, [subject_id], window_size, stride)

    elif mode == 'within_subject':
        # Single subject, 5-fold CV over 40 trials
        ds = DEAPDataset(data_root, [subject_id], window_size, stride)
        windows_per_trial = (STIMULUS_SAMPLES - window_size) // (stride or window_size) + 1
        n_trials = 40
        fold_size = n_trials // n_folds
        val_trials = set(range(fold * fold_size, (fold + 1) * fold_size))

        train_samples, val_samples = [], []
        for i, sample in enumerate(ds.samples):
            trial_idx = i // windows_per_trial
            if trial_idx in val_trials:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

        train_ds = DEAPDataset.__new__(DEAPDataset)
        train_ds.samples = train_samples
        val_ds = DEAPDataset.__new__(DEAPDataset)
        val_ds.samples = val_samples

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'cross_subject' or 'within_subject'.")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_deap, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_deap, num_workers=num_workers,
    )

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))
    return train_loader, val_loader
