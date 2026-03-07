# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""DeepLense gravitational lensing dataset.

Three-tier loading: cached .pt > raw .npy files > synthetic SIS lens fallback.

Data from ML4Sci/DeepLense: 3-class dark matter substructure classification
(no_substructure, CDM/subhalo, vortex/axion) with gravitational lensing images.
"""

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from log import get_logger

logger = get_logger(__name__)

# Dark matter substructure classes
CLASS_NAMES = {0: "no_substructure", 1: "subhalo_CDM", 2: "vortex_axion"}


# ---------------------------------------------------------------------------
# Synthetic SIS/NFW lens simulation (always available, no download needed)
# ---------------------------------------------------------------------------

def _sis_deflection(theta_x, theta_y, theta_E):
    """Singular Isothermal Sphere deflection angles.

    Args:
        theta_x, theta_y: Image-plane coordinates [H, W].
        theta_E: Einstein radius.

    Returns:
        (alpha_x, alpha_y): Deflection angle fields [H, W].
    """
    r = torch.sqrt(theta_x ** 2 + theta_y ** 2 + 1e-8)
    alpha_x = theta_E * theta_x / r
    alpha_y = theta_E * theta_y / r
    return alpha_x, alpha_y


def _sis_convergence(theta_x, theta_y, theta_E):
    """SIS convergence (kappa) field: kappa = theta_E / (2 * r)."""
    r = torch.sqrt(theta_x ** 2 + theta_y ** 2 + 1e-8)
    return theta_E / (2.0 * r)


def _make_source_galaxy(H, W, n_components=3, rng=None):
    """Generate a synthetic source galaxy as a sum of Gaussians.

    Args:
        H, W: Image dimensions.
        n_components: Number of Gaussian components.
        rng: numpy random generator.

    Returns:
        source: [H, W] tensor, values in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    source = torch.zeros(H, W)
    for _ in range(n_components):
        cx = rng.uniform(-0.3, 0.3)
        cy = rng.uniform(-0.3, 0.3)
        sx = rng.uniform(0.05, 0.2)
        sy = rng.uniform(0.05, 0.2)
        amp = rng.uniform(0.3, 1.0)
        angle = rng.uniform(0, math.pi)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = xx - cx
        dy = yy - cy
        rx = cos_a * dx + sin_a * dy
        ry = -sin_a * dx + cos_a * dy
        gauss = amp * torch.exp(-0.5 * ((rx / sx) ** 2 + (ry / sy) ** 2))
        source = source + gauss

    source = source / (source.max() + 1e-8)
    return source


def _lens_image(source, theta_E, H, W):
    """Apply SIS lensing to a source image via ray-tracing.

    For each pixel in the image plane, trace the ray back to the source plane
    using the deflection angle, and sample the source at that position.

    Args:
        source: [H, W] source galaxy image.
        theta_E: Einstein radius.
        H, W: Image dimensions.

    Returns:
        lensed: [H, W] lensed image.
        kappa: [H, W] convergence map.
    """
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    theta_y, theta_x = torch.meshgrid(y, x, indexing='ij')

    alpha_x, alpha_y = _sis_deflection(theta_x, theta_y, theta_E)
    kappa = _sis_convergence(theta_x, theta_y, theta_E)

    # Source plane coordinates: beta = theta - alpha
    beta_x = theta_x - alpha_x
    beta_y = theta_y - alpha_y

    # Normalize to grid_sample coordinates [-1, 1]
    grid = torch.stack([beta_x, beta_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    source_4d = source.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    lensed = torch.nn.functional.grid_sample(
        source_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    lensed = lensed.squeeze(0).squeeze(0)  # [H, W]

    return lensed, kappa


def _add_substructure(kappa, label, H, W, rng=None):
    """Add dark matter substructure perturbation to convergence map.

    Args:
        kappa: [H, W] base convergence map.
        label: 0=no_sub, 1=CDM subhalos, 2=vortex.
        rng: numpy random generator.

    Returns:
        kappa_perturbed: [H, W] with substructure added.
    """
    if rng is None:
        rng = np.random.default_rng()

    if label == 0:
        return kappa

    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    if label == 1:
        # CDM: many small point-like subhalos
        n_sub = rng.integers(5, 15)
        perturbation = torch.zeros(H, W)
        for _ in range(n_sub):
            cx = rng.uniform(-0.8, 0.8)
            cy = rng.uniform(-0.8, 0.8)
            mass = rng.uniform(0.01, 0.05)
            r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-6)
            perturbation = perturbation + mass / (2.0 * r)
        return kappa + perturbation

    if label == 2:
        # Vortex (axion): wave-like interference pattern
        n_modes = rng.integers(2, 5)
        perturbation = torch.zeros(H, W)
        for _ in range(n_modes):
            kx = rng.uniform(-10, 10)
            ky = rng.uniform(-10, 10)
            phase = rng.uniform(0, 2 * math.pi)
            amp = rng.uniform(0.01, 0.04)
            wave = amp * torch.cos(kx * xx + ky * yy + phase)
            perturbation = perturbation + wave
        return kappa + perturbation

    return kappa


def generate_synthetic_sample(image_size=64, rng=None):
    """Generate one synthetic gravitational lensing sample.

    Returns:
        dict with keys: lensed [1,H,W], source [1,H,W], kappa [1,H,W], label (int).
    """
    if rng is None:
        rng = np.random.default_rng()

    H = W = image_size
    label = int(rng.integers(0, 3))
    theta_E = rng.uniform(0.2, 0.6)

    source = _make_source_galaxy(H, W, rng=rng)
    lensed, kappa = _lens_image(source, theta_E, H, W)
    kappa = _add_substructure(kappa, label, H, W, rng=rng)

    # Add realistic noise
    noise_std = rng.uniform(0.01, 0.05)
    lensed = lensed + noise_std * torch.randn_like(lensed)
    lensed = lensed.clamp(0, 1)

    return {
        'lensed': lensed.unsqueeze(0),   # [1, H, W]
        'source': source.unsqueeze(0),   # [1, H, W]
        'kappa': kappa.unsqueeze(0),     # [1, H, W]
        'label': label,
    }


def generate_synthetic_dataset(n_samples=1000, image_size=64, seed=42):
    """Generate a full synthetic dataset.

    Args:
        n_samples: Number of samples to generate.
        image_size: H=W pixel size.
        seed: Random seed.

    Returns:
        List of sample dicts.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        sample = generate_synthetic_sample(image_size, rng=rng)
        samples.append(sample)
        if (i + 1) % 200 == 0:
            logger.debug("Generated %d/%d synthetic samples", i + 1, n_samples)
    return samples


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class DeepLenseDataset(Dataset):
    """Gravitational lensing dataset for dark matter substructure analysis.

    Returns:
        dict with:
            lensed: [1, H, W] lensed image
            source: [1, H, W] source galaxy (if available)
            kappa:  [1, H, W] convergence map (if available)
            label:  int class label (0=no_sub, 1=CDM, 2=vortex)
    """

    def __init__(self, samples):
        """Initialize from a list of sample dicts.

        Args:
            samples: List of dicts with 'lensed', 'source', 'kappa', 'label'.
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'lensed': s['lensed'].float(),
            'source': s['source'].float(),
            'kappa': s['kappa'].float(),
            'label': s['label'],
        }


# ---------------------------------------------------------------------------
# Raw .npy loading (DeepLense format: class-organized directories)
# ---------------------------------------------------------------------------

def _load_npy_dataset(root, image_size=64):
    """Load DeepLense data from .npy files organized by class directories.

    Expected structure:
        root/
            no_substructure/    *.npy (each file is one image)
            subhalo_CDM/        *.npy
            vortex_axion/       *.npy

    Args:
        root: Root directory containing class subdirectories.
        image_size: Target size to resize images to.

    Returns:
        List of sample dicts.
    """
    class_dirs = {
        0: "no_substructure",
        1: "subhalo_CDM",
        2: "vortex_axion",
    }

    samples = []
    for label, dirname in class_dirs.items():
        class_dir = os.path.join(root, dirname)
        if not os.path.isdir(class_dir):
            # Try alternative names
            alt_names = {
                0: ["no_substructure", "Model_I", "model_1", "axion"],
                1: ["subhalo_CDM", "CDM", "Model_II", "model_2", "subhalo"],
                2: ["vortex_axion", "vortex", "Model_III", "model_3"],
            }
            found = False
            for alt in alt_names.get(label, []):
                alt_path = os.path.join(root, alt)
                if os.path.isdir(alt_path):
                    class_dir = alt_path
                    found = True
                    break
            if not found:
                logger.warning("Directory not found for class %d: %s", label, dirname)
                continue

        npy_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        for fname in npy_files:
            img = np.load(os.path.join(class_dir, fname))
            img = torch.from_numpy(img).float()

            # Handle various shapes
            if img.dim() == 2:
                img = img.unsqueeze(0)  # [H, W] -> [1, H, W]
            elif img.dim() == 3 and img.shape[-1] in (1, 3):
                img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

            # Resize if needed
            if img.shape[-2] != image_size or img.shape[-1] != image_size:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(image_size, image_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)

            # Take first channel if multi-channel
            if img.shape[0] > 1:
                img = img[:1]

            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            samples.append({
                'lensed': img,
                'source': torch.zeros_like(img),  # not available in raw data
                'kappa': torch.zeros_like(img),    # not available in raw data
                'label': label,
            })

    logger.info("Loaded %d samples from %s", len(samples), root)
    return samples


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def get_deeplense_loaders(
    root="./data/deeplense",
    variant="synthetic",
    image_size=64,
    n_samples=3000,
    batch_size=16,
    seed=42,
    num_workers=2,
    pin_memory=None,
    max_samples=None,
):
    """Create train/val/test DataLoaders for DeepLense.

    Three-tier loading:
        1. Cached .pt file
        2. Raw .npy files from DeepLense download
        3. Synthetic SIS lens fallback

    Args:
        root: Data directory.
        variant: "synthetic", "model_1", "model_2", "model_3", "model_4".
        image_size: Target image size.
        n_samples: Number of synthetic samples (only for variant="synthetic").
        batch_size: Batch size.
        seed: Random seed.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for CUDA.
        max_samples: Max samples to use (None = all).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    cache_path = os.path.join(root, f"deeplense_{variant}_{image_size}.pt")

    # Tier 1: Cached .pt
    if os.path.exists(cache_path):
        logger.info("Loading cached dataset from %s", cache_path)
        samples = torch.load(cache_path, weights_only=False)
    else:
        # Tier 2: Raw .npy files
        raw_dir = os.path.join(root, variant)
        if variant != "synthetic" and os.path.isdir(raw_dir):
            logger.info("Loading raw .npy data from %s", raw_dir)
            samples = _load_npy_dataset(raw_dir, image_size=image_size)
        else:
            # Tier 3: Synthetic fallback
            if variant != "synthetic":
                logger.warning(
                    "Data for variant '%s' not found at %s. "
                    "Falling back to synthetic SIS lens data.",
                    variant, raw_dir,
                )
            logger.info("Generating %d synthetic samples (size=%d)", n_samples, image_size)
            samples = generate_synthetic_dataset(n_samples, image_size, seed)

        # Cache for next time
        os.makedirs(root, exist_ok=True)
        torch.save(samples, cache_path)
        logger.info("Cached dataset to %s", cache_path)

    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]

    dataset = DeepLenseDataset(samples)

    # Split: 70/15/15
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    pm = pin_memory if pin_memory is not None else False
    loader_kwargs = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pm
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    logger.info(
        "DeepLense splits: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    return train_loader, val_loader, test_loader
