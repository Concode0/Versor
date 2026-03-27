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
might be incomplete at this stage. If this geometric hypothesis proves structurally 
sound, it is planned to be spun off into a dedicated, independent repository 
for detailed research.

==============================================================================

This project requires data.
Arbitrary synthetic data may be used.
You may use SimGenerator implemented in the https://github.com/Concode0/Trajecto project,
but please pay attention to the license and note that while it is a sophisticated simulator, 
there are differences from reality.

==============================================================================
CALL FOR PARTICIPANTS
==============================================================================
This is an open experiment. We welcome contributions extending STA trajectory
reconstruction to:
  - Real Data Adoption with online inference
  - Higher-dimensional CGA Cl(4,1) for translation-as-rotation
  - Multi-sensor fusion (magnetometer, UWB)
  
If you use ideas from this experiment, please cite the Versor framework:
"""

from __future__ import annotations

import sys
import os
import argparse
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from core.metric import hermitian_grade_spectrum
from layers import RotorLayer, CliffordLayerNorm, MotherEmbedding, GeometricNeutralizer
from layers.primitives.base import CliffordModule
from functional.activation import GeometricGELU
from functional.loss import IsometryLoss, HermitianGradeRegularization
from optimizers.riemannian import RiemannianAdam

try:
    import h5py
except ImportError:
    h5py = None


# ============================================================================
# Data
# ============================================================================

def compute_gravity_rotation(gravity_vectors: np.ndarray) -> torch.Tensor:
    """Compute Procrustes rotation aligning mean gravity to -e3.

    Uses Rodrigues' rotation formula to find R such that R @ g_hat = [0, 0, -1].

    Args:
        gravity_vectors: [N, 3] gravity measurements in body frame.

    Returns:
        3x3 rotation matrix as torch.Tensor.
    """
    g_mean = gravity_vectors.mean(axis=0)
    g_hat = g_mean / (np.linalg.norm(g_mean) + 1e-8)
    target = np.array([0.0, 0.0, -1.0])

    v = np.cross(g_hat, target)
    s = np.linalg.norm(v)
    c = np.dot(g_hat, target)

    if s < 1e-8:
        # Already aligned (or anti-aligned)
        if c > 0:
            return torch.eye(3)
        else:
            # 180-degree rotation around x-axis
            R = np.diag([1.0, -1.0, -1.0])
            return torch.tensor(R, dtype=torch.float32)

    # Skew-symmetric cross-product matrix
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
    return torch.tensor(R, dtype=torch.float32)


class IMUTrajectoryDataset(Dataset):
    """Sliding-window dataset from SimGenerator HDF5 output.

    Loads sensor_data, gt_pos_data, gt_vel_data, gt_gravity_b_data from HDF5.
    Applies sliding window extraction with per-channel normalization.

    Args:
        h5_path: Path to HDF5 file.
        window_size: Sliding window length.
        stride: Window stride.
        split: One of 'train', 'val', 'test'.
        noise_scale: Multiplier for sensor noise augmentation.
    """

    def __init__(self, h5_path: str, window_size: int = 128, stride: int = 16,
                 split: str = 'train', noise_scale: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.noise_scale = noise_scale

        if h5py is None:
            raise ImportError("h5py required: uv pip install h5py")

        # Load all samples from HDF5
        sensors_list, pos_list, vel_list, gravity_list = [], [], [], []

        with h5py.File(h5_path, 'r') as f:
            # Try batched format first: /sensor_data [N, T, 7]
            if 'sensor_data' in f:
                sensors_all = np.array(f['sensor_data'])
                pos_all = np.array(f['gt_pos_data'])
                vel_all = np.array(f['gt_vel_data'])
                gravity_all = np.array(f['gt_gravity_b_data'])

                for i in range(sensors_all.shape[0]):
                    sensors_list.append(sensors_all[i])
                    pos_list.append(pos_all[i])
                    vel_list.append(vel_all[i])
                    gravity_list.append(gravity_all[i])
            else:
                # Per-sample groups: /0/sensor_data, /1/sensor_data, ...
                keys = sorted(f.keys(), key=lambda k: int(k) if k.isdigit() else k)
                for key in keys:
                    if key.isdigit() or key.startswith('sample'):
                        g = f[key]
                        sensors_list.append(np.array(g['sensor_data']))
                        pos_list.append(np.array(g['gt_pos_data']))
                        vel_list.append(np.array(g['gt_vel_data']))
                        gravity_list.append(np.array(g['gt_gravity_b_data']))

        n_samples = len(sensors_list)
        assert n_samples > 0, f"No samples found in {h5_path}"

        # Compute gravity rotation from all gravity data
        all_gravity = np.concatenate(gravity_list, axis=0)
        self.R_gravity = compute_gravity_rotation(all_gravity)

        # Sensor normalization stats (global, from full dataset)
        all_sensors = np.concatenate(sensors_list, axis=0)  # [total_T, 7]
        self.sensor_mean = torch.tensor(all_sensors.mean(axis=0), dtype=torch.float32)
        self.sensor_std = torch.tensor(
            all_sensors.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        # Delta position normalization: each window is localized so pos[0]=0.
        # Compute std from all per-window deltas across all samples.
        delta_pos_list = []
        for i in range(n_samples):
            T = sensors_list[i].shape[0]
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                p = pos_list[i][start:end]  # [W, 3]
                delta_pos_list.append(p - p[0:1])
        all_delta_pos = np.concatenate(delta_pos_list, axis=0)
        self.pos_mean = torch.zeros(3, dtype=torch.float32)
        self.pos_std = torch.tensor(
            all_delta_pos.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        all_vel = np.concatenate(vel_list, axis=0)
        self.vel_mean = torch.tensor(all_vel.mean(axis=0), dtype=torch.float32)
        self.vel_std = torch.tensor(
            all_vel.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        # 80/10/10 split
        indices = np.arange(n_samples)
        np.random.seed(42)
        np.random.shuffle(indices)
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)

        if split == 'train':
            split_idx = indices[:n_train]
        elif split == 'val':
            split_idx = indices[n_train:n_train + n_val]
        else:
            split_idx = indices[n_train + n_val:]

        # Extract sliding windows, pre-normalize, and stack into contiguous tensors
        sensor_windows = []
        pos_windows = []
        vel_windows = []

        for i in split_idx:
            T = sensors_list[i].shape[0]
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                s = torch.tensor(sensors_list[i][start:end], dtype=torch.float32)
                p = torch.tensor(pos_list[i][start:end], dtype=torch.float32)
                v = torch.tensor(vel_list[i][start:end], dtype=torch.float32)
                # Localize: set initial position to origin (delta trajectory)
                p = p - p[0:1]
                sensor_windows.append((s - self.sensor_mean) / self.sensor_std)
                pos_windows.append(p / self.pos_std)
                vel_windows.append((v - self.vel_mean) / self.vel_std)

        self.sensors = torch.stack(sensor_windows)     # [N, W, 7]
        self.positions = torch.stack(pos_windows)      # [N, W, 3]
        self.velocities = torch.stack(vel_windows)     # [N, W, 3]

    def __len__(self):
        return self.sensors.shape[0]

    def __getitem__(self, idx):
        sensor = self.sensors[idx]
        if self.noise_scale > 0:
            sensor = sensor + self.noise_scale * torch.randn_like(sensor)
        return sensor, self.positions[idx], self.velocities[idx]


# ============================================================================
# Models
# ============================================================================

class IMUToSTA(CliffordModule):
    """Embeds 7-channel IMU data into Cl(3,1) multivectors.

    Pipeline:
      1. Procrustes alignment (gravity → -e3) on raw 7-dim sensor data
      2. Grade-aware scatter into 16-dim multivector basis
      3. MotherEmbedding: channel expansion + CliffordLayerNorm
      4. GeometricNeutralizer: orthogonalize grade-0 drift from grade-2
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int,
                 R_gravity: Optional[torch.Tensor] = None,
                 use_neutralizer: bool = True,
                 use_procrustes: bool = True):
        super().__init__(algebra)
        self.channels = channels
        self.use_neutralizer = use_neutralizer

        # Procrustes: block-diagonal 7x7 rotation
        V = torch.eye(7)
        if R_gravity is not None and use_procrustes:
            V[:3, :3] = R_gravity       # rotate accel
            V[3:6, 3:6] = R_gravity     # rotate gyro
        self.register_buffer('R_procrustes', V)

        # MotherEmbedding for channel expansion (V=identity since Procrustes
        # is pre-applied on raw sensor data before scatter)
        self.mother = MotherEmbedding(
            algebra, input_dim=algebra.dim, channels=channels)

        if use_neutralizer:
            self.neutralizer = GeometricNeutralizer(algebra, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed sensor data into Cl(3,1) multivectors.

        Args:
            x: Sensor data [*, 7] (accel_xyz, gyro_xyz, fsr).

        Returns:
            Multivectors [*, channels, 16].
        """
        batch_shape = x.shape[:-1]

        # 1. Procrustes alignment in sensor space
        x = x @ self.R_procrustes.T

        # 2. Grade-aware scatter into Cl(3,1) basis
        mv = torch.zeros(*batch_shape, self.algebra.dim,
                          device=x.device, dtype=x.dtype)
        mv[..., 0] = x[..., 6]    # fsr → grade-0 scalar
        mv[..., 1] = x[..., 0]    # accel_x → e1 (grade-1)
        mv[..., 2] = x[..., 1]    # accel_y → e2 (grade-1)
        mv[..., 4] = x[..., 2]    # accel_z → e3 (grade-1)
        mv[..., 3] = x[..., 3]    # gyro_x → e12 (grade-2, spatial rotation)
        mv[..., 5] = x[..., 4]    # gyro_y → e13 (grade-2, spatial rotation)
        mv[..., 6] = x[..., 5]    # gyro_z → e23 (grade-2, spatial rotation)

        # 3. Channel expansion via MotherEmbedding (linear + LayerNorm)
        flat = mv.reshape(-1, self.algebra.dim)
        mv = self.mother(flat)   # [N, C, 16]
        mv = mv.reshape(*batch_shape, self.channels, self.algebra.dim)

        # 4. Neutralize grade-0/grade-2 coupling
        if self.use_neutralizer:
            flat = mv.reshape(-1, self.channels, self.algebra.dim)
            mv = self.neutralizer(flat)
            mv = mv.reshape(*batch_shape, self.channels, self.algebra.dim)

        return mv


class CausalRotorTCN(nn.Module):
    """RotorTCN with causal (left-only) padding for temporal processing.

    Replicates RotorTCN's architecture (per-frame rotor + 1D dilated conv)
    but uses left-only padding to prevent information leakage from future
    timesteps, matching real-time filtering constraints.

    Receptive field per layer: (kernel_size - 1) * dilation + 1
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int,
                 hidden_channels: int, kernel_size: int = 3,
                 dilation: int = 1):
        super().__init__()
        self.algebra = algebra

        self.rotor = RotorLayer(algebra, in_channels)

        input_dim = in_channels * algebra.dim
        hidden_dim = hidden_channels * algebra.dim
        self.hidden_channels = hidden_channels
        self.causal_pad = (kernel_size - 1) * dilation

        self.tcn = nn.Conv1d(
            input_dim, hidden_dim,
            kernel_size=kernel_size, dilation=dilation, padding=0)

        self.out_rotor = RotorLayer(algebra, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal rotor TCN.

        Args:
            x: [Batch, Time, Channels, Dim]

        Returns:
            [Batch, Time, HiddenChannels, Dim]
        """
        b, t, c, d = x.shape

        # 1. Per-frame rotor transform
        x_flat = x.reshape(b * t, c, d)
        x_rot = self.rotor(x_flat).reshape(b, t, c, d)

        # 2. Causal 1D convolution
        x_in = x_rot.reshape(b, t, c * d).transpose(1, 2)  # [B, C*D, T]
        x_in = F.pad(x_in, (self.causal_pad, 0))            # left-only pad
        y = self.tcn(x_in)                                   # [B, H*D, T]

        # 3. Reshape and output rotor
        y = y.transpose(1, 2)                                # [B, T, H*D]
        y = y.reshape(b, t, self.hidden_channels, d)

        y_flat = y.reshape(b * t, self.hidden_channels, d)
        return self.out_rotor(y_flat).reshape(b, t, self.hidden_channels, d)


class STATrajectoryNet(nn.Module):
    """Stacked Geometric TCN for 3D trajectory reconstruction.

    Architecture:
      IMUToSTA embedding → 5x (CausalRotorTCN + LayerNorm + GeometricGELU + residual)
      → grade-1 spatial readout → position/velocity heads

    Receptive field with 5 layers (k=3, dilation=1,2,4,8,16):
      1 + 2*(1+2+4+8+16) = 63 timesteps (~1.26s at 50Hz)
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int = 32,
                 num_layers: int = 5, kernel_size: int = 3,
                 R_gravity: Optional[torch.Tensor] = None,
                 use_neutralizer: bool = True,
                 use_procrustes: bool = True):
        super().__init__()
        self.algebra = algebra
        self.channels = channels

        # Embedding
        self.embedding = IMUToSTA(
            algebra, channels, R_gravity=R_gravity,
            use_neutralizer=use_neutralizer,
            use_procrustes=use_procrustes)

        # Stacked Geometric TCN with exponential dilation
        self.tcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                CausalRotorTCN(algebra, channels, channels,
                               kernel_size, dilation))
            self.norms.append(CliffordLayerNorm(algebra, channels))
            self.activations.append(GeometricGELU(algebra, channels))

        # Readout: extract grade-1 spatial components [e1, e2, e3]
        # indices [1, 2, 4] in Cl(3,1)
        self.pos_head = nn.Linear(channels * 3, 3)
        self.vel_head = nn.Linear(channels * 3, 3)

    def forward(self, x: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: sensor data → (position, velocity, features, embedding).

        Args:
            x: Sensor windows [B, W, 7].

        Returns:
            pos_pred: [B, W, 3]
            vel_pred: [B, W, 3]
            features: [B, W, C, 16] — post-TCN (for regularization)
            embedding: [B, W, C, 16] — pre-TCN (for isometry comparison)
        """
        B, W, _ = x.shape

        # Embed each timestep
        x_flat = x.reshape(B * W, 7)
        mv = self.embedding(x_flat)  # [B*W, C, 16]
        mv = mv.reshape(B, W, self.channels, self.algebra.dim)
        embedding = mv  # Save pre-TCN for isometry loss

        # Stacked TCN with residual connections
        features = mv
        for tcn, norm, act in zip(
                self.tcn_layers, self.norms, self.activations):
            residual = features
            out = tcn(features)  # [B, W, C, 16]
            # Per-timestep norm + activation
            b2, w2, c2, d2 = out.shape
            out = norm(out.reshape(b2 * w2, c2, d2)).reshape(b2, w2, c2, d2)
            out = act(out)
            features = out + residual

        # Grade-1 spatial extraction: indices [1, 2, 4] = e1, e2, e3
        spatial = torch.stack([
            features[..., 1],   # e1
            features[..., 2],   # e2
            features[..., 4],   # e3
        ], dim=-1)  # [B, W, C, 3]

        spatial_flat = spatial.reshape(B, W, self.channels * 3)
        pos_pred = self.pos_head(spatial_flat)   # [B, W, 3]
        vel_pred = self.vel_head(spatial_flat)   # [B, W, 3]

        return pos_pred, vel_pred, features, embedding


class MLPBaseline(nn.Module):
    """Simple MLP baseline for comparison.

    Flattens the full window and predicts position/velocity directly.
    """

    def __init__(self, window_size: int, input_dim: int = 7):
        super().__init__()
        flat_dim = window_size * input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.pos_head = nn.Linear(256, window_size * 3)
        self.vel_head = nn.Linear(256, window_size * 3)
        self.window_size = window_size

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [B, W, 7]

        Returns:
            pos_pred: [B, W, 3]
            vel_pred: [B, W, 3]
        """
        B = x.shape[0]
        h = self.net(x.reshape(B, -1))
        pos = self.pos_head(h).reshape(B, self.window_size, 3)
        vel = self.vel_head(h).reshape(B, self.window_size, 3)
        return pos, vel
    

def train_epoch(model: STATrajectoryNet, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                isometry_loss_fn: IsometryLoss,
                grade_reg_fn: HermitianGradeRegularization,
                device: str, lambda_mse: float = 1.0,
                lambda_vel: float = 0.5, lambda_iso: float = 0.1,
                lambda_grade: float = 0.01,
                use_isometry: bool = True) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_pos = 0.0
    total_vel = 0.0
    total_iso = 0.0
    total_grade = 0.0
    n_batches = 0

    for sensor, pos_gt, vel_gt in loader:
        sensor = sensor.to(device, non_blocking=True)
        pos_gt = pos_gt.to(device, non_blocking=True)
        vel_gt = vel_gt.to(device, non_blocking=True)

        pos_pred, vel_pred, features, embedding = model(sensor)

        loss_pos = F.mse_loss(pos_pred, pos_gt)
        loss_vel = F.mse_loss(vel_pred, vel_gt)

        loss = lambda_mse * loss_pos + lambda_vel * loss_vel

        if use_isometry:
            # Compare post-TCN features vs pre-TCN embedding (Hypothesis 1:
            # rotor transforms should preserve Lorentz metric norm)
            D = features.shape[-1]
            f_flat = features.reshape(-1, D)
            e_flat = embedding.reshape(-1, D)
            # Subsample for efficiency (full tensor can be B*W*C = 262144)
            n = f_flat.shape[0]
            if n > 4096:
                idx = torch.randperm(n, device=f_flat.device)[:4096]
                f_flat = f_flat[idx]
                e_flat = e_flat[idx]
            loss_iso = isometry_loss_fn(f_flat, e_flat)
            loss += lambda_iso * loss_iso
            total_iso += loss_iso.item()

        loss_grade = grade_reg_fn(features)
        loss += lambda_grade * loss_grade

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_pos += loss_pos.item()
        total_vel += loss_vel.item()
        total_grade += loss_grade.item()
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'pos_mse': total_pos / max(n_batches, 1),
        'vel_mse': total_vel / max(n_batches, 1),
        'iso': total_iso / max(n_batches, 1),
        'grade': total_grade / max(n_batches, 1),
    }


def train_epoch_mlp(model: MLPBaseline, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str) -> Dict[str, float]:
    """Train MLP baseline for one epoch."""
    model.train()
    total_loss = 0.0
    total_pos = 0.0
    total_vel = 0.0
    n_batches = 0

    for sensor, pos_gt, vel_gt in loader:
        sensor = sensor.to(device, non_blocking=True)
        pos_gt = pos_gt.to(device, non_blocking=True)
        vel_gt = vel_gt.to(device, non_blocking=True)

        pos_pred, vel_pred = model(sensor)

        loss_pos = F.mse_loss(pos_pred, pos_gt)
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        loss = loss_pos + 0.5 * loss_vel

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_pos += loss_pos.item()
        total_vel += loss_vel.item()
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'pos_mse': total_pos / max(n_batches, 1),
        'vel_mse': total_vel / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str,
             is_sta: bool = True) -> Dict[str, float]:
    """Evaluate model on a dataset split.

    Returns RMSE for position and velocity.
    """
    model.eval()
    pos_sq_sum = 0.0
    vel_sq_sum = 0.0
    n_samples = 0

    for sensor, pos_gt, vel_gt in loader:
        sensor = sensor.to(device, non_blocking=True)
        pos_gt = pos_gt.to(device, non_blocking=True)
        vel_gt = vel_gt.to(device, non_blocking=True)

        if is_sta:
            pos_pred, vel_pred, _, _ = model(sensor)
        else:
            pos_pred, vel_pred = model(sensor)

        pos_sq_sum += ((pos_pred - pos_gt) ** 2).sum().item()
        vel_sq_sum += ((vel_pred - vel_gt) ** 2).sum().item()
        n_samples += pos_gt.numel()

    pos_rmse = math.sqrt(pos_sq_sum / max(n_samples, 1))
    vel_rmse = math.sqrt(vel_sq_sum / max(n_samples, 1))

    return {'pos_rmse': pos_rmse, 'vel_rmse': vel_rmse}


@torch.no_grad()
def compute_grade_spectrum(model: STATrajectoryNet, loader: DataLoader,
                           algebra: CliffordAlgebra,
                           device: str) -> np.ndarray:
    """Compute mean grade energy spectrum across a dataset."""
    model.eval()
    spectra = []

    for sensor, _, _ in loader:
        sensor = sensor.to(device, non_blocking=True)
        _, _, features, _ = model(sensor)
        flat = features.reshape(-1, algebra.dim)
        spectrum = hermitian_grade_spectrum(algebra, flat)  # [N, n+1]
        spectra.append(spectrum.mean(dim=0).cpu().numpy())
        break  # Single batch snapshot

    return np.mean(spectra, axis=0) if spectra else np.zeros(algebra.n + 1)


def save_plots(history: Dict[str, List], output_dir: str):
    """Generate and save diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Training curves: STA vs MLP
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    if 'sta_loss' in history:
        ax.plot(history['sta_loss'], label='STA', linewidth=1)
    if 'mlp_loss' in history:
        ax.plot(history['mlp_loss'], label='MLP', linewidth=1)
    for key in history:
        if key.startswith('ablation_') and key.endswith('_loss'):
            label = key.replace('_loss', '').replace('ablation_', '')
            ax.plot(history[key], label=label, linewidth=1, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if 'sta_pos_rmse' in history:
        ax.plot(history['sta_pos_rmse'], label='STA', linewidth=1)
    if 'mlp_pos_rmse' in history:
        ax.plot(history['mlp_pos_rmse'], label='MLP', linewidth=1)
    for key in history:
        if key.startswith('ablation_') and key.endswith('_pos_rmse'):
            label = key.replace('_pos_rmse', '').replace('ablation_', '')
            ax.plot(history[key], label=label, linewidth=1, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Position RMSE')
    ax.set_title('Position RMSE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if 'sta_iso' in history:
        ax.plot(history['sta_iso'], label='Isometry Violation', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Isometry Loss')
    ax.set_title('Isometry Violation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # 2. Grade spectrum evolution heatmap
    if 'grade_spectrum' in history and history['grade_spectrum']:
        spectra = np.array(history['grade_spectrum'])  # [T, n+1]
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(spectra.T, aspect='auto', origin='lower',
                       cmap='viridis')
        ax.set_xlabel('Evaluation Step')
        ax.set_ylabel('Grade')
        ax.set_title('Grade Spectrum Evolution')
        ax.set_yticks(range(spectra.shape[1]))
        plt.colorbar(im, ax=ax, label='Energy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grade_spectrum.png'), dpi=150)
        plt.close()

    # 3. Noise robustness
    if 'noise_scales' in history and 'noise_sta_rmse' in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        scales = history['noise_scales']
        ax.plot(scales, history['noise_sta_rmse'], 'o-', label='STA',
                linewidth=2)
        if 'noise_mlp_rmse' in history:
            ax.plot(scales, history['noise_mlp_rmse'], 's-', label='MLP',
                    linewidth=2)
        ax.set_xlabel('Noise Scale')
        ax.set_ylabel('Position RMSE')
        ax.set_title('Noise Robustness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'noise_robustness.png'), dpi=150)
        plt.close()

    # 4. 3D trajectory comparison (if stored)
    if 'traj_gt' in history and 'traj_sta' in history:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            gt = history['traj_gt']
            sta = history['traj_sta']
            mlp = history.get('traj_mlp')

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-',
                    linewidth=2, label='Ground Truth')
            ax.plot(sta[:, 0], sta[:, 1], sta[:, 2], 'b-',
                    linewidth=1.5, label='STA')
            if mlp is not None:
                ax.plot(mlp[:, 0], mlp[:, 1], mlp[:, 2], 'r--',
                        linewidth=1.5, label='MLP')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Trajectory Reconstruction')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'trajectory_3d.png'),
                        dpi=150)
            plt.close()
        except Exception:
            pass

    print(f"[INFO] Plots saved to {output_dir}/")


def train_model(args):
    """Main training loop with optional ablation and noise sweep."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Algebra: Cl(3,1), dim=16")
    print(f"[CONFIG] Channels: {args.channels}, TCN layers: {args.num_layers}")
    print(f"[CONFIG] Window: {args.window_size}, Stride: {args.stride}")

    # Data
    print(f"\n[DATA] Loading {args.data}")
    train_ds = IMUTrajectoryDataset(
        args.data, args.window_size, args.stride, split='train')
    val_ds = IMUTrajectoryDataset(
        args.data, args.window_size, args.stride, split='val')
    test_ds = IMUTrajectoryDataset(
        args.data, args.window_size, args.stride, split='test')

    print(f"  Train: {len(train_ds)} windows")
    print(f"  Val:   {len(val_ds)} windows")
    print(f"  Test:  {len(test_ds)} windows")

    use_pin_memory = device != 'cpu'
    num_workers = 2 if device != 'cpu' else 0

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=use_pin_memory, num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        pin_memory=use_pin_memory, num_workers=num_workers)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        pin_memory=use_pin_memory, num_workers=num_workers)

    R_gravity = train_ds.R_gravity

    # Algebra — create on CPU; model.to(device) moves tables automatically
    algebra = CliffordAlgebra(3, 1, device='cpu')
    print(f"[INFO] Algebra dim: {algebra.dim}")

    # Loss functions
    iso_loss_fn = IsometryLoss(algebra)
    grade_reg_fn = HermitianGradeRegularization(algebra)

    # History
    history: Dict[str, List] = defaultdict(list)

    # ---- Define model configurations to train ----
    configs = [
        ('STA', dict(use_neutralizer=True, use_procrustes=True,
                      use_isometry=True)),
        ('MLP', None),  # Special: MLP baseline
    ]

    if args.ablation:
        configs.extend([
            ('no_neutralizer', dict(use_neutralizer=False,
                                     use_procrustes=True,
                                     use_isometry=True)),
            ('no_procrustes', dict(use_neutralizer=True,
                                    use_procrustes=False,
                                    use_isometry=True)),
            ('no_isometry', dict(use_neutralizer=True,
                                  use_procrustes=True,
                                  use_isometry=False)),
        ])

    # ---- Train each configuration ----
    trained_models = {}

    for name, cfg in configs:
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        if cfg is None:
            # MLP baseline
            model = MLPBaseline(args.window_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            is_sta = False
        else:
            model = STATrajectoryNet(
                algebra, channels=args.channels,
                num_layers=args.num_layers,
                kernel_size=args.kernel_size,
                R_gravity=R_gravity,
                use_neutralizer=cfg['use_neutralizer'],
                use_procrustes=cfg['use_procrustes'],
            ).to(device)
            optimizer = RiemannianAdam(
                model.parameters(), lr=args.lr, algebra=algebra)
            is_sta = True

        num_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Parameters: {num_params:,}")

        prefix = name.lower().replace(' ', '_')
        if name != 'STA' and name != 'MLP':
            prefix = f'ablation_{prefix}'

        use_isometry = cfg['use_isometry'] if cfg else False

        for epoch in range(1, args.epochs + 1):
            if is_sta:
                metrics = train_epoch(
                    model, train_loader, optimizer,
                    iso_loss_fn, grade_reg_fn, device,
                    use_isometry=use_isometry)
            else:
                metrics = train_epoch_mlp(
                    model, train_loader, optimizer, device)

            history[f'{prefix}_loss'].append(metrics['loss'])

            if epoch % args.diag_interval == 0 or epoch == 1:
                val_metrics = evaluate(
                    model, val_loader, device, is_sta=is_sta)
                history[f'{prefix}_pos_rmse'].append(val_metrics['pos_rmse'])
                history[f'{prefix}_vel_rmse'].append(val_metrics['vel_rmse'])

                if is_sta:
                    history[f'{prefix}_iso'].append(metrics.get('iso', 0.0))

                    # Grade spectrum snapshot
                    if name == 'STA':
                        spectrum = compute_grade_spectrum(
                            model, val_loader, algebra, device)
                        history['grade_spectrum'].append(spectrum)

                print(f"  [{name}] Epoch {epoch:4d}  "
                      f"loss={metrics['loss']:.4f}  "
                      f"pos_rmse={val_metrics['pos_rmse']:.4f}  "
                      f"vel_rmse={val_metrics['vel_rmse']:.4f}")

        # Final test evaluation
        test_metrics = evaluate(model, test_loader, device, is_sta=is_sta)
        print(f"\n  [{name}] TEST  "
              f"pos_rmse={test_metrics['pos_rmse']:.4f}  "
              f"vel_rmse={test_metrics['vel_rmse']:.4f}")

        trained_models[name] = (model, is_sta, test_metrics)

    # ---- 3D Trajectory visualization (first test batch) ----
    print("\n[VIS] Extracting sample trajectory for visualization...")
    for sensor, pos_gt, vel_gt in test_loader:
        sensor = sensor.to(device)

        # Denormalize
        pos_mean = test_ds.pos_mean.numpy()
        pos_std = test_ds.pos_std.numpy()
        history['traj_gt'] = pos_gt[0].numpy() * pos_std + pos_mean

        if 'STA' in trained_models:
            sta_model = trained_models['STA'][0]
            sta_model.eval()
            with torch.no_grad():
                pos_pred, _, _, _ = sta_model(sensor)
            history['traj_sta'] = (
                pos_pred[0].cpu().numpy() * pos_std + pos_mean)

        if 'MLP' in trained_models:
            mlp_model = trained_models['MLP'][0]
            mlp_model.eval()
            with torch.no_grad():
                pos_pred, _ = mlp_model(sensor)
            history['traj_mlp'] = (
                pos_pred[0].cpu().numpy() * pos_std + pos_mean)
        break

    # ---- Noise sweep ----
    if args.noise_sweep:
        noise_scales = [0.0, 0.5, 1.0, 1.5, 2.0]
        history['noise_scales'] = noise_scales
        sta_rmses = []
        mlp_rmses = []

        print("\n[NOISE] Running noise robustness sweep...")

        for ns in noise_scales:
            noisy_ds = IMUTrajectoryDataset(
                args.data, args.window_size, args.stride,
                split='test', noise_scale=ns)
            noisy_loader = DataLoader(
                noisy_ds, batch_size=args.batch_size)

            if 'STA' in trained_models:
                m = evaluate(trained_models['STA'][0], noisy_loader,
                             device, is_sta=True)
                sta_rmses.append(m['pos_rmse'])
            else:
                sta_rmses.append(float('nan'))

            if 'MLP' in trained_models:
                m = evaluate(trained_models['MLP'][0], noisy_loader,
                             device, is_sta=False)
                mlp_rmses.append(m['pos_rmse'])
            else:
                mlp_rmses.append(float('nan'))

            print(f"  noise={ns:.1f}  "
                  f"STA={sta_rmses[-1]:.4f}  "
                  f"MLP={mlp_rmses[-1]:.4f}")

        history['noise_sta_rmse'] = sta_rmses
        history['noise_mlp_rmse'] = mlp_rmses

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for name, (_, _, test_metrics) in trained_models.items():
        print(f"  {name:20s}  "
              f"pos_rmse={test_metrics['pos_rmse']:.4f}  "
              f"vel_rmse={test_metrics['vel_rmse']:.4f}")

    # Plots
    if args.save_plots:
        save_plots(dict(history), args.output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='STA Trajectory Reconstruction — Cl(3,1) for IMU-based '
                    '3D handwriting reconstruction')

    # Data
    p.add_argument('--data', type=str,
                   default='../Trajecto/data/sta_experiment.h5',
                   help='Path to HDF5 data from SimGenerator')
    p.add_argument('--window-size', type=int, default=128)
    p.add_argument('--stride', type=int, default=16)

    # Model
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--num-layers', type=int, default=5)
    p.add_argument('--kernel-size', type=int, default=3)

    # Training
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')

    # Evaluation
    p.add_argument('--diag-interval', type=int, default=10)

    # Experiment modes
    p.add_argument('--ablation', action='store_true',
                   help='Run 3 ablated variants: no Neutralizer, '
                        'no Procrustes, no IsometryLoss')
    p.add_argument('--noise-sweep', action='store_true',
                   help='Evaluate at noise scales [0.0, 0.5, 1.0, 1.5, 2.0]')

    # Output
    p.add_argument('--save-plots', action='store_true')
    p.add_argument('--output-dir', type=str, default='sta_plots')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
