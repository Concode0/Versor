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

STA IMU Trajectory Reconstruction in Cl(3,1).

Hypothesis
  A 7-channel IMU reading (accel, gyro, fsr) is naturally a single Cl(3,1)
  multivector: accel → grade-1 spatial vector, gyro → grade-2 Hodge bivector,
  fsr → grade-0 scalar. A learnable Spin(3,1) calibration rotor (Procrustes-
  initialised from gravity) followed by a causal rotor TCN reconstructs the
  3-D trajectory end-to-end through a single supervised MSE — without loss
  terms for isometry or grade confinement.

Natural loss
  ``MSE(pos) + 0.5 · MSE(vel)``. Isometry of the rotor TCN, grade-energy
  confinement, calibration magnitude, and noise robustness are all demoted
  to post-training measurements.

Data
  Expects SimGenerator HDF5 (https://github.com/Concode0/Trajecto) with keys
  ``sensor_data``, ``gt_pos_data``, ``gt_vel_data``, ``gt_gravity_b_data``.

Run
  uv run python -m experiments.inc_sta_trajectory --data <path-to-h5>
"""

from __future__ import annotations

import argparse
import math
import os, sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments._lib import (
    count_parameters, ensure_output_dir, make_experiment_parser,
    mean_grade_spectrum, print_banner, report_diagnostics,
    run_supervised_loop, save_training_curve, set_seed, setup_algebra,
)

from core.algebra import CliffordAlgebra
from core.metric import signature_norm_squared
from layers import CliffordLayerNorm, GeometricNeutralizer, MotherEmbedding, RotorLayer
from layers.primitives.base import CliffordModule
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam

try:
    import h5py
except ImportError:
    h5py = None


# ============================================================================
# Physical priors
# ============================================================================

def compute_gravity_bivector(gravity_vectors: np.ndarray,
                             algebra_dim: int = 16) -> torch.Tensor:
    """Procrustes bivector: grade-2 element ``B`` of Cl(3,1) such that
    ``exp(-B/2)`` rotates the mean body-frame gravity direction onto ``-e3``.

    Because the correction is a rotor, grade-1 (accel) and grade-2 (gyro)
    channels are aligned consistently by the same sandwich product.
    """
    g_hat = gravity_vectors.mean(axis=0)
    g_hat = g_hat / (np.linalg.norm(g_hat) + 1e-8)
    cos_theta = max(-1.0, min(1.0, float(-g_hat[2])))
    theta = math.acos(cos_theta)
    ax, ay = -g_hat[1], g_hat[0]
    an = math.sqrt(ax * ax + ay * ay) + 1e-8
    ax, ay = ax / an, ay / an

    B = torch.zeros(algebra_dim, dtype=torch.float32)
    if theta > 1e-6:
        B[6] = theta * ax      # e23
        B[5] = -theta * ay     # e13
        B[3] = 0.0             # e12 (az=0 by construction)
    return B


def _build_imu_scatter(algebra_dim: int = 16) -> torch.Tensor:
    """Routing matrix ``[7, dim]``: accel → grade-1, gyro → grade-2 (Hodge
    dual), fsr → scalar. Gyro as bivector makes a rotation rotor act on it
    identically to accel under the sandwich product.
    """
    S = torch.zeros(7, algebra_dim, dtype=torch.float32)
    S[0, 1] = 1.0     # accel_x → e1
    S[1, 2] = 1.0     # accel_y → e2
    S[2, 4] = 1.0     # accel_z → e3
    S[3, 6] = 1.0     # gyro_x  → e23
    S[4, 5] = -1.0    # gyro_y  → -e13  (= e31)
    S[5, 3] = 1.0     # gyro_z  → e12
    S[6, 0] = 1.0     # fsr     → scalar
    return S


# ============================================================================
# Dataset
# ============================================================================

class IMUTrajectoryDataset(Dataset):
    """Sliding-window IMU dataset from SimGenerator HDF5."""

    def __init__(self, h5_path: str, window_size: int = 128, stride: int = 16,
                 split: str = 'train', noise_scale: float = 0.0,
                 seed: int = 42):
        super().__init__()
        self.window_size = window_size
        self.noise_scale = noise_scale
        if h5py is None:
            raise ImportError('h5py required: uv pip install h5py')

        sensors_list, pos_list, vel_list, gravity_list = [], [], [], []
        with h5py.File(h5_path, 'r') as f:
            if 'sensor_data' in f:
                for i in range(f['sensor_data'].shape[0]):
                    sensors_list.append(np.asarray(f['sensor_data'][i]))
                    pos_list.append(np.asarray(f['gt_pos_data'][i]))
                    vel_list.append(np.asarray(f['gt_vel_data'][i]))
                    gravity_list.append(np.asarray(f['gt_gravity_b_data'][i]))
            else:
                keys = sorted(f.keys(), key=lambda k: int(k) if k.isdigit() else k)
                for key in keys:
                    if key.isdigit() or key.startswith('sample'):
                        g = f[key]
                        sensors_list.append(np.asarray(g['sensor_data']))
                        pos_list.append(np.asarray(g['gt_pos_data']))
                        vel_list.append(np.asarray(g['gt_vel_data']))
                        gravity_list.append(np.asarray(g['gt_gravity_b_data']))

        n_samples = len(sensors_list)
        assert n_samples > 0, f'No samples found in {h5_path}'

        all_gravity = np.concatenate(gravity_list, axis=0)
        self.gravity_bivector = compute_gravity_bivector(all_gravity)

        all_sensors = np.concatenate(sensors_list, axis=0)
        self.sensor_mean = torch.tensor(all_sensors.mean(axis=0), dtype=torch.float32)
        self.sensor_std = torch.tensor(
            all_sensors.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        # Per-window delta-position std (windows are localized so pos[0]=0)
        deltas = []
        for i in range(n_samples):
            T = sensors_list[i].shape[0]
            for start in range(0, T - window_size + 1, stride):
                p = pos_list[i][start:start + window_size]
                deltas.append(p - p[0:1])
        all_delta = np.concatenate(deltas, axis=0)
        self.pos_mean = torch.zeros(3, dtype=torch.float32)
        self.pos_std = torch.tensor(
            all_delta.std(axis=0).clip(min=1e-6), dtype=torch.float32)
        all_vel = np.concatenate(vel_list, axis=0)
        self.vel_mean = torch.tensor(all_vel.mean(axis=0), dtype=torch.float32)
        self.vel_std = torch.tensor(
            all_vel.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        rng = np.random.RandomState(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)
        if split == 'train':
            split_idx = indices[:n_train]
        elif split == 'val':
            split_idx = indices[n_train:n_train + n_val]
        else:
            split_idx = indices[n_train + n_val:]

        sensor_windows, pos_windows, vel_windows = [], [], []
        for i in split_idx:
            T = sensors_list[i].shape[0]
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                s = torch.tensor(sensors_list[i][start:end], dtype=torch.float32)
                p = torch.tensor(pos_list[i][start:end], dtype=torch.float32)
                v = torch.tensor(vel_list[i][start:end], dtype=torch.float32)
                p = p - p[0:1]
                sensor_windows.append((s - self.sensor_mean) / self.sensor_std)
                pos_windows.append(p / self.pos_std)
                vel_windows.append((v - self.vel_mean) / self.vel_std)

        self.sensors = torch.stack(sensor_windows)
        self.positions = torch.stack(pos_windows)
        self.velocities = torch.stack(vel_windows)

    def __len__(self) -> int:
        return self.sensors.shape[0]

    def __getitem__(self, idx: int):
        sensor = self.sensors[idx]
        if self.noise_scale > 0:
            sensor = sensor + self.noise_scale * torch.randn_like(sensor)
        return sensor, self.positions[idx], self.velocities[idx]


# ============================================================================
# Models
# ============================================================================

class STAEmbed(CliffordModule):
    """IMU → Cl(3,1) multivector with learnable Procrustes rotor + Neutralizer.

    ``raw [*, 7] —(scatter)→ [*, 16] —(calibration rotor)→ [*, 16]
    —(MotherEmbedding)→ [*, C, 16] —(Neutralizer)→ [*, C, 16]``

    The calibration rotor absorbs geometric (axis-alignment) bias via a
    single sandwich product acting on accel AND gyro consistently;
    Neutralizer cleans up the stochastic grade-0↔grade-2 covariance leak.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int,
                 gravity_bivector: Optional[torch.Tensor] = None):
        super().__init__(algebra)
        self.channels = channels

        self.register_buffer('scatter_matrix', _build_imu_scatter(algebra.dim))
        self.register_buffer(
            'g2_mask', algebra.grade_masks[2].to(dtype=torch.float32))

        if gravity_bivector is None:
            gravity_bivector = torch.zeros(algebra.dim)
        calib_init = (gravity_bivector.to(dtype=torch.float32,
                                          device=self.g2_mask.device)
                      * self.g2_mask)
        self.calib_bivector = nn.Parameter(calib_init.clone())

        self.mother = MotherEmbedding(
            algebra, input_dim=algebra.dim, channels=channels)
        self.neutralizer = GeometricNeutralizer(algebra, channels)

    def calibration_rotor(self) -> torch.Tensor:
        B = self.calib_bivector * self.g2_mask
        return self.algebra.exp(-0.5 * B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        dim = self.algebra.dim
        mv = x @ self.scatter_matrix
        flat = mv.reshape(-1, dim)
        flat = self.algebra.versor_product(self.calibration_rotor(), flat)
        mv = self.mother(flat)
        mv = self.neutralizer(mv)
        return mv.reshape(*batch_shape, self.channels, dim)


class CausalRotorTCN(CliffordModule):
    """Left-padded rotor-TCN block (per-frame rotor + causal 1-D conv + rotor).

    Receptive field per layer: ``(k - 1) * dilation + 1``.
    """

    def __init__(self, algebra: CliffordAlgebra, in_channels: int,
                 hidden_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__(algebra)
        self.rotor = RotorLayer(algebra, in_channels)
        self.hidden_channels = hidden_channels
        self.causal_pad = (kernel_size - 1) * dilation
        self.tcn = nn.Conv1d(
            in_channels * algebra.dim, hidden_channels * algebra.dim,
            kernel_size=kernel_size, dilation=dilation, padding=0)
        self.out_rotor = RotorLayer(algebra, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, d = x.shape
        x_rot = self.rotor(x.reshape(b * t, c, d)).reshape(b, t, c, d)
        x_in = x_rot.reshape(b, t, c * d).transpose(1, 2)
        x_in = F.pad(x_in, (self.causal_pad, 0))
        y = self.tcn(x_in).transpose(1, 2).reshape(b, t, self.hidden_channels, d)
        y_flat = y.reshape(b * t, self.hidden_channels, d)
        return self.out_rotor(y_flat).reshape(b, t, self.hidden_channels, d)


class STATrajectoryNet(CliffordModule):
    """STAEmbed → stacked CausalRotorTCN → grade-1 spatial readout."""

    def __init__(self, algebra: CliffordAlgebra, channels: int = 32,
                 num_layers: int = 5, kernel_size: int = 3,
                 gravity_bivector: Optional[torch.Tensor] = None):
        super().__init__(algebra)
        self.channels = channels
        self.embedding = STAEmbed(algebra, channels, gravity_bivector)
        self.tcn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(num_layers):
            self.tcn_layers.append(CausalRotorTCN(
                algebra, channels, channels, kernel_size, dilation=2 ** i))
            self.norms.append(CliffordLayerNorm(algebra, channels))
            self.activations.append(GeometricGELU(algebra, channels))
        self.pos_head = nn.Linear(channels * 3, 3)
        self.vel_head = nn.Linear(channels * 3, 3)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B, W, _ = x.shape
        mv = self.embedding(x.reshape(B * W, 7))
        return mv.reshape(B, W, self.channels, self.algebra.dim)

    def _tcn(self, mv: torch.Tensor) -> torch.Tensor:
        features = mv
        for tcn, norm, act in zip(self.tcn_layers, self.norms, self.activations):
            residual = features
            out = tcn(features)
            b, w, c, d = out.shape
            out = norm(out.reshape(b * w, c, d)).reshape(b, w, c, d)
            features = act(out) + residual
        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._tcn(self._embed(x))
        spatial = torch.stack(
            [features[..., 1], features[..., 2], features[..., 4]], dim=-1)
        flat = spatial.reshape(*features.shape[:2], self.channels * 3)
        return self.pos_head(flat), self.vel_head(flat)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pre-TCN embedding, post-TCN features) — diagnostics only."""
        embedding = self._embed(x)
        return embedding, self._tcn(embedding)


# ============================================================================
# Evaluation & post-training diagnostics
# ============================================================================

@torch.no_grad()
def evaluate_rmse(model: STATrajectoryNet, loader: DataLoader,
                  device: str) -> Dict[str, float]:
    model.eval()
    pos_sq, vel_sq, n = 0.0, 0.0, 0
    for sensor, pos_gt, vel_gt in loader:
        sensor = sensor.to(device, non_blocking=True)
        pos_gt = pos_gt.to(device, non_blocking=True)
        vel_gt = vel_gt.to(device, non_blocking=True)
        pos_pred, vel_pred = model(sensor)
        pos_sq += ((pos_pred - pos_gt) ** 2).sum().item()
        vel_sq += ((vel_pred - vel_gt) ** 2).sum().item()
        n += pos_gt.numel()
    return {
        'pos_rmse': math.sqrt(pos_sq / max(n, 1)),
        'vel_rmse': math.sqrt(vel_sq / max(n, 1)),
    }


@torch.no_grad()
def isometry_residual(model: STATrajectoryNet, loader: DataLoader,
                      algebra: CliffordAlgebra, device: str) -> float:
    """Mean |‖pre-TCN‖² − ‖post-TCN‖²| under the signature norm.

    Was the old ``IsometryLoss`` in the gradient path; here it measures
    how well the rotor TCN preserves the Cl(3,1) metric norm post-hoc.
    """
    model.eval()
    total, n = 0.0, 0
    for sensor, _, _ in loader:
        sensor = sensor.to(device, non_blocking=True)
        pre, post = model.features(sensor)
        sq_pre = signature_norm_squared(algebra, pre.reshape(-1, algebra.dim))
        sq_post = signature_norm_squared(algebra, post.reshape(-1, algebra.dim))
        total += (sq_pre - sq_post).abs().mean().item() * sensor.shape[0]
        n += sensor.shape[0]
        break  # one batch is enough for a stable mean
    return total / max(n, 1)


def post_training_diagnostics(
    model: STATrajectoryNet, test_loader: DataLoader,
    algebra: CliffordAlgebra, device: str, *,
    noisy_loader: Optional[DataLoader] = None,
) -> Dict[str, float]:
    rmse = evaluate_rmse(model, test_loader, device)
    diagnostics: Dict[str, float] = {
        'test_pos_rmse': rmse['pos_rmse'],
        'test_vel_rmse': rmse['vel_rmse'],
        'isometry_residual': isometry_residual(
            model, test_loader, algebra, device),
        'calib_bivector_norm': float(
            model.embedding.calib_bivector.detach().norm().item()),
    }
    feats = []
    with torch.no_grad():
        for sensor, _, _ in test_loader:
            _, post = model.features(sensor.to(device))
            feats.append(post)
            break
    spectrum = mean_grade_spectrum(feats, algebra)
    for k, val in enumerate(spectrum):
        diagnostics[f'grade_spectrum_{k}'] = float(val)
    if noisy_loader is not None:
        noisy_rmse = evaluate_rmse(model, noisy_loader, device)
        diagnostics['noise_robustness_pos_rmse'] = noisy_rmse['pos_rmse']
    return diagnostics


# ============================================================================
# Training entry point
# ============================================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device
    algebra = setup_algebra(p=3, q=1, device='cpu')

    train_ds = IMUTrajectoryDataset(
        args.data, args.window_size, args.stride, split='train', seed=args.seed)
    val_ds = IMUTrajectoryDataset(
        args.data, args.window_size, args.stride, split='val', seed=args.seed)
    test_ds = IMUTrajectoryDataset(
        args.data, args.window_size, args.stride, split='test', seed=args.seed)

    use_pin = device != 'cpu'
    num_workers = 2 if device != 'cpu' else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=use_pin, num_workers=num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        pin_memory=use_pin, num_workers=num_workers)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        pin_memory=use_pin, num_workers=num_workers)

    model = STATrajectoryNet(
        algebra, channels=args.channels, num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        gravity_bivector=train_ds.gravity_bivector,
    ).to(device)

    print_banner(
        'STA Trajectory Incubator — Cl(3,1) IMU reconstruction',
        signature='Cl(3, 1)  dim=16',
        channels=args.channels,
        num_layers=args.num_layers,
        window=args.window_size,
        natural_loss='MSE(pos) + 0.5 · MSE(vel)',
        parameters=f'{count_parameters(model):,}',
        train=f'{len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}',
    )

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    def loss_fn(_model, batch):
        sensor, pos_gt, vel_gt = (b.to(device, non_blocking=True) for b in batch)
        pos_pred, vel_pred = _model(sensor)
        return F.mse_loss(pos_pred, pos_gt) + 0.5 * F.mse_loss(vel_pred, vel_gt)

    def diag_fn(_model, _epoch) -> Dict[str, float]:
        return evaluate_rmse(_model, val_loader, device)

    history = run_supervised_loop(
        model, optimizer, loss_fn, train_loader,
        epochs=args.epochs, diag_interval=args.diag_interval, grad_clip=5.0,
        diag_fn=diag_fn, history_extra_keys=('pos_rmse', 'vel_rmse'),
    )

    noisy_loader = None
    if args.noise_scale > 0.0:
        noisy_ds = IMUTrajectoryDataset(
            args.data, args.window_size, args.stride,
            split='test', noise_scale=args.noise_scale, seed=args.seed)
        noisy_loader = DataLoader(noisy_ds, batch_size=args.batch_size)

    diagnostics = post_training_diagnostics(
        model, test_loader, algebra, device, noisy_loader=noisy_loader)
    print(report_diagnostics(
        diagnostics, title='STA trajectory post-training diagnostics'))

    if args.save_plots:
        ensure_output_dir(args.output_dir)
        path = save_training_curve(
            history, os.path.join(args.output_dir, 'training_curve.png'),
            title='STA Trajectory — supervised MSE')
        print(f'  curve saved to {path}')


def parse_args() -> argparse.Namespace:
    p = make_experiment_parser(
        'STA trajectory reconstruction — Cl(3,1) IMU incubator.',
        include=('seed', 'device', 'epochs', 'lr', 'batch_size',
                 'output_dir', 'save_plots', 'diag_interval'),
        defaults={'epochs': 200, 'lr': 0.001, 'batch_size': 64,
                  'output_dir': 'sta_plots', 'diag_interval': 10},
    )
    p.add_argument('--data', type=str,
                   default='../Trajecto/data/sta_experiment.h5',
                   help='HDF5 path from SimGenerator.')
    p.add_argument('--window-size', type=int, default=128)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--num-layers', type=int, default=5)
    p.add_argument('--kernel-size', type=int, default=3)
    p.add_argument('--noise-scale', type=float, default=1.0,
                   help='Noise robustness diagnostic scale (0 to disable).')
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == '__main__':
    main()
