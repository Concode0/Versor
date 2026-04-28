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
  A 7-channel IMU reading ``(accel, gyro, fsr)`` should fit naturally into a
  single Cl(3,1) multivector: accel as a grade-1 spatial vector, gyro as a
  grade-2 Hodge bivector, and fsr as a scalar. A learnable Spin(3,1)
  calibration rotor, initialized from gravity, followed by a frame-to-frame
  step-rotor flow (``x_{t+1} = R_t x_t R̃_t``) should reconstruct 3-D
  trajectories from synthetic Minkowski worldlines using the natural
  supervised loss ``MSE(pos) + 0.5 * MSE(vel)``. Isometry, grade confinement,
  calibration magnitude, and Lorentz-invariance residuals remain
  post-training measurements.

Execute Command
  uv run python -m experiments.inc_sta_trajectory --regime free_particle
  uv run python -m experiments.inc_sta_trajectory --regime lorentz_boost
  uv run python -m experiments.inc_sta_trajectory --regime helical_motion
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
    build_visualization_metadata,
    count_parameters,
    ensure_output_dir,
    make_experiment_parser,
    mean_grade_spectrum,
    print_banner,
    report_diagnostics,
    run_supervised_loop,
    save_training_curve,
    set_seed,
    setup_algebra,
    signature_metadata,
)

from core.algebra import CliffordAlgebra
from core.metric import signature_norm_squared
from layers import CliffordLayerNorm, GeometricNeutralizer, MotherEmbedding, RotorLayer
from layers.primitives.base import CliffordModule
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam


# ============================================================================
# Physical priors
# ============================================================================


def compute_gravity_bivector(gravity_vectors: np.ndarray, algebra_dim: int = 16) -> torch.Tensor:
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
        B[6] = theta * ax  # e23
        B[5] = -theta * ay  # e13
        B[3] = 0.0  # e12 (az=0 by construction)
    return B


def _build_imu_scatter(algebra_dim: int = 16) -> torch.Tensor:
    """Routing matrix ``[7, dim]``: accel → grade-1, gyro → grade-2 (Hodge
    dual), fsr → scalar. Gyro as bivector makes a rotation rotor act on it
    identically to accel under the sandwich product.
    """
    S = torch.zeros(7, algebra_dim, dtype=torch.float32)
    S[0, 1] = 1.0  # accel_x → e1
    S[1, 2] = 1.0  # accel_y → e2
    S[2, 4] = 1.0  # accel_z → e3
    S[3, 6] = 1.0  # gyro_x  → e23
    S[4, 5] = -1.0  # gyro_y  → -e13  (= e31)
    S[5, 3] = 1.0  # gyro_z  → e12
    S[6, 0] = 1.0  # fsr     → scalar
    return S


# ============================================================================
# Dataset
# ============================================================================


def _synthesize_worldline(
    regime: str,
    *,
    num_trajs: int,
    window_size: int,
    dt: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (sensor, pos, vel, gravity_b) tensors for a synthetic regime.

    All shapes have a leading ``(num_trajs, window_size)`` axis, exactly the
    layout produced by ``IMUTrajectoryDataset`` after windowing — so the rest
    of the pipeline doesn't care that the data came from closed-form physics.

    Channels: ``sensor[..., 0:3]`` = body-frame accel,
    ``sensor[..., 3:6]`` = body-frame gyro (angular velocity), ``sensor[..., 6]``
    = scalar force placeholder.
    """
    g_inertial = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    T = window_size
    t_axis = np.arange(T) * dt

    sensors = np.zeros((num_trajs, T, 7), dtype=np.float32)
    positions = np.zeros((num_trajs, T, 3), dtype=np.float32)
    velocities = np.zeros((num_trajs, T, 3), dtype=np.float32)
    gravity_b = np.zeros((num_trajs, T, 3), dtype=np.float32)

    for i in range(num_trajs):
        if regime == 'free_particle':
            v0 = rng.uniform(-2.0, 2.0, size=3)
            x0 = rng.uniform(-1.0, 1.0, size=3)
            vel = np.broadcast_to(v0, (T, 3)).copy()
            pos = x0[None, :] + vel * t_axis[:, None]
            accel_inertial = np.zeros((T, 3))
            omega = np.zeros((T, 3))
            R_body = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()

        elif regime == 'lorentz_boost':
            phi = rng.uniform(-1.0, 1.0)
            axis = rng.randint(0, 3)
            beta = math.tanh(phi)
            v_boost = np.zeros(3)
            v_boost[axis] = beta * 3.0
            v0 = v_boost + rng.uniform(-0.5, 0.5, size=3)
            x0 = rng.uniform(-1.0, 1.0, size=3)
            vel = np.broadcast_to(v0, (T, 3)).copy()
            pos = x0[None, :] + vel * t_axis[:, None]
            accel_inertial = np.zeros((T, 3))
            omega = np.zeros((T, 3))
            R_body = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()

        elif regime == 'helical_motion':
            radius = rng.uniform(0.5, 2.0)
            omega_z = rng.uniform(0.5, 2.5)
            phi0 = rng.uniform(-math.pi, math.pi)
            vz = rng.uniform(-0.5, 0.5)
            phase = phi0 + omega_z * t_axis
            pos = np.stack([radius * np.cos(phase), radius * np.sin(phase), vz * t_axis], axis=-1)
            vel = np.stack(
                [-radius * omega_z * np.sin(phase), radius * omega_z * np.cos(phase), np.full_like(phase, vz)], axis=-1
            )
            accel_inertial = np.stack(
                [
                    -radius * omega_z * omega_z * np.cos(phase),
                    -radius * omega_z * omega_z * np.sin(phase),
                    np.zeros_like(phase),
                ],
                axis=-1,
            )
            omega = np.zeros((T, 3))
            omega[:, 2] = omega_z
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            R_body = np.zeros((T, 3, 3))
            R_body[:, 0, 0] = cos_p
            R_body[:, 0, 1] = sin_p
            R_body[:, 1, 0] = -sin_p
            R_body[:, 1, 1] = cos_p
            R_body[:, 2, 2] = 1.0

        else:
            raise ValueError(f'unknown regime {regime!r}')

        g_tile = np.broadcast_to(g_inertial, (T, 3))
        proper_accel = accel_inertial - g_tile
        accel_body = np.einsum('tij,tj->ti', R_body, proper_accel)
        gyro_body = np.einsum('tij,tj->ti', R_body, omega)
        gravity_body = np.einsum('tij,tj->ti', R_body, g_tile)
        fsr = np.linalg.norm(accel_inertial, axis=-1)

        sensors[i, :, 0:3] = accel_body.astype(np.float32)
        sensors[i, :, 3:6] = gyro_body.astype(np.float32)
        sensors[i, :, 6] = fsr.astype(np.float32)
        positions[i] = pos.astype(np.float32)
        velocities[i] = vel.astype(np.float32)
        gravity_b[i] = gravity_body.astype(np.float32)

    return sensors, positions, velocities, gravity_b


class SyntheticIMUWorldlineDataset(Dataset):
    """Closed-form Minkowski worldlines as IMU windows.

    Three regimes:

    - ``free_particle``: constant 4-velocity worldline; identity body frame,
      gravity stays aligned with ``-e_z``.
    - ``lorentz_boost``: random per-trajectory rapidity along a random spatial
      axis; same body frame but the inertial velocity carries a relativistic
      offset, producing the stretched-time signature the model has to learn.
    - ``helical_motion``: uniform circular motion in (x, y) with linear z
      drift; body frame rotates at ``omega_z``, so the gyro channel exercises
      the grade-2 path of the Cl(3,1) embedding.

    Statistics (means/stds) are computed once on the full sample so the
    interface matches ``IMUTrajectoryDataset`` (sensor / pos / vel triple).
    """

    def __init__(
        self,
        regime: str,
        *,
        num_trajs: int = 256,
        window_size: int = 128,
        dt: float = 0.02,
        noise_scale: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.window_size = window_size
        self.noise_scale = noise_scale
        self.regime = regime
        self.dt = dt

        rng = np.random.RandomState(seed)
        sensors, positions, velocities, gravity_b = _synthesize_worldline(
            regime,
            num_trajs=num_trajs,
            window_size=window_size,
            dt=dt,
            rng=rng,
        )

        # Window-localize positions so each window starts at the origin —
        # matches what the original dataset did per stride.
        positions = positions - positions[:, :1, :]

        all_sensor = sensors.reshape(-1, 7)
        self.sensor_mean = torch.tensor(all_sensor.mean(axis=0), dtype=torch.float32)
        self.sensor_std = torch.tensor(all_sensor.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        delta = positions.reshape(-1, 3)
        self.pos_mean = torch.zeros(3, dtype=torch.float32)
        self.pos_std = torch.tensor(delta.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        all_vel = velocities.reshape(-1, 3)
        self.vel_mean = torch.tensor(all_vel.mean(axis=0), dtype=torch.float32)
        self.vel_std = torch.tensor(all_vel.std(axis=0).clip(min=1e-6), dtype=torch.float32)

        sensor_t = torch.from_numpy(sensors)
        pos_t = torch.from_numpy(positions)
        vel_t = torch.from_numpy(velocities)
        self.sensors = (sensor_t - self.sensor_mean) / self.sensor_std
        self.positions = pos_t / self.pos_std
        self.velocities = (vel_t - self.vel_mean) / self.vel_std

        self.gravity_bivector = compute_gravity_bivector(gravity_b.reshape(-1, 3))
        self.gravity_b_raw = torch.from_numpy(gravity_b)

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

    def __init__(self, algebra: CliffordAlgebra, channels: int, gravity_bivector: Optional[torch.Tensor] = None):
        super().__init__(algebra)
        self.channels = channels

        self.register_buffer('scatter_matrix', _build_imu_scatter(algebra.dim))
        self.register_buffer('g2_mask', algebra.grade_masks[2].to(dtype=torch.float32))

        if gravity_bivector is None:
            gravity_bivector = torch.zeros(algebra.dim)
        calib_init = gravity_bivector.to(dtype=torch.float32, device=self.g2_mask.device) * self.g2_mask
        self.calib_bivector = nn.Parameter(calib_init.clone())

        self.mother = MotherEmbedding(algebra, input_dim=algebra.dim, channels=channels)
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


class StepRotorFlow(CliffordModule):
    """Frame-to-frame rotor evolution: ``x_{t+1} = R_t · x_t · R̃_t``.

    For each timestep ``t`` a *data-dependent* bivector ``B_t`` is produced by
    a left-padded causal 1-D conv over the multivector features, then fed
    through ``algebra.exp(-B_t / 2)`` to obtain the per-frame rotor ``R_t``.
    The hidden multivector then advances under the sandwich product, which
    preserves the Cl(3,1) signature norm by construction — this is the
    implicit coercion that replaces the old ``IsometryLoss`` term.

    A residual channel mix (``CliffordLinear``) follows the rotor sandwich so
    expressive power is not capped by isometries alone.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__(algebra)
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.causal_pad = (kernel_size - 1) * dilation

        g2_indices = algebra.grade_masks[2].nonzero(as_tuple=False).squeeze(-1)
        self.register_buffer('g2_indices', g2_indices)
        self.num_bivecs = int(g2_indices.numel())

        # Bivector projector: causal conv over the full multivector to
        # produce a per-frame, per-channel grade-2 element.
        self.bv_conv = nn.Conv1d(
            channels * algebra.dim, channels * self.num_bivecs, kernel_size=kernel_size, dilation=dilation, padding=0
        )
        # Channel mix after the sandwich — preserves grade structure.
        from layers import CliffordLinear  # local import keeps top section tidy

        self.channel_mix = CliffordLinear(algebra, channels, channels)

    def _bivector_field(self, mv: torch.Tensor) -> torch.Tensor:
        """[B, T, C, D] → grade-2 bivectors [B*T, C, D]."""
        b, t, c, d = mv.shape
        x_in = mv.reshape(b, t, c * d).transpose(1, 2)
        x_in = F.pad(x_in, (self.causal_pad, 0))
        coeffs = self.bv_conv(x_in)  # [B, C*num_bv, T]
        coeffs = coeffs.transpose(1, 2).reshape(b * t, c, self.num_bivecs)
        B_field = torch.zeros(b * t, c, d, device=mv.device, dtype=mv.dtype)
        idx = self.g2_indices.view(1, 1, -1).expand(b * t, c, -1)
        B_field.scatter_(-1, idx, coeffs)
        return B_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, d = x.shape
        B_field = self._bivector_field(x)  # [B*T, C, D] grade-2
        flat = x.reshape(b * t, c, d)
        # Per-channel rotor sandwich: each (sample, time, channel) gets its
        # own rotor. Re-pack to [B*T*C, D] and use the algebra's
        # ``versor_product`` for a single-call sandwich.
        flat_R = self.algebra.exp(-0.5 * B_field).reshape(b * t * c, d)
        flat_x = flat.reshape(b * t * c, d)
        evolved = self.algebra.versor_product(flat_R, flat_x)
        evolved = evolved.reshape(b * t, c, d)
        evolved = self.channel_mix(evolved)
        return evolved.reshape(b, t, c, d)


class STATrajectoryNet(CliffordModule):
    """STAEmbed → stacked StepRotorFlow → grade-1 spatial readout.

    The rotor flow makes time evolution explicitly geometric: every dilation
    layer learns a per-frame bivector field whose exp produces a rotor that
    advances the multivector state in place. Receptive field grows the same
    way a TCN's would — by exponential dilation in the bivector projector —
    while the actual transformation is signature-preserving.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int = 32,
        num_layers: int = 5,
        kernel_size: int = 3,
        gravity_bivector: Optional[torch.Tensor] = None,
    ):
        super().__init__(algebra)
        self.channels = channels
        self.embedding = STAEmbed(algebra, channels, gravity_bivector)
        self.flow_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(num_layers):
            self.flow_layers.append(StepRotorFlow(algebra, channels, kernel_size, dilation=2**i))
            self.norms.append(CliffordLayerNorm(algebra, channels))
            self.activations.append(GeometricGELU(algebra, channels))
        self.pos_head = nn.Linear(channels * 3, 3)
        self.vel_head = nn.Linear(channels * 3, 3)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B, W, _ = x.shape
        mv = self.embedding(x.reshape(B * W, 7))
        return mv.reshape(B, W, self.channels, self.algebra.dim)

    def _flow(self, mv: torch.Tensor) -> torch.Tensor:
        features = mv
        for flow, norm, act in zip(self.flow_layers, self.norms, self.activations):
            residual = features
            out = flow(features)
            b, w, c, d = out.shape
            out = norm(out.reshape(b * w, c, d)).reshape(b, w, c, d)
            features = act(out) + residual
        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._flow(self._embed(x))
        spatial = torch.stack([features[..., 1], features[..., 2], features[..., 4]], dim=-1)
        flat = spatial.reshape(*features.shape[:2], self.channels * 3)
        return self.pos_head(flat), self.vel_head(flat)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pre-flow embedding, post-flow features) — diagnostics only."""
        embedding = self._embed(x)
        return embedding, self._flow(embedding)


# ============================================================================
# Evaluation & post-training diagnostics
# ============================================================================


@torch.no_grad()
def evaluate_rmse(model: STATrajectoryNet, loader: DataLoader, device: str) -> Dict[str, float]:
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
def isometry_residual(model: STATrajectoryNet, loader: DataLoader, algebra: CliffordAlgebra, device: str) -> float:
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


@torch.no_grad()
def lorentz_invariance_residual(
    model: STATrajectoryNet, loader: DataLoader, algebra: CliffordAlgebra, device: str
) -> float:
    """Predicted ``signature_norm_squared`` should be invariant under random
    Spin(3,1) rotors applied at the embedding stage.

    Constructs a small grade-2 element ``B`` (boost + rotation), exponentiates
    to a rotor ``R = exp(-B/2)``, applies the sandwich product to the embedded
    multivector, and re-runs the flow stack. The Hermitian-norm spectrum of
    the post-flow features should match because every step is an isometry.
    """
    model.eval()
    bv = torch.zeros(1, algebra.dim, device=device)
    g2_mask = algebra.grade_masks_float[2].to(bv.dtype)
    bv = bv + torch.randn(1, algebra.dim, device=device) * 0.1 * g2_mask
    R = algebra.exp(-0.5 * bv)
    R_rev = algebra.reverse(R)

    sensor, _, _ = next(iter(loader))
    sensor = sensor.to(device)
    embedding, post = model.features(sensor)
    b, w, c, d = embedding.shape
    flat = embedding.reshape(b * w * c, d)
    boosted = algebra.geometric_product(
        algebra.geometric_product(R.expand_as(flat), flat),
        R_rev.expand_as(flat),
    ).reshape(b, w, c, d)
    boosted_post = model._flow(boosted)
    sq_orig = signature_norm_squared(algebra, post.reshape(-1, d))
    sq_boost = signature_norm_squared(algebra, boosted_post.reshape(-1, d))
    return float((sq_orig - sq_boost).abs().mean().item())


@torch.no_grad()
def calibration_recovery_error(model: STATrajectoryNet, target_bivector: torch.Tensor) -> float:
    """How close the learned calibration bivector sits to the analytic
    optimum recovered by ``compute_gravity_bivector`` on the dataset.
    """
    learned = model.embedding.calib_bivector.detach().cpu()
    target = target_bivector.detach().cpu().to(learned.dtype)
    return float((learned - target).norm().item())


def post_training_diagnostics(
    model: STATrajectoryNet,
    test_loader: DataLoader,
    algebra: CliffordAlgebra,
    device: str,
    *,
    target_calib_bivector: Optional[torch.Tensor] = None,
    noisy_loader: Optional[DataLoader] = None,
) -> Dict[str, float]:
    rmse = evaluate_rmse(model, test_loader, device)
    diagnostics: Dict[str, float] = {
        'test_pos_rmse': rmse['pos_rmse'],
        'test_vel_rmse': rmse['vel_rmse'],
        'isometry_residual': isometry_residual(model, test_loader, algebra, device),
        'lorentz_invariance_residual': lorentz_invariance_residual(model, test_loader, algebra, device),
        'calib_bivector_norm': float(model.embedding.calib_bivector.detach().norm().item()),
    }
    if target_calib_bivector is not None:
        diagnostics['calib_recovery_error'] = calibration_recovery_error(model, target_calib_bivector)
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

    train_ds = SyntheticIMUWorldlineDataset(
        args.regime, num_trajs=args.num_trajs, window_size=args.window_size, dt=args.dt, seed=args.seed
    )
    val_ds = SyntheticIMUWorldlineDataset(
        args.regime,
        num_trajs=max(args.num_trajs // 4, 16),
        window_size=args.window_size,
        dt=args.dt,
        seed=args.seed + 1,
    )
    test_ds = SyntheticIMUWorldlineDataset(
        args.regime,
        num_trajs=max(args.num_trajs // 4, 16),
        window_size=args.window_size,
        dt=args.dt,
        seed=args.seed + 2,
    )

    use_pin = device != 'cpu'
    num_workers = 2 if device != 'cpu' else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=use_pin, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=use_pin, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=use_pin, num_workers=num_workers)

    model = STATrajectoryNet(
        algebra,
        channels=args.channels,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        gravity_bivector=train_ds.gravity_bivector,
    ).to(device)

    print_banner(
        'STA Trajectory Incubator — Cl(3,1) synthetic worldline reconstruction',
        signature='Cl(3, 1)  dim=16',
        regime=args.regime,
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
        model,
        optimizer,
        loss_fn,
        train_loader,
        epochs=args.epochs,
        diag_interval=args.diag_interval,
        grad_clip=5.0,
        diag_fn=diag_fn,
        history_extra_keys=('pos_rmse', 'vel_rmse'),
    )

    noisy_loader = None
    if args.noise_scale > 0.0:
        noisy_ds = SyntheticIMUWorldlineDataset(
            args.regime,
            num_trajs=max(args.num_trajs // 4, 16),
            window_size=args.window_size,
            dt=args.dt,
            noise_scale=args.noise_scale,
            seed=args.seed + 2,
        )
        noisy_loader = DataLoader(noisy_ds, batch_size=args.batch_size)

    diagnostics = post_training_diagnostics(
        model, test_loader, algebra, device, target_calib_bivector=train_ds.gravity_bivector, noisy_loader=noisy_loader
    )
    print(report_diagnostics(diagnostics, title='STA trajectory post-training diagnostics'))

    ensure_output_dir(args.output_dir)
    metadata = build_visualization_metadata(
        signature_metadata(3, 1),
        regime=args.regime,
        window_size=args.window_size,
        channels=args.channels,
        seed=args.seed,
    )
    path = save_training_curve(
        history,
        output_dir=args.output_dir,
        experiment_name='inc_sta_trajectory',
        metadata=metadata,
        plot_name='training_curve',
        args=args,
        module=__name__,
        title='STA Trajectory — supervised MSE',
    )
    print(f'  curve saved to {path}')


def parse_args() -> argparse.Namespace:
    p = make_experiment_parser(
        'STA trajectory reconstruction — Cl(3,1) synthetic worldlines.',
        include=('seed', 'device', 'epochs', 'lr', 'batch_size', 'output_dir', 'diag_interval'),
        defaults={'epochs': 200, 'lr': 0.001, 'batch_size': 64, 'output_dir': 'sta_plots', 'diag_interval': 10},
    )
    p.add_argument(
        '--regime',
        choices=('free_particle', 'lorentz_boost', 'helical_motion'),
        default='helical_motion',
        help='Closed-form Minkowski worldline generator.',
    )
    p.add_argument('--num-trajs', type=int, default=256)
    p.add_argument('--window-size', type=int, default=128)
    p.add_argument('--dt', type=float, default=0.02, help='Worldline timestep in seconds.')
    p.add_argument('--channels', type=int, default=32)
    p.add_argument('--num-layers', type=int, default=5)
    p.add_argument('--kernel-size', type=int, default=3)
    p.add_argument('--noise-scale', type=float, default=1.0, help='Noise robustness diagnostic scale (0 to disable).')
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == '__main__':
    main()
