# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""
==============================================================================
VERSOR EXPERIMENT: MATHEMATICAL DEBUGGER
==============================================================================

Does Versor's geometric bias reconstruct a plane-wave electromagnetic field
(F = E + I·B, pure grade-2 in Cl(3,1)) through backpropagation without
breaking? A single supervised loss — masked MSE on the six grade-2 slots —
drives learning. Grade-2 purity, Hodge-dual symmetry, Lorentz invariants
(|E|² − |B|², E·B), and the source-free Maxwell residual ||∇F|| are all
**post-training measurements**, not gradient terms.

==============================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments._lib import (
    add_standard_args, ensure_output_dir, print_banner, report_diagnostics,
    run_supervised_loop, save_training_curve, set_seed, setup_algebra,
)
from core.algebra import CliffordAlgebra
from layers.primitives.base import CliffordModule
from layers import BladeSelector, CliffordLayerNorm, CliffordLinear, RotorLayer
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam


# ---------------------------------------------------------------------------
# Cl(3,1) basis layout
# ---------------------------------------------------------------------------

E_BLADES = (9, 10, 12)          # e14, e24, e34
B_BLADES = (6, 5, 3)            # e23, e13, e12
SPACETIME_BLADES = (8, 1, 2, 4) # t->e4, x->e1, y->e2, z->e3
PSEUDOSCALAR = 15
G2_SLOTS = list(E_BLADES) + list(B_BLADES)


def _assert_cl31(algebra: CliffordAlgebra) -> None:
    if algebra.p != 3 or algebra.q != 1 or getattr(algebra, 'r', 0) != 0:
        raise ValueError(
            f"Expected Cl(3,1); got Cl({algebra.p},{algebra.q},{getattr(algebra,'r',0)})"
        )


def _boost_bivector(algebra, axis: int) -> torch.Tensor:
    bv = torch.zeros(algebra.dim, device=algebra.device)
    bv[E_BLADES[axis]] = 1.0
    return bv


# ---------------------------------------------------------------------------
# Plane-wave ground truth
# ---------------------------------------------------------------------------

def _random_unit_vector(rng, dim=3):
    v = rng.randn(dim)
    return v / (np.linalg.norm(v) + 1e-12)


def _sample_wave(rng, kmin: float, kmax: float) -> Dict[str, np.ndarray]:
    k_hat = _random_unit_vector(rng)
    k_mag = float(rng.uniform(kmin, kmax))
    k = k_mag * k_hat
    u = _random_unit_vector(rng)
    u -= np.dot(u, k_hat) * k_hat
    if np.linalg.norm(u) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0]) if abs(k_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = fallback - np.dot(fallback, k_hat) * k_hat
    e_hat = u / (np.linalg.norm(u) + 1e-12)
    b_hat = np.cross(k_hat, e_hat)
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    return {'k': k, 'e_hat': e_hat, 'b_hat': b_hat, 'phase': phase, 'w': k_mag}


def compute_plane_wave_F(tr: np.ndarray, wave: Dict[str, np.ndarray]) -> np.ndarray:
    t = tr[:, 0]; r = tr[:, 1:4]
    c = np.cos(r @ wave['k'] - wave['w'] * t + wave['phase'])
    F = np.zeros((tr.shape[0], 16), dtype=np.float64)
    for axis in range(3):
        F[:, E_BLADES[axis]] = wave['e_hat'][axis] * c
        F[:, B_BLADES[axis]] = wave['b_hat'][axis] * c
    return F


class PlaneWaveDataset(Dataset):
    """Spacetime-event / bivector-F pairs drawn from a small pool of plane waves."""

    def __init__(self, num_samples: int, num_waves: int,
                 tmax: float, rmax: float, kmin: float, kmax: float,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        self.waves = [_sample_wave(rng, kmin, kmax) for _ in range(num_waves)]
        self.wave_idx = rng.randint(0, num_waves, size=num_samples)
        t = rng.uniform(-tmax, tmax, num_samples)
        r = rng.uniform(-rmax, rmax, size=(num_samples, 3))
        self.tr = np.concatenate([t[:, None], r], axis=1)
        F = np.zeros((num_samples, 16), dtype=np.float64)
        k_feat = np.zeros((num_samples, 3), dtype=np.float64)
        for i in range(num_waves):
            mask = self.wave_idx == i
            if mask.any():
                F[mask] = compute_plane_wave_F(self.tr[mask], self.waves[i])
                k_feat[mask] = self.waves[i]['k']
        inputs = torch.zeros(num_samples, 16, dtype=torch.float32)
        for axis, blade in enumerate(SPACETIME_BLADES):
            inputs[:, blade] = torch.tensor(self.tr[:, axis], dtype=torch.float32)
        self.inputs = inputs
        self.targets = torch.tensor(F, dtype=torch.float32)
        self.k_feat = torch.tensor(k_feat, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.k_feat[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class MaxwellNet(CliffordModule):
    """GBN predicting F at a spacetime event, conditioned on the wave vector k."""

    def __init__(self, algebra, hidden_dim: int = 64,
                 num_layers: int = 6, num_freqs: int = 32):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim
        self.register_buffer('freq_bands', torch.randn(7, num_freqs) * 0.5)
        input_dim = 7 + 2 * num_freqs
        self.input_lift = nn.Linear(input_dim, hidden_dim * algebra.dim)
        self.input_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm':   CliffordLayerNorm(algebra, hidden_dim),
                'rotor':  RotorLayer(algebra, hidden_dim),
                'act':    GeometricGELU(algebra, channels=hidden_dim),
                'linear': CliffordLinear(algebra, hidden_dim, hidden_dim),
            })
            for _ in range(num_layers)
        ])
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)
        self.output_proj = CliffordLinear(algebra, hidden_dim, 1)

    def forward(self, x_mv: torch.Tensor, k_feat: torch.Tensor) -> torch.Tensor:
        B = x_mv.shape[0]
        tr = torch.stack([x_mv[:, blade] for blade in SPACETIME_BLADES], dim=-1)
        features_raw = torch.cat([tr, k_feat], dim=-1)
        proj = features_raw @ self.freq_bands
        features = torch.cat([features_raw, torch.sin(proj), torch.cos(proj)], dim=-1)
        h = self.input_norm(
            self.input_lift(features).reshape(B, self.hidden_dim, self.algebra.dim)
        )
        for block in self.blocks:
            res = h
            h = block['norm'](h)
            h = block['rotor'](h)
            h = block['act'](h)
            h = block['linear'](h)
            h = res + h
        h = self.output_norm(h)
        h = self.blade_selector(h)
        return self.output_proj(h).squeeze(1)


# ---------------------------------------------------------------------------
# Normalization bundle
# ---------------------------------------------------------------------------

class _Norms:
    def __init__(self, input_mean, input_std, k_mean, k_std, target_mean, target_std):
        self.input_mean, self.input_std = input_mean, input_std
        self.k_mean, self.k_std = k_mean, k_std
        self.target_mean, self.target_std = target_mean, target_std

    def to(self, device):
        for attr in ('input_mean', 'input_std', 'k_mean', 'k_std',
                     'target_mean', 'target_std'):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def apply_input(self, inputs, k_feat):
        return (inputs - self.input_mean) / self.input_std, (k_feat - self.k_mean) / self.k_std


def _build_norms(train_ds: PlaneWaveDataset) -> _Norms:
    return _Norms(
        input_mean=train_ds.inputs.mean(dim=0),
        input_std=train_ds.inputs.std(dim=0).clamp(min=1e-4),
        k_mean=train_ds.k_feat.mean(dim=0),
        k_std=train_ds.k_feat.std(dim=0).clamp(min=1e-4),
        target_mean=train_ds.targets.mean(dim=0),
        target_std=train_ds.targets.std(dim=0).clamp(min=1e-4),
    )


# ---------------------------------------------------------------------------
# Single natural loss
# ---------------------------------------------------------------------------

def _masked_mse(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    diff_sq = (pred - target) ** 2
    return (diff_sq * mask).sum() / (mask.sum() * pred.shape[0] + 1e-12)


# ---------------------------------------------------------------------------
# Post-training diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_denorm(model, inputs, k_feat, norms) -> torch.Tensor:
    inp_n, k_n = norms.apply_input(inputs, k_feat)
    return model(inp_n, k_n) * norms.target_std + norms.target_mean


@torch.no_grad()
def grade2_purity_fraction(model, dataset, norms, device) -> float:
    """||P_2(F_pred)||^2 / ||F_pred||^2 over the test set."""
    inputs = dataset.inputs.to(device)
    k_feat = dataset.k_feat.to(device)
    pred = _predict_denorm(model, inputs, k_feat, norms)
    total_energy = (pred ** 2).sum().clamp(min=1e-12)
    g2_energy = (pred[:, G2_SLOTS] ** 2).sum()
    return (g2_energy / total_energy).item()


@torch.no_grad()
def test_l2_g2(model, loader, norms, device) -> float:
    total_err, n = 0.0, 0
    for inputs, k_feat, targets in loader:
        inputs = inputs.to(device); k_feat = k_feat.to(device); targets = targets.to(device)
        pred_denorm = _predict_denorm(model, inputs, k_feat, norms)
        diff = pred_denorm[:, G2_SLOTS] - targets[:, G2_SLOTS]
        err = (diff ** 2).sum(dim=-1).sqrt()
        total_err += err.sum().item(); n += err.shape[0]
    return total_err / max(n, 1)


@torch.no_grad()
def lorentz_invariants(model, algebra, dataset, norms, device, n_tests=200) -> Dict[str, float]:
    """Residuals of the two Lorentz invariants of F under a random boost."""
    rng = np.random.RandomState(1234)
    idx = rng.randint(0, len(dataset), size=n_tests)
    inputs = dataset.inputs[idx].to(device)
    k_feat = dataset.k_feat[idx].to(device)
    targets = dataset.targets[idx].to(device)
    pred = _predict_denorm(model, inputs, k_feat, norms)

    def _scalars(F):
        FF = algebra.geometric_product(F, F)
        return FF[..., 0], FF[..., PSEUDOSCALAR]

    phi = torch.tensor(rng.uniform(-1.0, 1.0, size=(n_tests,)),
                       dtype=torch.float32, device=device)
    axis = rng.randint(0, 3, size=n_tests)
    B = torch.zeros_like(pred)
    for i in range(n_tests):
        B[i] = phi[i] * _boost_bivector(algebra, int(axis[i])).to(device)
    R = algebra.exp(-0.5 * B)
    R_rev = algebra.reverse(R)

    def _boost(F):
        return algebra.geometric_product(algebra.geometric_product(R, F), R_rev)

    s1_t, s2_t = _scalars(targets); s1_tb, s2_tb = _scalars(_boost(targets))
    s1_p, s2_p = _scalars(pred);    s1_pb, s2_pb = _scalars(_boost(pred))
    return {
        'gt_inv1_residual':    (s1_tb - s1_t).abs().mean().item(),
        'gt_inv2_residual':    (s2_tb - s2_t).abs().mean().item(),
        'pred_inv1_residual':  (s1_pb - s1_p).abs().mean().item(),
        'pred_inv2_residual':  (s2_pb - s2_p).abs().mean().item(),
        'pred_vs_true_inv1':   (s1_p - s1_t).abs().mean().item(),
        'pred_vs_true_inv2':   (s2_p - s2_t).abs().mean().item(),
    }


@torch.no_grad()
def dual_symmetry_residual(model, algebra, dataset, norms, device, n_tests=200) -> float:
    rng = np.random.RandomState(4321)
    idx = rng.randint(0, len(dataset), size=n_tests)
    inputs = dataset.inputs[idx].to(device)
    k_feat = dataset.k_feat[idx].to(device)
    targets = dataset.targets[idx].to(device)
    pred = _predict_denorm(model, inputs, k_feat, norms)
    I = torch.zeros(algebra.dim, device=device)
    I[PSEUDOSCALAR] = 1.0
    I = I.unsqueeze(0).expand(n_tests, -1)
    diff = algebra.geometric_product(I, pred) - algebra.geometric_product(I, targets)
    return (diff[:, G2_SLOTS] ** 2).sum(dim=-1).sqrt().mean().item()


@torch.no_grad()
def maxwell_fd_residual(model, algebra, dataset, norms, device,
                        n_tests=200, h: float = 1e-3) -> Tuple[float, float]:
    """Source-free ||∇F|| via central differences on the predicted field."""
    rng = np.random.RandomState(5678)
    idx = rng.randint(0, len(dataset), size=n_tests)
    base_inputs = dataset.inputs[idx].to(device)
    k_feat = dataset.k_feat[idx].to(device)

    def _predict(inputs):
        return _predict_denorm(model, inputs, k_feat, norms)

    def _true_at(inputs):
        tr_np = torch.stack(
            [inputs[:, b] for b in SPACETIME_BLADES], dim=-1,
        ).cpu().numpy()
        out = np.zeros((inputs.shape[0], 16), dtype=np.float64)
        for j, k in enumerate(idx):
            out[j] = compute_plane_wave_F(tr_np[j:j+1], dataset.waves[dataset.wave_idx[k]])[0]
        return torch.tensor(out, dtype=torch.float32, device=device)

    axis_blades = (1, 2, 4, 8)
    shifted = []
    for blade in axis_blades:
        for sign in (-1.0, +1.0):
            inp_s = base_inputs.clone()
            inp_s[:, blade] = inp_s[:, blade] + sign * h
            shifted.append(inp_s)

    def _grad_norm(values):
        dF = [(values[2*a+1] - values[2*a]) / (2.0 * h) for a in range(4)]
        total = torch.zeros_like(dF[0])
        for a, blade in enumerate(axis_blades):
            e_a = torch.zeros(algebra.dim, device=device)
            e_a[blade] = 1.0
            total = total + algebra.geometric_product(
                e_a.unsqueeze(0).expand_as(dF[a]), dF[a],
            )
        return (total ** 2).sum(dim=-1).sqrt()

    pred_norm = _grad_norm([_predict(s) for s in shifted]).mean().item()
    true_norm = _grad_norm([_true_at(s) for s in shifted]).mean().item()
    return pred_norm, true_norm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    set_seed(args.seed)
    device = args.device
    algebra = setup_algebra(p=3, q=1, device=device)
    _assert_cl31(algebra)

    print_banner(
        f'Maxwell Reconstruction — Cl(3,1)',
        field='F = E + I·B  (pure grade-2 bivector)',
        natural_loss='Masked MSE on grade-2 slots {e14,e24,e34,e23,e13,e12}',
    )

    train_ds = PlaneWaveDataset(
        num_samples=args.num_train, num_waves=args.num_waves,
        tmax=args.tmax, rmax=args.rmax, kmin=args.kmin, kmax=args.kmax,
        seed=args.seed,
    )
    test_ds = PlaneWaveDataset(
        num_samples=args.num_test, num_waves=args.num_waves,
        tmax=args.tmax, rmax=args.rmax, kmin=args.kmin, kmax=args.kmax,
        seed=args.seed + 1,
    )
    # Ensure test set samples target the same wave pool as train.
    test_ds.waves = train_ds.waves
    test_ds.targets = torch.tensor(
        np.concatenate([
            compute_plane_wave_F(test_ds.tr[i:i+1], train_ds.waves[int(w)])
            for i, w in enumerate(test_ds.wave_idx)
        ], axis=0),
        dtype=torch.float32,
    )
    test_ds.k_feat = torch.stack([
        torch.tensor(train_ds.waves[int(w)]['k'], dtype=torch.float32)
        for w in test_ds.wave_idx
    ])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    norms = _build_norms(train_ds).to(device)
    g2_mask = algebra.grade_masks[2].to(dtype=torch.float32, device=device)

    model = MaxwellNet(algebra, hidden_dim=args.hidden_dim,
                       num_layers=args.num_layers, num_freqs=args.num_freqs).to(device)
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')
    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    def loss_fn(_model, batch):
        inputs, k_feat, targets = [b.to(device) for b in batch]
        inp_n, k_n = norms.apply_input(inputs, k_feat)
        norm_targets = (targets - norms.target_mean) / norms.target_std
        return _masked_mse(_model(inp_n, k_n), norm_targets, g2_mask)

    def diag_fn(_model, _epoch) -> Dict[str, float]:
        return {'test_l2_g2': test_l2_g2(_model, test_loader, norms, device)}

    history = run_supervised_loop(
        model, optimizer, loss_fn, train_loader,
        epochs=args.epochs, diag_interval=args.diag_interval, grad_clip=1.0,
    )

    # Post-training measurements — every ex-loss-term is now a number.
    inv = lorentz_invariants(model, algebra, test_ds, norms, device)
    maxw_pred_res, maxw_true_res = maxwell_fd_residual(
        model, algebra, test_ds, norms, device,
    )
    diagnostics = {
        'test_l2_g2':                  test_l2_g2(model, test_loader, norms, device),
        'grade2_purity':               grade2_purity_fraction(model, test_ds, norms, device),
        'dual_symmetry_residual':      dual_symmetry_residual(model, algebra, test_ds, norms, device),
        'gt_inv1_residual':            inv['gt_inv1_residual'],
        'gt_inv2_residual':            inv['gt_inv2_residual'],
        'pred_inv1_residual':          inv['pred_inv1_residual'],
        'pred_inv2_residual':          inv['pred_inv2_residual'],
        'pred_vs_true_inv1':           inv['pred_vs_true_inv1'],
        'pred_vs_true_inv2':           inv['pred_vs_true_inv2'],
        'maxwell_pred_residual':       maxw_pred_res,
        'maxwell_true_fd_floor':       maxw_true_res,
    }
    print(report_diagnostics(
        diagnostics, title='Maxwell post-training physics diagnostics',
    ))

    if args.save_plots:
        ensure_output_dir(args.output_dir)
        path = save_training_curve(
            history, os.path.join(args.output_dir, 'training_curve.png'),
            title='Maxwell — grade-2 reconstruction loss',
        )
        print(f'  curve saved to {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Maxwell equations debugger in Cl(3,1)')
    add_standard_args(
        p,
        include=('seed', 'device', 'epochs', 'lr', 'batch_size',
                 'output_dir', 'save_plots', 'diag_interval'),
        defaults={'epochs': 200, 'lr': 0.001, 'batch_size': 128,
                  'output_dir': 'maxwell_plots', 'diag_interval': 20},
    )
    p.add_argument('--tmax', type=float, default=5.0)
    p.add_argument('--rmax', type=float, default=5.0)
    p.add_argument('--kmin', type=float, default=0.5)
    p.add_argument('--kmax', type=float, default=2.0)
    p.add_argument('--num-waves', type=int, default=8)
    p.add_argument('--num-train', type=int, default=5000)
    p.add_argument('--num-test', type=int, default=1000)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--num-freqs', type=int, default=32)
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == '__main__':
    main()
