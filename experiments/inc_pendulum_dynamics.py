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

Hamiltonian Phase-Space Flow in Cl(2,2) (double pendulum) or Cl(3,0) (Lorenz).

Hypothesis
  The symplectic structure of a Hamiltonian system — position- and momentum-
  like variables with opposite signatures — is the natural inductive bias of a
  GBN in Cl(p, q). A residual rotor stack maps phase-space state to its one-
  step image under the flow; energy conservation, grade confinement on the
  even subalgebra, and finite Lyapunov separation emerge without being forced
  as loss terms.

Natural loss
  Single MSE on the grade-1 readout. Every ex-loss-term (energy drift, even/
  odd grade ratio, chaotic divergence) is a post-training measurement.

Run
  uv run python -m experiments.inc_pendulum_dynamics --epochs 200
  uv run python -m experiments.inc_pendulum_dynamics --system lorenz --p 3 --q 0
"""

from __future__ import annotations

import argparse
import math
import os, sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments._lib import (
    add_standard_args, apply_residual_block, count_parameters,
    ensure_output_dir, extract_grade1, gbn_residual_block, mean_grade_spectrum,
    print_banner, report_diagnostics, run_supervised_loop, save_training_curve,
    set_seed, setup_algebra,
)

from core.algebra import CliffordAlgebra
from layers import BladeSelector, CliffordLayerNorm, CliffordLinear
from layers.primitives.base import CliffordModule
from optimizers.riemannian import RiemannianAdam


# ==============================================================================
# Physics Simulation
# ==============================================================================

class DoublePendulumODE:
    """Double pendulum with Lagrangian mechanics. State = (θ1, θ2, ω1, ω2)."""

    def __init__(self, l1: float = 1.0, l2: float = 1.0,
                 m1: float = 1.0, m2: float = 1.0, g: float = 9.81):
        self.l1, self.l2 = l1, l2
        self.m1, self.m2 = m1, m2
        self.g = g

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        t1, t2, w1, w2 = state
        d = t1 - t2
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        s, c = np.sin(d), np.cos(d)
        M11, M12 = l1 * (m1 + m2), m2 * l2 * c
        M21, M22 = m2 * l1 * c, m2 * l2
        f1 = -m2 * l2 * w2 ** 2 * s - (m1 + m2) * g * np.sin(t1)
        f2 = m2 * l1 * w1 ** 2 * s - m2 * g * np.sin(t2)
        det = M11 * M22 - M12 * M21
        dw1 = (M22 * f1 - M12 * f2) / det
        dw2 = (M11 * f2 - M21 * f1) / det
        return np.array([w1, w2, dw1, dw2], dtype=np.float64)

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def energy_batch(self, states: np.ndarray) -> np.ndarray:
        """Hamiltonian H = T + V for a batch of states [N, 4]."""
        t1, t2, w1, w2 = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        T = (0.5 * (m1 + m2) * l1 ** 2 * w1 ** 2
             + 0.5 * m2 * l2 ** 2 * w2 ** 2
             + m2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2))
        V = -(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2)
        return T + V

    def generate_trajectory(self, x0: np.ndarray,
                            n_steps: int, dt: float) -> np.ndarray:
        traj = np.zeros((n_steps, x0.shape[0]), dtype=np.float64)
        traj[0] = x0.copy()
        state = x0.copy()
        for i in range(1, n_steps):
            state = self.rk4_step(state, dt)
            traj[i] = state
        return traj

    def random_ic(self, rng: np.random.RandomState,
                  regime: str = 'mixed') -> np.ndarray:
        if regime == 'mixed':
            regime = rng.choice(['regular', 'chaotic'])
        if regime == 'regular':
            lo, hi, wmax = -np.pi / 4, np.pi / 4, 1.0
        else:
            lo, hi, wmax = -np.pi, np.pi, 3.0
        return np.array([rng.uniform(lo, hi), rng.uniform(lo, hi),
                         rng.uniform(-wmax, wmax), rng.uniform(-wmax, wmax)],
                        dtype=np.float64)


class LorenzODE:
    """Lorenz (1963) dissipative attractor. State = (x, y, z)."""

    def __init__(self, sigma: float = 10.0, rho: float = 28.0,
                 beta: float = 8.0 / 3.0):
        self.sigma, self.rho, self.beta = sigma, rho, beta

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array([self.sigma * (y - x),
                         x * (self.rho - z) - y,
                         x * y - self.beta * z], dtype=np.float64)

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def generate_trajectory(self, x0: np.ndarray,
                            n_steps: int, dt: float) -> np.ndarray:
        traj = np.zeros((n_steps, 3), dtype=np.float64)
        traj[0] = x0.copy()
        state = x0.copy()
        for i in range(1, n_steps):
            state = self.rk4_step(state, dt)
            traj[i] = state
        return traj

    def random_ic(self, rng: np.random.RandomState,
                  regime: str = 'mixed') -> np.ndarray:
        if regime == 'mixed':
            regime = rng.choice(['regular', 'chaotic'])
        center = math.sqrt(self.beta * (self.rho - 1))
        sign = rng.choice([-1, 1])
        ic = np.array([sign * center, sign * center, self.rho - 1],
                      dtype=np.float64)
        scale = 1.0 if regime == 'regular' else 5.0
        return ic + rng.randn(3) * scale


# ==============================================================================
# Dataset
# ==============================================================================

class TrajectoryDataset(Dataset):
    """One-step-ahead (state_t, state_{t+k}) pairs, z-score normalized."""

    def __init__(self, ode, n_traj: int = 500, traj_len: int = 200,
                 dt: float = 0.01, n_steps_ahead: int = 1,
                 regime: str = 'mixed', seed: int = 42):
        rng = np.random.RandomState(seed)
        self.ode = ode
        self.dt = dt
        states_list, nexts_list = [], []
        for _ in range(n_traj):
            x0 = ode.random_ic(rng, regime)
            traj = ode.generate_trajectory(x0, traj_len + n_steps_ahead, dt)
            for t in range(traj_len):
                states_list.append(traj[t].astype(np.float32))
                nexts_list.append(traj[t + n_steps_ahead].astype(np.float32))

        states = np.asarray(states_list, dtype=np.float32)
        nexts = np.asarray(nexts_list, dtype=np.float32)
        self.state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
        self.state_std = torch.tensor(
            states.std(axis=0).clip(min=1e-6), dtype=torch.float32)
        self.states = torch.from_numpy(
            (states - self.state_mean.numpy()) / self.state_std.numpy())
        self.nexts = torch.from_numpy(
            (nexts - self.state_mean.numpy()) / self.state_std.numpy())

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.nexts[idx]

    def denormalize(self, state_norm: np.ndarray) -> np.ndarray:
        return state_norm * self.state_std.numpy() + self.state_mean.numpy()

    def normalize(self, state_phys: np.ndarray) -> np.ndarray:
        return (state_phys - self.state_mean.numpy()) / self.state_std.numpy()


# ==============================================================================
# Model
# ==============================================================================

class HamiltonianRotorNet(CliffordModule):
    """GBN for Hamiltonian phase-space flow.

    Lift grade-1 state → N × (norm → rotor → act → linear + skip) → grade-1
    readout. The rotor sandwich x → R x R̃ carries the symplectic inductive
    bias; a channel-wise ``CliffordLinear`` mixes multivectors between blocks.
    """

    def __init__(self, algebra: CliffordAlgebra, state_dim: int,
                 hidden_dim: int = 64, num_layers: int = 6):
        super().__init__(algebra)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.input_lift = nn.Linear(state_dim, hidden_dim * algebra.dim)
        self.input_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blocks = nn.ModuleList(
            [gbn_residual_block(algebra, hidden_dim) for _ in range(num_layers)]
        )
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blade_sel = BladeSelector(algebra, channels=hidden_dim)
        self.output_proj = CliffordLinear(algebra, hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        h = self.input_lift(state).reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)
        for block in self.blocks:
            h = apply_residual_block(block, h)
        h = self.output_norm(h)
        h = self.blade_sel(h)
        out = self.output_proj(h).squeeze(1)
        return extract_grade1(out, self.algebra, self.state_dim)

    @torch.no_grad()
    def hidden(self, state: torch.Tensor) -> torch.Tensor:
        """Post-block hidden multivector — for the grade spectrum diagnostic."""
        B = state.shape[0]
        h = self.input_lift(state).reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)
        for block in self.blocks:
            h = apply_residual_block(block, h)
        return h


# ==============================================================================
# Rollout & post-training diagnostics
# ==============================================================================

@torch.no_grad()
def rollout(model: HamiltonianRotorNet, x0_norm: torch.Tensor,
            n_steps: int) -> torch.Tensor:
    """Autoregressive rollout from a single normalized initial condition."""
    model.eval()
    steps = [x0_norm.unsqueeze(0)]
    state = x0_norm.unsqueeze(0)
    for _ in range(n_steps - 1):
        state = model(state)
        steps.append(state)
    return torch.cat(steps, dim=0)


@torch.no_grad()
def test_mse(model, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for states, nexts in loader:
        states, nexts = states.to(device), nexts.to(device)
        pred = model(states)
        total += F.mse_loss(pred, nexts).item() * states.shape[0]
        n += states.shape[0]
    return total / max(n, 1)


def post_training_diagnostics(
    model: HamiltonianRotorNet, ode, dataset: TrajectoryDataset,
    test_loader: DataLoader, device: str, *, rollout_steps: int, is_pendulum: bool,
) -> Dict[str, float]:
    """Gather every ex-loss-term and spectral claim as a measurement."""
    diagnostics: Dict[str, float] = {
        'test_mse': test_mse(model, test_loader, device),
    }

    rng = np.random.RandomState(999)
    x0_phys = ode.random_ic(rng, 'chaotic')
    x0_norm = dataset.normalize(x0_phys.astype(np.float32))
    x0_t = torch.tensor(x0_norm, dtype=torch.float32, device=device)
    traj_norm = rollout(model, x0_t, rollout_steps).cpu().numpy()
    traj_phys = dataset.denormalize(traj_norm)

    gt_phys = ode.generate_trajectory(x0_phys, rollout_steps, dataset.dt)
    final_rmse = float(np.linalg.norm(traj_phys[-1] - gt_phys[-1]))
    diagnostics['rollout_rmse_phys'] = final_rmse

    if is_pendulum:
        H = ode.energy_batch(traj_phys)
        drift = np.abs(H - H[0])
        diagnostics['energy_drift_mean'] = float(drift.mean())
        diagnostics['energy_drift_final'] = float(drift[-1])

        # Butterfly exponent: fit log‖x1 − x2‖ ~ λ t over the rollout
        x0_pert = x0_phys.copy()
        x0_pert[0] += 1e-3
        traj2 = rollout(
            model,
            torch.tensor(dataset.normalize(x0_pert.astype(np.float32)),
                         dtype=torch.float32, device=device),
            rollout_steps,
        ).cpu().numpy()
        traj2_phys = dataset.denormalize(traj2)
        sep = np.linalg.norm(traj2_phys - traj_phys, axis=-1).clip(min=1e-12)
        t_axis = np.arange(rollout_steps) * dataset.dt
        lam, _ = np.polyfit(t_axis, np.log(sep), 1)
        diagnostics['butterfly_lyapunov'] = float(lam)

    # Grade spectrum of hidden multivectors (hypothesis: even-subalgebra dominant)
    hiddens = []
    for states, _ in test_loader:
        hiddens.append(model.hidden(states.to(device)))
        break  # one batch is enough for the mean spectrum
    spectrum = mean_grade_spectrum(hiddens, model.algebra)
    total = spectrum.sum() + 1e-12
    for k, val in enumerate(spectrum):
        diagnostics[f'grade_spectrum_{k}'] = float(val)
    even = sum(spectrum[k] for k in range(len(spectrum)) if k % 2 == 0)
    diagnostics['even_subalgebra_ratio'] = float(even / total)
    return diagnostics


# ==============================================================================
# Training entry point
# ==============================================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device
    is_pendulum = (args.system == 'double_pendulum')

    if is_pendulum:
        ode = DoublePendulumODE()
        algebra = setup_algebra(p=2, q=2, device=device)
        state_dim = 4
    else:
        ode = LorenzODE()
        algebra = setup_algebra(p=args.p, q=args.q, device=device)
        state_dim = 3

    dt = 0.01
    train_ds = TrajectoryDataset(
        ode, n_traj=args.n_train, traj_len=200, dt=dt,
        regime=args.chaos, seed=args.seed)
    test_ds = TrajectoryDataset(
        ode, n_traj=max(args.n_train // 5, 50), traj_len=200, dt=dt,
        regime=args.chaos, seed=args.seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = HamiltonianRotorNet(
        algebra, state_dim=state_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
    ).to(device)

    print_banner(
        'Pendulum / Lorenz Incubator — Hamiltonian phase-space flow',
        system=args.system,
        signature=f'Cl({algebra.p},{algebra.q})  dim={algebra.dim}',
        regime=args.chaos,
        natural_loss='MSE on grade-1 readout',
        parameters=f'{count_parameters(model):,}',
        train=f'{len(train_ds):,}  test={len(test_ds):,}',
    )

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    def loss_fn(_model, batch):
        states, nexts = (b.to(device) for b in batch)
        return F.mse_loss(_model(states), nexts)

    def diag_fn(_model, _epoch) -> Dict[str, float]:
        return {'test_mse': test_mse(_model, test_loader, device)}

    history = run_supervised_loop(
        model, optimizer, loss_fn, train_loader,
        epochs=args.epochs, diag_interval=args.diag_interval, grad_clip=1.0,
        diag_fn=diag_fn, history_extra_keys=('test_mse',),
    )

    diagnostics = post_training_diagnostics(
        model, ode, train_ds, test_loader, device,
        rollout_steps=args.rollout_steps, is_pendulum=is_pendulum,
    )
    print(report_diagnostics(
        diagnostics, title='Pendulum post-training diagnostics',
    ))

    if args.save_plots:
        ensure_output_dir(args.output_dir)
        path = save_training_curve(
            history, os.path.join(args.output_dir, 'training_curve.png'),
            title=f'Pendulum — {args.system} MSE',
        )
        print(f'  curve saved to {path}')


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Hamiltonian phase-space flow in Cl(p,q) — pendulum / Lorenz.')
    add_standard_args(
        p,
        include=('seed', 'device', 'epochs', 'lr', 'batch_size',
                 'output_dir', 'save_plots', 'diag_interval', 'p', 'q'),
        defaults={'epochs': 200, 'batch_size': 256, 'diag_interval': 20,
                  'output_dir': 'pendulum_plots', 'p': 3, 'q': 0},
    )
    p.add_argument('--system', choices=['double_pendulum', 'lorenz'],
                   default='double_pendulum')
    p.add_argument('--chaos', choices=['regular', 'chaotic', 'mixed'],
                   default='mixed')
    p.add_argument('--n-train', type=int, default=500)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--rollout-steps', type=int, default=100)
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == '__main__':
    main()
