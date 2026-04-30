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

Hamiltonian Phase-Space Flow in Cl(2,2) or Cl(3,0).

Hypothesis
  The symplectic structure of a Hamiltonian system, with position-like and
  momentum-like variables carried by opposite signatures, should be a natural
  inductive bias for a GBN in ``Cl(p, q)``. A residual rotor stack predicts
  one-step phase-space flow from the current state using a single MSE on the
  grade-1 readout, while energy drift, even or odd grade ratio, and chaotic
  divergence remain post-training measurements.

Execute Command
  uv run python -m experiments.inc_pendulum_dynamics --epochs 200
  uv run python -m experiments.inc_pendulum_dynamics --system lorenz --p 3 --q 0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm
from core.module import CliffordModule
from experiments._lib import (
    apply_residual_block,
    build_visualization_metadata,
    count_parameters,
    ensure_output_dir,
    extract_grade1,
    gbn_residual_block,
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
from layers import BladeSelector, CliffordLayerNorm, CliffordLinear
from optimizers.riemannian import RiemannianAdam

# ==============================================================================
# Physics Simulation
# ==============================================================================


class DoublePendulumODE:
    """Double pendulum with Lagrangian mechanics. State = (θ1, θ2, ω1, ω2)."""

    state_dim = 4

    def __init__(self, l1: float = 1.0, l2: float = 1.0, m1: float = 1.0, m2: float = 1.0, g: float = 9.81):
        self.l1, self.l2 = l1, l2
        self.m1, self.m2 = m1, m2
        self.g = g

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """Vectorized derivatives. Accepts [4] (single) or [N, 4] (batch)."""
        t1, t2, w1, w2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
        d = t1 - t2
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        s, c = np.sin(d), np.cos(d)
        M11 = l1 * (m1 + m2)
        M12 = m2 * l2 * c
        M21 = m2 * l1 * c
        M22 = m2 * l2
        f1 = -m2 * l2 * w2**2 * s - (m1 + m2) * g * np.sin(t1)
        f2 = m2 * l1 * w1**2 * s - m2 * g * np.sin(t2)
        det = M11 * M22 - M12 * M21
        dw1 = (M22 * f1 - M12 * f2) / det
        dw2 = (M11 * f2 - M21 * f1) / det
        return np.stack([w1, w2, dw1, dw2], axis=-1)

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def energy_batch(self, states: np.ndarray) -> np.ndarray:
        """Hamiltonian H = T + V for arbitrary-rank states [..., 4]."""
        t1 = states[..., 0]
        t2 = states[..., 1]
        w1 = states[..., 2]
        w2 = states[..., 3]
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        T = 0.5 * (m1 + m2) * l1**2 * w1**2 + 0.5 * m2 * l2**2 * w2**2 + m2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2)
        V = -(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2)
        return T + V

    def generate_trajectory(self, x0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        """Single-IC trajectory; preserved for compatibility."""
        return self.generate_trajectories(x0[None, :], n_steps, dt)[:, 0]

    def generate_trajectories(self, x0_batch: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        """Batched RK4: [n_traj, state_dim] → [n_steps, n_traj, state_dim].

        One numpy step advances all trajectories simultaneously, replacing the
        outer Python ``for traj`` loop in the dataset constructor.
        """
        traj = np.empty((n_steps, x0_batch.shape[0], x0_batch.shape[1]), dtype=np.float64)
        traj[0] = x0_batch
        state = x0_batch.copy()
        for i in range(1, n_steps):
            state = self.rk4_step(state, dt)
            traj[i] = state
        return traj

    def random_ic(self, rng: np.random.RandomState, regime: str = "mixed") -> np.ndarray:
        if regime == "mixed":
            regime = rng.choice(["regular", "chaotic"])
        if regime == "regular":
            lo, hi, wmax = -np.pi / 4, np.pi / 4, 1.0
        else:
            lo, hi, wmax = -np.pi, np.pi, 3.0
        return np.array(
            [rng.uniform(lo, hi), rng.uniform(lo, hi), rng.uniform(-wmax, wmax), rng.uniform(-wmax, wmax)],
            dtype=np.float64,
        )

    def project_to_energy_shell(self, state_pred: np.ndarray, state_input: np.ndarray) -> np.ndarray:
        """Rescale predicted (ω1, ω2) so H(state_pred) = H(state_input).

        Solves the quadratic ``A s² + B s + C = E_target`` in the kinetic-
        energy scaling ``s`` (angles untouched). Falls back to the unscaled
        prediction if the quadratic has no positive root (off-shell beyond
        what kinetic-only rescaling can fix).
        """
        E_target = self.energy_batch(state_input)
        t1 = state_pred[..., 0]
        t2 = state_pred[..., 1]
        w1 = state_pred[..., 2]
        w2 = state_pred[..., 3]
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
        V = -(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2)
        T_pred = 0.5 * (m1 + m2) * l1**2 * w1**2 + 0.5 * m2 * l2**2 * w2**2 + m2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2)
        T_target = E_target - V
        # s² = T_target / T_pred; fall back to s=1 when target negative or
        # T_pred ~ 0 (network briefly visited a degenerate region).
        ratio = np.where(T_pred > 1e-9, T_target / T_pred, 1.0)
        s = np.sqrt(np.clip(ratio, 0.25, 4.0))
        out = state_pred.copy()
        out[..., 2] = w1 * s
        out[..., 3] = w2 * s
        return out


class LorenzODE:
    """Lorenz (1963) dissipative attractor. State = (x, y, z)."""

    state_dim = 3

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0):
        self.sigma, self.rho, self.beta = sigma, rho, beta

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """Vectorized derivatives. Accepts [3] (single) or [N, 3] (batch)."""
        x, y, z = state[..., 0], state[..., 1], state[..., 2]
        return np.stack(
            [
                self.sigma * (y - x),
                x * (self.rho - z) - y,
                x * y - self.beta * z,
            ],
            axis=-1,
        )

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def generate_trajectory(self, x0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        return self.generate_trajectories(x0[None, :], n_steps, dt)[:, 0]

    def generate_trajectories(self, x0_batch: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
        traj = np.empty((n_steps, x0_batch.shape[0], x0_batch.shape[1]), dtype=np.float64)
        traj[0] = x0_batch
        state = x0_batch.copy()
        for i in range(1, n_steps):
            state = self.rk4_step(state, dt)
            traj[i] = state
        return traj

    def random_ic(self, rng: np.random.RandomState, regime: str = "mixed") -> np.ndarray:
        if regime == "mixed":
            regime = rng.choice(["regular", "chaotic"])
        center = math.sqrt(self.beta * (self.rho - 1))
        sign = rng.choice([-1, 1])
        ic = np.array([sign * center, sign * center, self.rho - 1], dtype=np.float64)
        scale = 1.0 if regime == "regular" else 5.0
        return ic + rng.randn(3) * scale


# ==============================================================================
# Dataset
# ==============================================================================


class TrajectoryDataset(Dataset):
    """One-step-ahead (state_t, state_{t+k}) pairs, z-score normalized."""

    def __init__(
        self,
        ode,
        n_traj: int = 500,
        traj_len: int = 200,
        dt: float = 0.01,
        n_steps_ahead: int = 1,
        regime: str = "mixed",
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        self.ode = ode
        self.dt = dt
        # Batched RK4 across all trajectories — replaces the per-traj loop.
        x0_batch = np.stack(
            [ode.random_ic(rng, regime) for _ in range(n_traj)],
            axis=0,
        ).astype(np.float64)
        full = ode.generate_trajectories(x0_batch, traj_len + n_steps_ahead, dt)
        states = full[:traj_len].reshape(-1, x0_batch.shape[1]).astype(np.float32)
        nexts = full[n_steps_ahead : n_steps_ahead + traj_len].reshape(-1, x0_batch.shape[1]).astype(np.float32)
        self.state_mean = torch.tensor(states.mean(axis=0), dtype=torch.float32)
        self.state_std = torch.tensor(states.std(axis=0).clip(min=1e-6), dtype=torch.float32)
        self.states = torch.from_numpy((states - self.state_mean.numpy()) / self.state_std.numpy())
        self.nexts = torch.from_numpy((nexts - self.state_mean.numpy()) / self.state_std.numpy())

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

    Implicit coercion (active when ``coerce_even=True``):

      * ``even_grade_mask`` — between blocks the hidden multivector is
        multiplicatively masked onto the even subalgebra (grades 0, 2, 4 in
        Cl(2,2)). The rotor sandwich is grade-parity preserving by
        construction; the per-block ``CliffordLinear`` is not, so this
        projection blocks the ``minute leakage`` of energy into odd grades
        without entering the gradient as an explicit penalty term.
      * Hermitian-norm renormalization — after each block we divide the
        hidden state by its (signature-aware) Hermitian norm. Caps numeric
        magnitude drift through deep stacks; geometric content untouched.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 6,
        coerce_even: bool = True,
        hermitian_renorm: bool = True,
    ):
        super().__init__(algebra)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.coerce_even = coerce_even
        self.hermitian_renorm = hermitian_renorm

        self.input_lift = nn.Linear(state_dim, hidden_dim * algebra.dim)
        self.input_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blocks = nn.ModuleList([gbn_residual_block(algebra, hidden_dim) for _ in range(num_layers)])
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blade_sel = BladeSelector(algebra, channels=hidden_dim)
        self.output_proj = CliffordLinear(algebra, hidden_dim, 1)

        even_mask = torch.zeros(algebra.dim)
        for k in range(0, algebra.n + 1, 2):
            even_mask = even_mask + algebra.grade_masks_float[k].cpu()
        self.register_buffer("even_grade_mask", even_mask)

    def _coerce(self, h: torch.Tensor) -> torch.Tensor:
        if self.coerce_even:
            mask = self.even_grade_mask.to(h.dtype)
            h = h * mask
        if self.hermitian_renorm:
            norm = hermitian_norm(self.algebra, h).clamp(min=1e-6)
            h = h / norm
        return h

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        h = self.input_lift(state).reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)
        for block in self.blocks:
            h = apply_residual_block(block, h)
            h = self._coerce(h)
        h = self.output_norm(h)
        h = self.blade_sel(h)
        out = self.output_proj(h).squeeze(1)
        return extract_grade1(out, self.algebra, self.state_dim)

    @torch.no_grad()
    def hidden(self, state: torch.Tensor, *, apply_coercion: bool = True) -> torch.Tensor:
        """Post-block hidden multivector — for the grade spectrum diagnostic.

        ``apply_coercion=False`` returns the raw post-block state so the
        odd-grade leak diagnostic can measure what the lift would have done
        if not blocked.
        """
        B = state.shape[0]
        h = self.input_lift(state).reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)
        for block in self.blocks:
            h = apply_residual_block(block, h)
            if apply_coercion:
                h = self._coerce(h)
        return h


# ==============================================================================
# Rollout & post-training diagnostics
# ==============================================================================


@torch.no_grad()
def rollout(
    model: HamiltonianRotorNet,
    x0_norm: torch.Tensor,
    n_steps: int,
    *,
    ode=None,
    dataset: "TrajectoryDataset" = None,
    energy_shell_project: bool = False,
) -> torch.Tensor:
    """Autoregressive rollout from a single normalized initial condition.

    When ``energy_shell_project`` is on (and ``ode`` exposes
    ``project_to_energy_shell``), each predicted step is rescaled in physical
    units so that ``H(x_pred) = H(x_input)`` — a literal renormalization onto
    the original Hamiltonian shell. Toggled off for systems without a closed-
    form energy (e.g., Lorenz).
    """
    model.eval()
    steps = [x0_norm.unsqueeze(0)]
    state = x0_norm.unsqueeze(0)
    can_project = (
        energy_shell_project and ode is not None and dataset is not None and hasattr(ode, "project_to_energy_shell")
    )
    for _ in range(n_steps - 1):
        next_state = model(state)
        if can_project:
            cur_phys = dataset.denormalize(state.cpu().numpy())
            nxt_phys = dataset.denormalize(next_state.cpu().numpy())
            nxt_phys = ode.project_to_energy_shell(nxt_phys, cur_phys)
            next_state = torch.as_tensor(
                dataset.normalize(nxt_phys),
                dtype=state.dtype,
                device=state.device,
            )
        steps.append(next_state)
        state = next_state
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


@torch.no_grad()
def _rotor_norm_residual(model: HamiltonianRotorNet) -> float:
    """Mean ``|hermitian_norm(R) - 1|`` across every RotorLayer in the model.

    Each ``RotorLayer`` learns a bivector ``B``; its rotor is ``R = exp(-B/2)``.
    Up to numerical precision the closed-form exp produces a unit-norm rotor,
    so this number should sit near machine epsilon. A spike means the bivector
    parameter has slipped outside the regime where the closed form is exact.
    """
    from layers.primitives.rotor import RotorLayer

    deviations = []
    for module in model.modules():
        if isinstance(module, RotorLayer) and module.grade == 2:
            V = module._build_grade_element(model.even_grade_mask.device, model.even_grade_mask.dtype)
            R = model.algebra.exp(-0.5 * V)
            n = hermitian_norm(model.algebra, R).squeeze(-1)
            deviations.append((n - 1.0).abs().mean().item())
    return float(sum(deviations) / max(len(deviations), 1))


@torch.no_grad()
def _gauge_covariance_residual(model: HamiltonianRotorNet, loader: DataLoader, device: str) -> float:
    """|‖model(x)‖² − ‖model(R · x · R̃)‖²| under a random phase-space rotor.

    Pendulum-like analogue of ``gauge_covariance_residual`` from
    :mod:`experiments.dbg_yang_mills` — a small bivector perturbation is
    applied to the (lifted) input multivector and we verify that the predicted
    grade-1 readout has the same Hermitian norm. Sensitive to architectural
    asymmetry in how the network treats geometrically equivalent inputs.
    """
    algebra = model.algebra
    bv = torch.zeros(1, algebra.dim, device=device)
    g2_mask = algebra.grade_masks_float[2].to(device=device, dtype=bv.dtype)
    bv = bv + torch.randn(1, algebra.dim, device=device) * 0.1 * g2_mask
    R = algebra.exp(-0.5 * bv)
    R_rev = algebra.reverse(R)

    states, _ = next(iter(loader))
    states = states.to(device)
    pred = model(states)

    # Apply the rotor in grade-1 input space: lift -> sandwich -> extract -> model.
    state_mv = algebra.embed_vector(states)
    rotated_mv = algebra.geometric_product(
        algebra.geometric_product(R.expand_as(state_mv), state_mv),
        R_rev.expand_as(state_mv),
    )
    rotated_state = extract_grade1(rotated_mv, algebra, model.state_dim)
    pred_rot = model(rotated_state)

    norm_orig = (pred * pred).sum(dim=-1)
    norm_rot = (pred_rot * pred_rot).sum(dim=-1)
    return float((norm_orig - norm_rot).abs().mean().item())


def post_training_diagnostics(
    model: HamiltonianRotorNet,
    ode,
    dataset: TrajectoryDataset,
    test_loader: DataLoader,
    device: str,
    *,
    rollout_steps: int,
    is_pendulum: bool,
    energy_shell_project: bool = False,
) -> Dict[str, float]:
    """Gather every ex-loss-term and spectral claim as a measurement."""
    diagnostics: Dict[str, float] = {
        "test_mse": test_mse(model, test_loader, device),
    }

    rng = np.random.RandomState(999)
    x0_phys = ode.random_ic(rng, "chaotic")
    x0_norm = dataset.normalize(x0_phys.astype(np.float32))
    x0_t = torch.tensor(x0_norm, dtype=torch.float32, device=device)
    traj_norm = (
        rollout(
            model,
            x0_t,
            rollout_steps,
            ode=ode,
            dataset=dataset,
            energy_shell_project=energy_shell_project and is_pendulum,
        )
        .cpu()
        .numpy()
    )
    traj_phys = dataset.denormalize(traj_norm)

    gt_phys = ode.generate_trajectory(x0_phys, rollout_steps, dataset.dt)
    final_rmse = float(np.linalg.norm(traj_phys[-1] - gt_phys[-1]))
    diagnostics["rollout_rmse_phys"] = final_rmse

    if is_pendulum:
        H = ode.energy_batch(traj_phys)
        drift = np.abs(H - H[0])
        diagnostics["energy_drift_mean"] = float(drift.mean())
        diagnostics["energy_drift_final"] = float(drift[-1])

        # Butterfly exponent: fit log‖x1 − x2‖ ~ λ t over the rollout
        x0_pert = x0_phys.copy()
        x0_pert[0] += 1e-3
        traj2 = (
            rollout(
                model,
                torch.tensor(dataset.normalize(x0_pert.astype(np.float32)), dtype=torch.float32, device=device),
                rollout_steps,
                ode=ode,
                dataset=dataset,
                energy_shell_project=energy_shell_project,
            )
            .cpu()
            .numpy()
        )
        traj2_phys = dataset.denormalize(traj2)
        sep = np.linalg.norm(traj2_phys - traj_phys, axis=-1).clip(min=1e-12)
        t_axis = np.arange(rollout_steps) * dataset.dt
        lam, _ = np.polyfit(t_axis, np.log(sep), 1)
        diagnostics["butterfly_lyapunov"] = float(lam)

    # Grade spectrum on the *coerced* hidden state — what training sees.
    hiddens = []
    for states, _ in test_loader:
        hiddens.append(model.hidden(states.to(device), apply_coercion=True))
        break
    spectrum = mean_grade_spectrum(hiddens, model.algebra)
    total = spectrum.sum() + 1e-12
    for k, val in enumerate(spectrum):
        diagnostics[f"grade_spectrum_{k}"] = float(val)
    even = sum(spectrum[k] for k in range(len(spectrum)) if k % 2 == 0)
    diagnostics["even_subalgebra_ratio"] = float(even / total)

    # Pre-coercion leak — what the lift+linear stack would have produced
    # without the even-grade mask. Drops to ~zero when coerce_even is off
    # (because then there's nothing to subtract); meaningful only when on.
    raw_hiddens = []
    for states, _ in test_loader:
        raw_hiddens.append(model.hidden(states.to(device), apply_coercion=False))
        break
    raw_spec = mean_grade_spectrum(raw_hiddens, model.algebra)
    raw_total = raw_spec.sum() + 1e-12
    raw_even = sum(raw_spec[k] for k in range(len(raw_spec)) if k % 2 == 0)
    diagnostics["odd_grade_leak"] = float(1.0 - raw_even / raw_total)

    diagnostics["rotor_norm_residual"] = _rotor_norm_residual(model)
    diagnostics["gauge_covariance_residual"] = _gauge_covariance_residual(model, test_loader, device)
    return diagnostics


# ==============================================================================
# Training entry point
# ==============================================================================


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device
    is_pendulum = args.system == "double_pendulum"

    if is_pendulum:
        ode = DoublePendulumODE()
        algebra = setup_algebra(p=2, q=2, device=device)
        state_dim = 4
    else:
        ode = LorenzODE()
        algebra = setup_algebra(p=args.p, q=args.q, device=device)
        state_dim = 3

    dt = 0.01
    train_ds = TrajectoryDataset(ode, n_traj=args.n_train, traj_len=200, dt=dt, regime=args.chaos, seed=args.seed)
    test_ds = TrajectoryDataset(
        ode, n_traj=max(args.n_train // 5, 50), traj_len=200, dt=dt, regime=args.chaos, seed=args.seed + 1
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    coerce_even = args.coerce_even and is_pendulum
    energy_shell = args.energy_shell_project and is_pendulum
    model = HamiltonianRotorNet(
        algebra,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        coerce_even=coerce_even,
        hermitian_renorm=args.hermitian_renorm,
    ).to(device)

    print_banner(
        "Pendulum / Lorenz Incubator — Hamiltonian phase-space flow",
        system=args.system,
        signature=f"Cl({algebra.p},{algebra.q})  dim={algebra.dim}",
        regime=args.chaos,
        natural_loss="MSE on grade-1 readout",
        coerce_even=coerce_even,
        hermitian_renorm=args.hermitian_renorm,
        energy_shell_project=energy_shell,
        parameters=f"{count_parameters(model):,}",
        train=f"{len(train_ds):,}  test={len(test_ds):,}",
    )

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    def loss_fn(_model, batch):
        states, nexts = (b.to(device) for b in batch)
        return F.mse_loss(_model(states), nexts)

    def diag_fn(_model, _epoch) -> Dict[str, float]:
        return {"test_mse": test_mse(_model, test_loader, device)}

    history = run_supervised_loop(
        model,
        optimizer,
        loss_fn,
        train_loader,
        epochs=args.epochs,
        diag_interval=args.diag_interval,
        grad_clip=1.0,
        diag_fn=diag_fn,
        history_extra_keys=("test_mse",),
    )

    diagnostics = post_training_diagnostics(
        model,
        ode,
        train_ds,
        test_loader,
        device,
        rollout_steps=args.rollout_steps,
        is_pendulum=is_pendulum,
        energy_shell_project=energy_shell,
    )
    print(
        report_diagnostics(
            diagnostics,
            title="Pendulum post-training diagnostics",
        )
    )

    ensure_output_dir(args.output_dir)
    q = 2 if is_pendulum else args.q
    metadata = build_visualization_metadata(
        signature_metadata(args.p if not is_pendulum else 2, q),
        system=args.system,
        chaos=args.chaos,
        seed=args.seed,
    )
    path = save_training_curve(
        history,
        output_dir=args.output_dir,
        experiment_name="inc_pendulum_dynamics",
        metadata=metadata,
        plot_name="training_curve",
        args=args,
        module=__name__,
        title=f"Pendulum — {args.system} MSE",
    )
    print(f"  curve saved to {path}")


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    p = make_experiment_parser(
        "Hamiltonian phase-space flow in Cl(p,q) — pendulum / Lorenz.",
        include=("seed", "device", "epochs", "lr", "batch_size", "output_dir", "diag_interval", "p", "q"),
        defaults={
            "epochs": 200,
            "batch_size": 256,
            "diag_interval": 20,
            "output_dir": "pendulum_plots",
            "p": 3,
            "q": 0,
        },
    )
    p.add_argument("--system", choices=["double_pendulum", "lorenz"], default="double_pendulum")
    p.add_argument("--chaos", choices=["regular", "chaotic", "mixed"], default="mixed")
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--rollout-steps", type=int, default=100)
    p.add_argument(
        "--coerce-even",
        dest="coerce_even",
        action="store_true",
        default=True,
        help="(default for double_pendulum) Mask the hidden state onto the even subalgebra between blocks.",
    )
    p.add_argument("--no-coerce-even", dest="coerce_even", action="store_false")
    p.add_argument(
        "--hermitian-renorm",
        dest="hermitian_renorm",
        action="store_true",
        default=True,
        help="Renormalize hidden multivectors to unit Hermitian norm after each block.",
    )
    p.add_argument("--no-hermitian-renorm", dest="hermitian_renorm", action="store_false")
    p.add_argument(
        "--energy-shell-project",
        dest="energy_shell_project",
        action="store_true",
        default=True,
        help="(pendulum only) Renormalize predicted (ω1, ω2) at rollout so H(state_pred) = H(state_input).",
    )
    p.add_argument("--no-energy-shell-project", dest="energy_shell_project", action="store_false")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
