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

This script is designed to validate topological and algebraic phenomena rather
than to achieve State-of-the-Art (SOTA) on traditional benchmarks. Its focus
is to confirm that the Clifford Algebra framework computes known identities
and physical laws correctly, and to surface regressions when they do not.

Please kindly note that as an experimental module, formal mathematical proofs
and exhaustive literature reviews may still be in progress. Contributions that
tighten the validation suite — additional check_* methods, sharper tolerances,
cross-references to the literature — are warmly welcomed.

==============================================================================

Lorentz Rotor Recovery in Cl(3,1).

Hypothesis
  Versor's geometric bias should recover Lorentz boosts in Cl(3,1) from
  ``(original, boosted)`` event pairs through backpropagation without
  breaking. A single natural loss, Hermitian distance between predicted and
  true rotor with sign ambiguity handled by ``min``, drives learning;
  interval preservation, grade confinement, and other physics invariants are
  measured after training rather than enforced as gradient terms.

Execute Command
  uv run python -m experiments.dbg_lorentz
  uv run python -m experiments.dbg_lorentz --epochs 20
  uv run python -m experiments.dbg_lorentz --boost-type pure_rotation
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments._lib import (
    build_visualization_metadata, ensure_output_dir, make_experiment_parser,
    print_banner, report_diagnostics, run_supervised_loop, save_training_curve,
    set_seed, setup_algebra, signature_metadata,
)
from core.metric import (
    hermitian_distance, hermitian_grade_spectrum,
    signature_norm_squared, signature_trace_form,
)
from layers.primitives.base import CliffordModule
from layers import BladeSelector, CliffordLayerNorm, CliffordLinear, RotorLayer
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam


# ---------------------------------------------------------------------------
# Boost / rotation bivector helpers
# ---------------------------------------------------------------------------

def _boost_bivector(algebra, axis: int) -> torch.Tensor:
    bv = torch.zeros(algebra.dim, device=algebra.device)
    bv[(1 << axis) | (1 << 3)] = 1.0
    return bv


def _rotation_bivector(algebra, plane: int) -> torch.Tensor:
    bv = torch.zeros(algebra.dim, device=algebra.device)
    a, b = [(0, 1), (0, 2), (1, 2)][plane]
    bv[(1 << a) | (1 << b)] = 1.0
    return bv


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LorentzDataset(Dataset):
    """(original, boosted) event pairs in Cl(3,1) with the analytic rotor."""

    def __init__(self, algebra, num_samples: int, *,
                 boost_type: str = 'pure_boost', rapidity_max: float = 1.5,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        dim = algebra.dim
        events = torch.zeros(num_samples, 2, dim)
        rotors = torch.zeros(num_samples, dim)
        rapidities = torch.zeros(num_samples)
        for i in range(num_samples):
            spatial = rng.uniform(-2.0, 2.0, 3).astype(np.float32)
            t = np.float32(np.sqrt((spatial ** 2).sum()) + rng.uniform(0.5, 3.0))
            coords = np.concatenate([spatial, [t]])
            event = algebra.embed_vector(
                torch.tensor(coords, dtype=torch.float32, device=algebra.device),
            )
            phi = float(rng.uniform(-rapidity_max, rapidity_max))
            rapidities[i] = phi
            if boost_type == 'pure_boost':
                bv = _boost_bivector(algebra, rng.randint(0, 3))
            elif boost_type == 'pure_rotation':
                bv = _rotation_bivector(algebra, rng.randint(0, 3))
            else:
                bv_b = _boost_bivector(algebra, rng.randint(0, 3))
                bv_r = _rotation_bivector(algebra, rng.randint(0, 3))
                alpha = np.float32(rng.uniform(0.3, 0.7))
                bv = alpha * bv_b + (1 - alpha) * bv_r
                n = bv.norm()
                if n > 1e-6:
                    bv = bv / n
            rotor = algebra.exp(((-phi / 2.0) * bv).unsqueeze(0)).squeeze(0)
            rotor_rev = algebra.reverse(rotor.unsqueeze(0)).squeeze(0)
            temp = algebra.geometric_product(rotor.unsqueeze(0), event.unsqueeze(0)).squeeze(0)
            boosted = algebra.geometric_product(
                temp.unsqueeze(0), rotor_rev.unsqueeze(0),
            ).squeeze(0)
            events[i, 0] = event.cpu()
            events[i, 1] = boosted.cpu()
            rotors[i] = rotor.cpu()
        self.events, self.rotors, self.rapidities = events, rotors, rapidities

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        return self.events[idx], self.rotors[idx], self.rapidities[idx]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class LorentzNet(CliffordModule):
    """GBN that predicts the rotor from an (original, boosted) event pair."""

    def __init__(self, algebra, hidden_dim: int = 64,
                 num_layers: int = 6, num_freqs: int = 32):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim
        self.coord_norm = nn.LayerNorm(8)
        self.register_buffer('freq_bands', torch.randn(8, num_freqs) * 2.0)
        input_dim = 8 + 2 * num_freqs
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

    def forward(self, event_pairs: torch.Tensor) -> torch.Tensor:
        B = event_pairs.shape[0]
        grade1_idx = [1 << i for i in range(self.algebra.n)]
        coords = [event_pairs[:, ev, gi] for ev in range(2) for gi in grade1_idx]
        raw = self.coord_norm(torch.stack(coords, dim=-1))
        proj = raw @ self.freq_bands
        features = torch.cat([raw, torch.sin(proj), torch.cos(proj)], dim=-1)
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
# Physics diagnostics (post-training, no gradients)
# ---------------------------------------------------------------------------

class LorentzDebugger:
    """Closed-form physics checks on ground-truth and predicted rotors."""

    def __init__(self, algebra):
        self.algebra = algebra

    def interval_invariance(self, events: torch.Tensor) -> float:
        s2_o = signature_norm_squared(self.algebra, events[:, 0])
        s2_b = signature_norm_squared(self.algebra, events[:, 1])
        return (s2_o - s2_b).abs().mean().item()

    def rotor_normalization(self, rotors: torch.Tensor) -> float:
        return (signature_trace_form(self.algebra, rotors, rotors) - 1.0).abs().mean().item()

    def grade_confinement(self, rotors: torch.Tensor) -> Tuple[float, float]:
        spec = hermitian_grade_spectrum(self.algebra, rotors)
        even = (spec[:, 0] + spec[:, 2] + spec[:, 4]).mean().item()
        odd = (spec[:, 1] + spec[:, 3]).mean().item()
        total = even + odd + 1e-12
        return even / total, odd

    def causality_preservation(self, events, rotors_pred, tol: float = 0.05) -> float:
        s2_o = signature_norm_squared(self.algebra, events[:, 0]).squeeze(-1)
        R_rev = self.algebra.reverse(rotors_pred)
        transformed = self.algebra.geometric_product(
            self.algebra.geometric_product(rotors_pred, events[:, 0]), R_rev,
        )
        s2_t = signature_norm_squared(self.algebra, transformed).squeeze(-1)
        tl_o, sl_o, ll_o = (s2_o < -tol), (s2_o > tol), (s2_o.abs() < tol)
        tl_t, sl_t, ll_t = (s2_t < -tol), (s2_t > tol), (s2_t.abs() < tol)
        return ((tl_o & tl_t) | (sl_o & sl_t) | (ll_o & ll_t)).float().mean().item()

    def invariant_mass_error(self, events, rotors_pred) -> float:
        s2_o = signature_norm_squared(self.algebra, events[:, 0]).squeeze(-1)
        m2_o = -s2_o
        R_rev = self.algebra.reverse(rotors_pred)
        transformed = self.algebra.geometric_product(
            self.algebra.geometric_product(rotors_pred, events[:, 0]), R_rev,
        )
        m2_t = -signature_norm_squared(self.algebra, transformed).squeeze(-1)
        timelike = m2_o > 0.05
        if timelike.sum() == 0:
            return float('nan')
        return (m2_o[timelike] - m2_t[timelike]).abs().mean().item()

    def rapidity_mae(self, rotors_pred, rapidities_true) -> Tuple[float, float, int]:
        scalar = rotors_pred[:, 0]
        bv_mask = torch.tensor(
            [bin(i).count('1') == 2 for i in range(self.algebra.dim)],
            dtype=torch.bool, device=rotors_pred.device,
        )
        bv_norm = rotors_pred[:, bv_mask].norm(dim=-1)
        raw_ratio = bv_norm / (scalar.abs() + 1e-8)
        n_clamped = int((raw_ratio >= 0.999).sum().item())
        phi_pred = 2.0 * torch.atanh(raw_ratio.clamp(0.0, 0.999))
        phi_true = rapidities_true.abs()
        mae = (phi_pred - phi_true).abs().mean().item()
        corr = 0.0
        if phi_pred.std() > 1e-6 and phi_true.std() > 1e-6:
            corr = torch.corrcoef(torch.stack([phi_pred, phi_true]))[0, 1].item()
        return mae, corr, n_clamped

    def velocity_addition_error(self, n_tests: int = 100,
                                rapidity_max: float = 1.5,
                                seed: int = 123) -> float:
        rng = np.random.RandomState(seed)
        errors = []
        for _ in range(n_tests):
            phi1 = rng.uniform(-rapidity_max, rapidity_max)
            phi2 = rng.uniform(-rapidity_max, rapidity_max)
            bv = _boost_bivector(self.algebra, rng.randint(0, 3))
            R1 = self.algebra.exp(((-phi1 / 2.0) * bv).unsqueeze(0)).squeeze(0)
            R2 = self.algebra.exp(((-phi2 / 2.0) * bv).unsqueeze(0)).squeeze(0)
            R12 = self.algebra.exp(((-(phi1 + phi2) / 2.0) * bv).unsqueeze(0)).squeeze(0)
            composed = self.algebra.geometric_product(
                R1.unsqueeze(0), R2.unsqueeze(0),
            ).squeeze(0)
            errors.append(min((composed - R12).norm().item(),
                              (composed + R12).norm().item()))
        return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Training (single natural loss)
# ---------------------------------------------------------------------------

def _rotor_hermitian_loss(algebra, pred_rotors, true_rotors) -> torch.Tensor:
    """Hermitian distance with sign ambiguity; the natural loss for a rotor."""
    d_pos = hermitian_distance(algebra, pred_rotors, true_rotors).mean()
    d_neg = hermitian_distance(algebra, pred_rotors, -true_rotors).mean()
    return torch.min(d_pos, d_neg)


@torch.no_grad()
def _evaluate(model, loader, algebra, device) -> Dict[str, torch.Tensor]:
    model.eval()
    all_events, all_true, all_pred, all_raps = [], [], [], []
    total, n = 0.0, 0
    for events, true_rotors, raps in loader:
        events = events.to(device); true_rotors = true_rotors.to(device)
        pred = model(events)
        total += _rotor_hermitian_loss(algebra, pred, true_rotors).item() * events.shape[0]
        n += events.shape[0]
        all_events.append(events.cpu()); all_true.append(true_rotors.cpu())
        all_pred.append(pred.cpu()); all_raps.append(raps)
    return {
        'test_loss': torch.tensor(total / max(n, 1)),
        'events': torch.cat(all_events),
        'rotors_true': torch.cat(all_true),
        'rotors_pred': torch.cat(all_pred),
        'rapidities': torch.cat(all_raps),
    }


def train(args) -> None:
    set_seed(args.seed)
    device = args.device
    algebra = setup_algebra(p=3, q=1, device=device)
    debugger = LorentzDebugger(algebra)

    print_banner(
        'Lorentz Debugger — Cl(3,1)',
        signature='(+,+,+,-)',
        boost_type=args.boost_type,
        natural_loss='Hermitian rotor distance (sign-ambiguous)',
    )

    # Pure-algebra sanity (not a loss term)
    vel_err = debugger.velocity_addition_error(
        n_tests=100, rapidity_max=args.rapidity_max,
    )
    print(f'  Rapidity additivity error (algebra kernel): {vel_err:.3e}')
    if vel_err > 1e-4:
        warnings.warn(
            f'Rapidity additivity error {vel_err:.2e} is large; '
            'inspect algebra.exp() before trusting training results.',
            RuntimeWarning, stacklevel=2,
        )

    train_ds = LorentzDataset(algebra, args.num_train,
                              boost_type=args.boost_type,
                              rapidity_max=args.rapidity_max,
                              seed=args.seed)
    test_ds = LorentzDataset(algebra, args.num_test,
                             boost_type=args.boost_type,
                             rapidity_max=args.rapidity_max,
                             seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LorentzNet(algebra, hidden_dim=args.hidden_dim,
                       num_layers=args.num_layers, num_freqs=args.num_freqs).to(device)
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')
    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    def loss_fn(_model, batch):
        events, true_rotors, _ = batch
        events = events.to(device); true_rotors = true_rotors.to(device)
        return _rotor_hermitian_loss(algebra, _model(events), true_rotors)

    def diag_fn(_model, _epoch) -> Dict[str, float]:
        ev = _evaluate(_model, test_loader, algebra, device)
        return {'test_loss': float(ev['test_loss'])}

    history = run_supervised_loop(
        model, optimizer, loss_fn, train_loader,
        epochs=args.epochs, diag_interval=args.diag_interval,
        grad_clip=1.0,
    )

    # Post-training report: every ex-loss-term is now a measurement.
    ev = _evaluate(model, test_loader, algebra, device)
    even_ratio, odd_energy = debugger.grade_confinement(ev['rotors_pred'])
    rap_mae, rap_corr, rap_clamped = debugger.rapidity_mae(
        ev['rotors_pred'], ev['rapidities'],
    )
    diagnostics = {
        'test_loss':              float(ev['test_loss']),
        'interval_err_true':      debugger.interval_invariance(ev['events']),
        'rotor_norm_err_true':    debugger.rotor_normalization(ev['rotors_true']),
        'rotor_norm_err_pred':    debugger.rotor_normalization(ev['rotors_pred']),
        'even_ratio_pred':        even_ratio,
        'odd_energy_pred':        odd_energy,
        'causality_preservation': debugger.causality_preservation(
            ev['events'], ev['rotors_pred'],
        ),
        'invariant_mass_err':     debugger.invariant_mass_error(
            ev['events'], ev['rotors_pred'],
        ),
        'rapidity_mae':           rap_mae,
        'rapidity_corr':          rap_corr,
        'rapidity_clamped_frac':  rap_clamped / max(len(ev['rotors_pred']), 1),
        'velocity_addition_err':  vel_err,
    }
    print(report_diagnostics(
        diagnostics, title='Lorentz post-training physics diagnostics',
    ))

    ensure_output_dir(args.output_dir)
    metadata = build_visualization_metadata(
        signature_metadata(3, 1),
        boost_type=args.boost_type,
        seed=args.seed,
    )
    path = save_training_curve(
        history,
        output_dir=args.output_dir,
        experiment_name='dbg_lorentz',
        metadata=metadata,
        plot_name='training_curve',
        args=args,
        module=__name__,
        title='Lorentz — rotor Hermitian loss',
    )
    print(f'  curve saved to {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = make_experiment_parser(
        'Lorentz boost debugger in Cl(3,1)',
        include=('seed', 'device', 'epochs', 'lr', 'batch_size',
                 'output_dir', 'diag_interval'),
        defaults={'epochs': 200, 'lr': 0.001, 'batch_size': 128,
                  'output_dir': 'lorentz_plots', 'diag_interval': 20},
    )
    p.add_argument('--boost-type', choices=['pure_boost', 'pure_rotation', 'combined'],
                   default='pure_boost')
    p.add_argument('--rapidity-max', type=float, default=1.5)
    p.add_argument('--num-train', type=int, default=3000)
    p.add_argument('--num-test', type=int, default=500)
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--num-freqs', type=int, default=32)
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == '__main__':
    main()
