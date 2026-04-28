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

Incubator Template — 2-D Rotation Regression in Cl(2,0).

Hypothesis
  A tiny GBN ``(CliffordLayerNorm -> RotorLayer -> GeometricGELU ->
  CliffordLinear)`` should learn a fixed 2-D rotation from grade-1 point
  pairs ``(x, R_target x)`` in under a thousand gradient steps. This is the
  smallest non-trivial demonstration of the rotor sandwich
  ``x' = R x R~`` as an inductive bias.

Execute Command
  uv run python -m experiments._templates.inc_template
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Bootstrap project root so the file runs both via ``-m`` and as a bare script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from experiments._lib import (
    count_parameters,
    ensure_output_dir,
    make_experiment_parser,
    print_banner,
    save_training_curve,
    set_seed,
    setup_algebra,
)
from layers import BladeSelector, CliffordLayerNorm, CliffordLinear, RotorLayer
from layers.primitives.base import CliffordModule
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam


# ---------------------------------------------------------------------------
# Data — random 2-D points; target is the same point rotated by a fixed angle.
# ---------------------------------------------------------------------------


def _embed_as_vector(points_2d: torch.Tensor, dim: int) -> torch.Tensor:
    """Embed [..., 2] into a grade-1 multivector [..., dim] for Cl(2,0).

    Basis order for Cl(2,0) is (1, e1, e2, e12) → grade-1 positions are 1 and 2.
    """
    batch = points_2d.shape[0]
    mv = torch.zeros(batch, 1, dim, dtype=points_2d.dtype)
    mv[..., 0, 1] = points_2d[..., 0]
    mv[..., 0, 2] = points_2d[..., 1]
    return mv


def make_rotation_dataset(n: int, angle_rad: float, algebra_dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x_mv, y_mv)`` where ``y = R_theta @ x`` for a fixed ``theta``."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(n, 2, generator=g)
    cos_t, sin_t = math.cos(angle_rad), math.sin(angle_rad)
    rot = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = pts @ rot.T
    return _embed_as_vector(pts, algebra_dim), _embed_as_vector(rotated, algebra_dim)


# ---------------------------------------------------------------------------
# Model — the smallest GBN that can represent a rotor sandwich.
# ---------------------------------------------------------------------------


class RotorRegressorNet(CliffordModule):
    """Embed → (norm → rotor → act → linear) * num_layers → output.

    The block structure is inlined here — not imported from ``_lib`` — because
    each incubator experiment composes its own stack. This template just shows
    the canonical four-step block the rest of the codebase uses.
    """

    def __init__(self, algebra, hidden_channels: int = 8, num_layers: int = 2):
        super().__init__(algebra)
        self.hidden_channels = hidden_channels
        self.in_proj = CliffordLinear(algebra, 1, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': CliffordLayerNorm(algebra, hidden_channels),
                        'rotor': RotorLayer(algebra, hidden_channels),
                        'act': GeometricGELU(algebra, channels=hidden_channels),
                        'linear': CliffordLinear(algebra, hidden_channels, hidden_channels),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = CliffordLayerNorm(algebra, hidden_channels)
        self.blade_select = BladeSelector(algebra, channels=hidden_channels)
        self.out_proj = CliffordLinear(algebra, hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            residual = h
            h = block['norm'](h)
            h = block['rotor'](h)
            h = block['act'](h)
            h = block['linear'](h)
            h = residual + h
        h = self.out_norm(h)
        h = self.blade_select(h)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# Train / eval — local, not imported from _lib (loops vary by experiment).
# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total += F.mse_loss(pred, y).item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = make_experiment_parser(
        'Incubator template — 2-D rotation regression.',
        include=('seed', 'device', 'epochs', 'lr', 'batch_size', 'output_dir', 'diag_interval'),
        defaults={'epochs': 30, 'batch_size': 64, 'output_dir': 'template_inc_plots', 'diag_interval': 5},
    )
    parser.add_argument('--hidden-channels', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--n-train', type=int, default=512)
    parser.add_argument('--n-test', type=int, default=128)
    parser.add_argument('--angle-deg', type=float, default=45.0, help='Target rotation angle in degrees.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    algebra = setup_algebra(p=2, q=0, r=0, device=args.device)
    model = RotorRegressorNet(algebra, args.hidden_channels, args.num_layers).to(args.device)

    print_banner(
        'Incubator Template — 2-D Rotation Regression',
        signature='Cl(2, 0)',
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        params=count_parameters(model),
        target_angle_deg=args.angle_deg,
    )

    angle_rad = math.radians(args.angle_deg)
    x_train, y_train = make_rotation_dataset(args.n_train, angle_rad, algebra.dim, args.seed)
    x_test, y_test = make_rotation_dataset(args.n_test, angle_rad, algebra.dim, args.seed + 1)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)

    history: Dict[str, List[float]] = {'epochs': [], 'train_loss': [], 'test_loss': []}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        if epoch % args.diag_interval == 0 or epoch == args.epochs:
            test_loss = evaluate(model, test_loader, args.device)
            history['epochs'].append(epoch)
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            print(f'Epoch {epoch:4d}/{args.epochs} | train={train_loss:.6f} | test={test_loss:.6f}')

    out_dir = ensure_output_dir(args.output_dir)
    saved = save_training_curve(
        history, os.path.join(out_dir, 'training_curves.png'), title='Incubator Template — training curves'
    )
    print(f'Saved {saved}')


if __name__ == '__main__':
    main()
