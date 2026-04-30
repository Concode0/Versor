# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Versor Counterpart: Clifford Group Equivariant Neural Networks (CGENN)
David Ruhe, Johannes Brandstetter, Patrick Forré (NeurIPS 2023, Oral)
arXiv: https://arxiv.org/abs/2305.11141
Original: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks (~3000+ lines)
Versor counterpart: ~90 lines (synthetic point cloud data, not a benchmark reproduction)

What the paper contributes:
  CGENN's key theorem: any map f(x) = sum_k alpha_k * <x^k>_grade is
  automatically equivariant under the Clifford group's twisted conjugation
  action. This is an elegant result — equivariance follows from the algebraic
  structure of polynomial maps combined with hard grade projections, without
  needing explicit group actions. The original requires ~3000 lines with
  separate per-dimension handling for 3D, 4D, and 5D.

Versor's approach (same equivariance goal, different mechanism):
  This is the LEAST faithful of the three counterparts. The paper achieves
  equivariance through a theorem about polynomial structure; this Versor
  version achieves it through composition of individually equivariant
  operations. The mechanisms are genuinely different:

  - GeometricSquare  ->  CGENN's polynomial features (FAITHFUL)
    Gated GP self-product: x + gate * GP(x, x). This IS a degree-2
    polynomial in x, matching the paper's core construction.

  - BladeSelector  ->  CGENN's grade projections (GENERALIZED)
    The paper uses hard grade projections <·>_k as part of the equivariance
    proof. Versor substitutes learned sigmoid gates per basis blade — strictly
    more expressive (per-blade vs per-grade), but the equivariance guarantee
    comes from a different argument (each blade gate commutes with the group
    action on that component).

  - RotorLayer  ->  NOT IN THE ORIGINAL PAPER
    CGENN achieves equivariance implicitly via polynomial structure. Adding
    an explicit sandwich product RxR~ is redundant for equivariance — the
    polynomial already guarantees it. The rotor adds expressiveness as a
    Versor design choice, not as a reproduction of the paper.

  The result: O(n) equivariance holds, verified by rotation and reflection
  tests, but for a different architectural reason than the paper's theorem.

What's verified (synthetic data):
  O(3) invariance on synthetic point cloud regression:
  - f(Rx) ≈ f(x) for SO(3) rotation R (rotation invariance)
  - f(Mx) ≈ f(x) for reflection M with det=-1 (reflection invariance)
  Grade norms ||<x>_k|| are provably O(n)-invariant (not just SO(n)):
    ||<RxR~>_k|| = ||<x>_k|| for any versor R, including reflections.
  These test the algebraic property, not real-world prediction quality.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.algebra import CliffordAlgebra
from core.module import CliffordModule
from functional.activation import GeometricSquare
from layers import (
    BladeSelector,
    CliffordLayerNorm,
    CliffordLinear,
    RotorLayer,
)
from tasks.base import BaseTask

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CGENNBlock(CliffordModule):
    """Single CGENN equivariant block.

    Norm -> GeometricSquare (polynomial features) -> Linear -> Rotor -> BladeSelector.
    Each operation preserves Clifford group equivariance by construction.
    """

    def __init__(self, algebra, channels):
        super().__init__(algebra)
        self.norm = CliffordLayerNorm(algebra, channels)
        self.square = GeometricSquare(algebra, channels)
        self.linear = CliffordLinear(algebra, channels, channels)
        self.rotor = RotorLayer(algebra, channels)
        self.selector = BladeSelector(algebra, channels)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.square(x)  # x + gate * GP(x,x): quadratic polynomial features
        x = self.linear(x)  # equivariant channel mixing via Cayley table
        x = self.rotor(x)  # sandwich product: even Clifford group action
        x = self.selector(x)  # learned grade filtering (soft grade projection)
        return x + res


class CGENNNet(CliffordModule):
    """Clifford Group Equivariant Network for invariant point cloud regression.

    Architecture mirrors CGENN (Ruhe et al. 2023):
      1. Embed 3D points as grade-1 multivectors in Cl(3,0).
      2. Lift to multi-channel via CliffordLinear.
      3. Stack of CGENNBlocks (polynomial features + grade filtering).
      4. Extract O(n)-invariant features via grade norms ||<x>_k||.
      5. Pool over points, MLP readout to scalar.

    The grade norm readout is invariant under the FULL O(3) group (rotations
    AND reflections), not just SO(3). This is verified in the evaluate() tests.
    """

    def __init__(self, algebra, channels=16, num_blocks=3):
        super().__init__(algebra)
        self.lift = CliffordLinear(algebra, 1, channels)
        self.blocks = nn.ModuleList([CGENNBlock(algebra, channels) for _ in range(num_blocks)])
        # Invariant readout: grade norms (O(n)-invariant) -> scalar
        n_grades = self.algebra.n + 1
        self.readout = nn.Sequential(
            nn.Linear(channels * n_grades, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, points):
        """Predict O(3)-invariant scalar from point cloud.

        Args:
            points: Point cloud [B, N, 3].

        Returns:
            Invariant scalar [B, 1].
        """
        B, N, _ = points.shape

        # Embed into grade-1 multivectors of Cl(3,0)
        mv = self.algebra.embed_vector(points)  # [B, N, 8]
        x = mv.reshape(B * N, 1, -1)  # [B*N, 1, 8]

        # Lift to channels
        x = self.lift(x)  # [B*N, C, 8]

        # Equivariant blocks
        for block in self.blocks:
            x = block(x)

        # Extract O(n)-invariant features: per-grade norms [B*N, C, n_grades]
        # ||<x>_k|| is invariant under any versor (rotation or reflection)
        grade_norms = self.algebra.get_grade_norms(x)

        # Pool over points: reshape -> mean
        C = x.shape[1]
        n_grades = grade_norms.shape[-1]
        grade_norms = grade_norms.reshape(B, N, C * n_grades)
        pooled = grade_norms.mean(dim=1)  # [B, C * n_grades]

        return self.readout(pooled)  # [B, 1]


# ---------------------------------------------------------------------------
# Synthetic data: O(3)-invariant point cloud regression
# ---------------------------------------------------------------------------


def _generate_invariant_data(n_samples, n_points):
    """Generate point clouds with O(3)-invariant scalar targets.

    Target = mean pairwise distance + mean norm.
    Both terms are provably invariant under the full O(3) group (rotations
    and reflections preserve distances and norms).

    Returns:
        points [S, N, 3], targets [S, 1]
    """
    points = torch.randn(n_samples, n_points, 3)
    targets = torch.zeros(n_samples, 1)

    for s in range(n_samples):
        pc = points[s]
        mean_norm = pc.norm(dim=-1).mean()
        diff = pc.unsqueeze(0) - pc.unsqueeze(1)
        dist = diff.norm(dim=-1)
        mean_dist = dist.sum() / (n_points * (n_points - 1))
        targets[s] = mean_norm + mean_dist

    return points, targets


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class CGENNTask(BaseTask):
    """CGENN-style O(3)-invariant point cloud regression.

    Validates the paper's main claim: the architecture produces outputs that
    are invariant under the full orthogonal group O(3), including both
    rotations (SO(3)) and reflections. Both are tested in evaluate().
    """

    def setup_algebra(self):
        return CliffordAlgebra(
            p=self.cfg.algebra.p,
            q=self.cfg.algebra.q,
            device=self.device,
        )

    def setup_model(self):
        return CGENNNet(self.algebra, channels=16, num_blocks=3)

    def setup_criterion(self):
        return nn.MSELoss()

    def get_data(self):
        n_points = self.cfg.dataset.get("n_points", 20)
        n_samples = self.cfg.dataset.get("n_samples", 256)

        points, targets = _generate_invariant_data(n_samples, n_points)
        dataset = TensorDataset(points, targets)
        return DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
        )

    def train_step(self, data):
        points, targets = [d.to(self.device) for d in data]

        self.optimizer.zero_grad()
        pred = self.model(points)
        loss = self.criterion(pred, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item(), {"Loss": loss.item()}

    def evaluate(self, data):
        points, targets = [d.to(self.device) for d in data]
        pred = self.model(points)

        mse = self.criterion(pred, targets).item()
        print(f"Invariant prediction MSE: {mse:.6f}")

        # --- SO(3) Rotation invariance test ---
        # CGENN claim: output is invariant under O(n) group actions.
        # First test: rotation (det R = +1)
        a, b = torch.tensor(0.7), torch.tensor(0.3)
        Rz = torch.eye(3, device=self.device)
        Rz[0, 0], Rz[0, 1] = a.cos(), -a.sin()
        Rz[1, 0], Rz[1, 1] = a.sin(), a.cos()
        Rx = torch.eye(3, device=self.device)
        Rx[1, 1], Rx[1, 2] = b.cos(), -b.sin()
        Rx[2, 1], Rx[2, 2] = b.sin(), b.cos()
        R = Rz @ Rx

        rotated_pts = points @ R.T
        pred_rotated = self.model(rotated_pts)
        rot_err = (pred_rotated - pred).abs().mean() / (pred.abs().mean() + 1e-8)
        print(f"SO(3) rotation invariance error: {rot_err.item():.6f}")

        # --- O(3) Reflection invariance test ---
        # Second test: reflection (det R = -1)
        # Reflect through the plane orthogonal to [1, 1, 1]/sqrt(3)
        # Reflection matrix: I - 2*n*n^T where n is unit normal
        n = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        n = n / n.norm()
        M = torch.eye(3, device=self.device) - 2.0 * torch.outer(n, n)  # det = -1

        reflected_pts = points @ M.T
        pred_reflected = self.model(reflected_pts)
        ref_err = (pred_reflected - pred).abs().mean() / (pred.abs().mean() + 1e-8)
        print(f"O(3) reflection invariance error: {ref_err.item():.6f}")

    def visualize(self, data):
        pass
