# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Versor Counterpart: Geometric Algebra Transformer (GATr)
Johann Brehmer, Pim de Haan, Sönke Behrends, Taco Cohen (NeurIPS 2023)
arXiv: https://arxiv.org/abs/2305.18415
Original: https://github.com/Qualcomm-AI-research/geometric-algebra-transformer (~2500+ lines)
Versor counterpart: ~80 lines (synthetic n-body data, not a benchmark reproduction)

What the paper contributes:
  GATr builds E(3)-equivariant transformers in Projective Geometric Algebra
  Cl(3,0,1). The key contribution is a complete set of equivariant primitives —
  linear maps derived from representation theory, geometric product attention
  scoring (<QK~>_0 + lambda * ||<QK~>_2||), gated nonlinearities, and
  join-based normalization — each hand-derived to commute with the PGA group
  action. The original requires ~2500 lines of custom equivariant code.

Versor's approach (architecturally close, different construction path):
  This is the most faithful of the three counterparts. The architecture follows
  GATr's structure: PGA embedding -> lift -> transformer blocks -> project ->
  extract. Versor's existing primitives map to GATr's components:

  - ProjectiveEmbedding  ->  GATr's PGA point/direction embedding (faithful)
  - GeometricProductAttention  ->  GATr's equivariant attention (same GP scoring
    formula, precomputed bilinear tables for memory efficiency)
  - GeometricGELU / MultiRotorFFN  ->  GATr's equivariant nonlinearity
  - CliffordLayerNorm  ->  GATr's join-based normalization (different mechanism:
    norm-preserving vs join computation, same stabilization goal)
  - CliffordLinear  ->  GATr's equivariant linear maps (different construction:
    Cayley table coefficient mixing vs representation-theoretic basis derivation)

  The architecture is structurally similar, but the equivariance mechanism differs:
  GATr derives equivariant bases via representation theory; Versor operates
  directly on multivector coefficients where the Cayley table enforces
  equivariance by construction. Same guarantee, different derivation.

What's verified (synthetic data):
  E(3) equivariance on synthetic n-body spring dynamics:
  - f(Rx, Rv) ≈ R·f(x, v) for SO(3) rotation R
  - f(x+t, v) ≈ f(x, v) + t for translation t
  These test the architectural property, not real-world prediction quality.

PGA Convention (Cl(3,0,1)):
  Versor follows the standard PGA convention from Gunn (2011) and
  De Keninck & Dorst (2019), consistent with GATr:
    - Basis: e_1, e_2, e_3 (Euclidean, e_i^2 = +1), e_0 (degenerate, e_0^2 = 0)
    - e_0 is the LAST basis vector (index p+q = 3 in the algebra), stored at
      binary index 2^3 = 8 in the 16-dimensional multivector.
    - Finite point:  P(x) = x_1*e_1 + x_2*e_2 + x_3*e_3 + 1*e_0
    - Direction:     D(v) = v_1*e_1 + v_2*e_2 + v_3*e_3 + 0*e_0  (ideal point)
    - Extraction normalizes by the e_0 coefficient (homogeneous division).
  This matches GATr Table 1 where points carry e_0 = 1 and translations act
  via e_0 sandwiches. See layers/adapters/projective.py for implementation.
"""

import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from core.algebra import CliffordAlgebra
from layers import CliffordLinear, GeometricTransformerBlock
from layers.adapters.projective import ProjectiveEmbedding
from tasks.base import BaseTask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GATrNet(nn.Module):
    """Geometric Algebra Transformer for n-body dynamics.

    Architecture mirrors GATr (Brehmer et al. 2023):
      1. Embed positions as PGA points (e_0 = 1) and velocities as
         PGA directions (e_0 = 0) — two input channels per particle.
      2. CliffordLinear lifts to hidden channels.
      3. Stack of GeometricTransformerBlocks (GP attention + multi-rotor FFN).
      4. CliffordLinear projects back to 1 channel, extract R^3 coordinates.

    Particles are treated as sequence tokens: each particle attends to all
    others via geometric product attention, naturally capturing pairwise
    geometric relationships (relative position, orientation).
    """

    def __init__(self, algebra, channels=8, num_heads=4, num_layers=2):
        super().__init__()
        # PGA embedding: R^3 <-> grade-1 of Cl(3,0,1)
        # Points get e_0 = 1 (finite), velocities get e_0 = 0 (ideal/direction)
        self.pga = ProjectiveEmbedding(algebra, euclidean_dim=3)

        # pos + vel = 2 input channels per particle
        self.lift = CliffordLinear(algebra, 2, channels)
        self.blocks = nn.ModuleList([
            GeometricTransformerBlock(
                algebra, channels, num_heads=num_heads,
                num_rotors=4, dropout=0.0,
            )
            for _ in range(num_layers)
        ])
        self.project = CliffordLinear(algebra, channels, 1)

    def forward(self, pos, vel):
        """Predict next-step positions.

        The model predicts a displacement (direction/ideal point, e_0 = 0)
        which is added to the input position. This is more stable than
        predicting absolute PGA points (which require homogeneous division
        by e_0), and more physical — the model learns how particles move.

        Args:
            pos: Particle positions [B, N, 3].
            vel: Particle velocities [B, N, 3].

        Returns:
            Predicted next positions [B, N, 3].
        """
        B, N, _ = pos.shape

        # Embed into PGA grade-1 multivectors (16-dim for Cl(3,0,1))
        # Points: x1*e1 + x2*e2 + x3*e3 + 1*e0  (finite, e0 coeff = 1)
        # Directions: v1*e1 + v2*e2 + v3*e3       (ideal, e0 coeff = 0)
        p_mv = self.pga.embed(pos)            # [B, N, 16]
        v_mv = self.pga.embed_direction(vel)  # [B, N, 16]

        # Stack as 2 channels: [B, N, 2, 16]
        x = torch.stack([p_mv, v_mv], dim=2)

        # Lift to hidden channels: [B*N, 2, 16] -> [B*N, C, 16]
        x = self.lift(x.reshape(B * N, 2, -1))
        x = x.reshape(B, N, -1, x.shape[-1])  # [B, N, C, 16]

        # Transformer blocks (expect [B, L, C, D])
        # Each particle is a token; attention captures pairwise interactions
        for block in self.blocks:
            x = block(x)

        # Project back: [B, N, C, 16] -> [B*N, 1, 16]
        x = self.project(x.reshape(B * N, -1, x.shape[-1]))
        x = x.reshape(B, N, -1)  # [B, N, 16]

        # Extract displacement as direction (read grade-1 Euclidean components)
        # Avoids unstable homogeneous division; displacement is an ideal point
        disp = self.pga.embed_direction(torch.zeros_like(pos))  # template
        disp = torch.gather(x, -1, self.pga._g1_idx.expand(B, N, 3))

        return pos + disp


# ---------------------------------------------------------------------------
# Synthetic data: N-body spring system
# ---------------------------------------------------------------------------

def _generate_nbody_data(n_samples, n_particles, dt=0.01, n_steps=10):
    """Generate n-body spring system trajectories.

    Particles interact via pairwise spring forces (Hookean, unit rest length).
    Uses Euler integration to compute target next-state positions.

    Returns:
        positions [S, N, 3], velocities [S, N, 3], targets [S, N, 3]
    """
    positions = torch.zeros(n_samples, n_particles, 3)
    velocities = torch.zeros(n_samples, n_particles, 3)
    targets = torch.zeros(n_samples, n_particles, 3)

    k_spring = 1.0

    for s in range(n_samples):
        pos = torch.rand(n_particles, 3) * 2 - 1
        vel = torch.rand(n_particles, 3) * 1.0 - 0.5

        positions[s] = pos
        velocities[s] = vel

        p, v = pos.clone(), vel.clone()
        for _ in range(n_steps):
            forces = torch.zeros_like(p)
            for i in range(n_particles):
                for j in range(n_particles):
                    if i == j:
                        continue
                    r = p[j] - p[i]
                    dist = r.norm() + 1e-6
                    forces[i] += k_spring * r / dist

            v = v + forces * dt
            p = p + v * dt

        targets[s] = p

    return positions, velocities, targets


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class GATrTask(BaseTask):
    """GATr-style n-body dynamics prediction.

    Demonstrates E(3)-equivariant prediction: rotating/translating the input
    system produces correspondingly rotated/translated output. Verified by
    the equivariance test in evaluate().
    """

    def setup_algebra(self):
        # Cl(3,0,1): 3 Euclidean + 1 degenerate = Projective Geometric Algebra
        # dim = 2^4 = 16 multivector components
        r = self.cfg.algebra.get('r', 1)
        return CliffordAlgebra(
            p=self.cfg.algebra.p, q=self.cfg.algebra.q,
            r=r, device=self.device,
        )

    def setup_model(self):
        return GATrNet(self.algebra, channels=8, num_heads=4, num_layers=2)

    def setup_criterion(self):
        return nn.MSELoss()

    def get_data(self):
        n_particles = self.cfg.dataset.get('n_particles', 5)
        n_samples = self.cfg.dataset.get('n_samples', 512)

        torch.manual_seed(self.cfg.training.get('seed', 42))
        pos, vel, tgt = _generate_nbody_data(n_samples, n_particles)
        dataset = TensorDataset(pos, vel, tgt)
        torch.manual_seed(torch.seed())  # re-randomize
        return DataLoader(
            dataset, batch_size=self.cfg.training.batch_size, shuffle=True,
        )

    def train_step(self, data):
        pos, vel, tgt = [d.to(self.device) for d in data]

        self.optimizer.zero_grad()
        pred = self.model(pos, vel)
        loss = self.criterion(pred, tgt)
        loss.backward()
        self.optimizer.step()

        return loss.item(), {"Loss": loss.item()}

    def evaluate(self, data):
        pos, vel, tgt = [d.to(self.device) for d in data]
        pred = self.model(pos, vel)

        mse = self.criterion(pred, tgt).item()
        rmse = math.sqrt(mse)
        print(f"N-body prediction RMSE: {rmse:.6f}")

        # --- E(3) Equivariance test ---
        # GATr's main claim: the architecture is E(3)-equivariant.
        # Verify: f(R*x, R*v) == R * f(x, v) for rotation R.

        # SO(3) rotation: Rz(0.7) @ Rx(0.3)
        a, b = torch.tensor(0.7), torch.tensor(0.3)
        Rz = torch.eye(3, device=self.device)
        Rz[0, 0], Rz[0, 1] = a.cos(), -a.sin()
        Rz[1, 0], Rz[1, 1] = a.sin(), a.cos()
        Rx = torch.eye(3, device=self.device)
        Rx[1, 1], Rx[1, 2] = b.cos(), -b.sin()
        Rx[2, 1], Rx[2, 2] = b.sin(), b.cos()
        R = Rz @ Rx

        pos_rot = pos @ R.T
        vel_rot = vel @ R.T
        pred_rot = self.model(pos_rot, vel_rot)
        pred_then_rot = pred @ R.T

        rot_err = (pred_rot - pred_then_rot).norm() / (pred.norm() + 1e-8)
        print(f"Rotation equivariance error: {rot_err.item():.6f}")

        # Translation equivariance: f(x + t, v) == f(x, v) + t
        t = torch.tensor([0.5, -0.3, 0.7], device=self.device)
        pred_shifted = self.model(pos + t, vel)
        shift_err = (pred_shifted - (pred + t)).norm() / (pred.norm() + 1e-8)
        print(f"Translation equivariance error: {shift_err.item():.6f}")

    def visualize(self, data):
        pass
