# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Cl(1,1) learnable search controller with rook movement.

4D state: cursor = s*1 + h*e3 + d*e4 + p*e34
  s  (scalar):   confidence
  h  (e3):       hypothesis index
  d  (e4):       depth
  p  (e34):      phase (bivector)

Rook movement: only horizontal OR vertical per step.
  Horizontal boost: R_h = exp(alpha * e34) — boosts e3 (explore hypotheses)
  Vertical boost:   R_v = exp(-beta * e34) — boosts e4 (go deeper)
  Direction gate (sigmoid MLP): selects pos' = gate*R_h(pos) + (1-gate)*R_v(pos)

e34 is hyperbolic (sq=+1 in Cl(1,1)), so boosts use cosh/sinh.
"""

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.primitives.base import CliffordModule


class ControlPlane(CliffordModule):
    """Cl(1,1) learnable search controller."""

    def __init__(self, algebra_ctrl: CliffordAlgebra, channels: int,
                 max_hypotheses: int = 4):
        assert algebra_ctrl.p == 1 and algebra_ctrl.q == 1, \
            f"ControlPlane requires Cl(1,1), got Cl({algebra_ctrl.p},{algebra_ctrl.q})"
        super().__init__(algebra_ctrl)
        self.channels = channels
        self.max_hypotheses = max_hypotheses
        # Cl(1,1) dim = 4: {1, e3, e4, e34} mapped to indices {0, 1, 2, 3}

        # Boost parameters (learnable)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(channels + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(channels + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Direction gate: horizontal vs vertical
        self.direction_gate = nn.Sequential(
            nn.Linear(channels + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Residual correction for boost-invariant components (scalar, e34)
        # Sandwich product with e34 bivector only boosts grade-1 (e3, e4);
        # scalar and pseudoscalar are algebraically invariant. This MLP
        # provides a learned additive update so those components can evolve.
        # Outputs 2 values: (delta_scalar, delta_e34), NOT all 4 components,
        # to avoid double-counting with the boost on e3/e4.
        self.cursor_residual = nn.Sequential(
            nn.Linear(channels + 4, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
        # Initialize near-zero so early training is dominated by the boost
        nn.init.zeros_(self.cursor_residual[-1].weight)
        nn.init.zeros_(self.cursor_residual[-1].bias)

        # Halt signal from cursor
        self.halt_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def step(self, cursor: torch.Tensor,
             cpu_context: torch.Tensor) -> tuple:
        """Advance the control cursor one step.

        Args:
            cursor: Current cursor [B, 4] in Cl(1,1).
            cpu_context: Summary of CPU state [B, channels] (e.g., mean-pooled grade norms).

        Returns:
            Tuple of (new_cursor [B, 4], direction_logit [B, 1], halt_prob [B]).
        """
        B = cursor.shape[0]
        device = cursor.device
        self.algebra.ensure_device(device)

        # Combine cursor with CPU context for MLPs
        combined = torch.cat([cursor, cpu_context], dim=-1)  # [B, 4 + channels]

        # Compute boost magnitudes
        alpha = self.alpha_mlp(combined).squeeze(-1)  # [B]
        beta = self.beta_mlp(combined).squeeze(-1)     # [B]

        # Build bivector for horizontal boost: alpha * e34
        bv_h = torch.zeros(B, 4, device=device, dtype=cursor.dtype)
        bv_h[:, 3] = alpha  # e34 component

        # Build bivector for vertical boost: -beta * e34
        bv_v = torch.zeros(B, 4, device=device, dtype=cursor.dtype)
        bv_v[:, 3] = -beta  # e34 component

        # Exponentiate boosts
        R_h = self.algebra.exp(-0.5 * bv_h)  # [B, 4]
        R_v = self.algebra.exp(-0.5 * bv_v)  # [B, 4]

        # Apply boosts to cursor via sandwich product
        # For Cl(1,1), we can use geometric_product directly (1D batch)
        R_h_rev = self.algebra.reverse(R_h)
        R_v_rev = self.algebra.reverse(R_v)

        cursor_h = self.algebra.geometric_product(
            self.algebra.geometric_product(R_h, cursor), R_h_rev
        )
        cursor_v = self.algebra.geometric_product(
            self.algebra.geometric_product(R_v, cursor), R_v_rev
        )

        # Direction gate
        direction_logit = self.direction_gate(combined)  # [B, 1]
        gate = torch.sigmoid(direction_logit)  # [B, 1]
        new_cursor = gate * cursor_h + (1.0 - gate) * cursor_v  # [B, 4]

        # Residual correction: only update boost-invariant components
        delta = self.cursor_residual(combined)  # [B, 2] -> (delta_scalar, delta_e34)
        new_cursor = new_cursor.clone()
        new_cursor[:, 0] = new_cursor[:, 0] + delta[:, 0]  # scalar
        new_cursor[:, 3] = new_cursor[:, 3] + delta[:, 1]  # e34

        # Symmlog normalization: prevents unbounded drift across steps
        # while preserving gradient (grad = 1/(1+|x|), never zero)
        new_cursor = torch.sign(new_cursor) * torch.log1p(new_cursor.abs())

        # Halt probability from grade-0 of cursor
        halt_prob = torch.sigmoid(self.halt_mlp(new_cursor)).squeeze(-1)  # [B]

        return new_cursor, direction_logit, halt_prob
