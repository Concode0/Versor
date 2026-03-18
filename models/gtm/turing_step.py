# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Single GTM step: SuperpositionSearch + Cross-Grade Attention + ControlPlane.

Key design choices:
  - NO additive residual (addition destroys geometric structure after rotations).
    Instead, the instruction can learn B~0 to approximate identity.
  - Cross-grade dense Q/K attention: allows learning diagonal, distance,
    and other 2D spatial relationships.
  - Values remain raw multivectors with per-grade gain (preserves geometry).
  - Geometric gating (rotor interpolation) instead of additive skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from layers.primitives.normalization import CliffordLayerNorm
from .superposition import GeometricSuperpositionSearch
from .control_plane import ControlPlane


# Grade-to-index mapping for Cl(3,0,1), dim=16
# Grade 0: [0]
# Grade 1: [1, 2, 4, 8]
# Grade 2: [3, 5, 6, 9, 10, 12]
# Grade 3: [7, 11, 13, 14]
# Grade 4: [15]
_GRADE_MAP_16 = torch.zeros(16, dtype=torch.long)
_GRADE_MAP_16[0] = 0
_GRADE_MAP_16[[1, 2, 4, 8]] = 1
_GRADE_MAP_16[[3, 5, 6, 9, 10, 12]] = 2
_GRADE_MAP_16[[7, 11, 13, 14]] = 3
_GRADE_MAP_16[15] = 4


class CellAttention(nn.Module):
    """Cross-grade self-attention over grid cells in Cl(3,0,1).

    Dense Q/K projections allow learning cross-grade features like
    diagonals (e0+e1), distances, and 2D spatial relationships.
    Values remain raw multivectors with per-grade gain to preserve
    geometric structure in the convex combination.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra, num_heads: int = 4,
                 head_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        D = algebra_cpu.dim  # 16
        attn_dim = num_heads * head_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Dense Q, K projections: allow cross-grade mixing for scoring
        self.q_proj = nn.Linear(D, attn_dim)
        self.k_proj = nn.Linear(D, attn_dim)

        # Per-grade gain on values (preserves geometric structure)
        self.v_gain = nn.ParameterDict({
            f'g{k}': nn.Parameter(torch.ones(1)) for k in range(5)
        })

        self.dropout = nn.Dropout(dropout)

        # Grade map buffer for applying per-grade gains
        self.register_buffer('grade_map', _GRADE_MAP_16.clone())

    def _apply_grade_gains(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-grade isotropic gains to multivector components."""
        gains = torch.ones(16, device=x.device, dtype=x.dtype)
        for k in range(5):
            mask = self.grade_map == k
            gains[mask] = self.v_gain[f'g{k}']
        return x * gains

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Cross-grade self-attention over cells.

        Args:
            x: [B, N, 16] multivectors in Cl(3,0,1).
            mask: [B, N] bool, True=valid.

        Returns:
            [B, N, 16] attended multivectors.
        """
        B, N, D = x.shape

        # Dense Q, K projections
        Q = self.q_proj(x)  # [B, N, attn_dim]
        K = self.k_proj(x)  # [B, N, attn_dim]

        # Multi-head reshape
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, hd]
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, hd]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        if mask is not None:
            pad_mask = ~mask
            scores = scores.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(scores, dim=-1)  # [B, H, N, N]
        attn = self.dropout(attn)
        attn_avg = attn.mean(dim=1)       # [B, N, N]

        # Values: raw multivector, weighted average (convex combination)
        attended = torch.bmm(attn_avg, x)  # [B, N, 16]

        # Per-grade gain on output
        return self._apply_grade_gains(attended)


class TuringStep(nn.Module):
    """One step of the Geometric Turing Machine.

    Composes:
        1. Cell attention (cross-cell communication with cross-grade features)
        2. Superposition search (per-cell transformation via PGA motor)
        3. Geometric write gate (interpolates via scalar gating, no additive residual)
        4. CliffordLayerNorm
        5. Control plane step
    """

    def __init__(self, algebra_cpu: CliffordAlgebra,
                 algebra_ctrl: CliffordAlgebra,
                 channels: int,
                 num_hypotheses: int = 4,
                 top_k: int = 1,
                 temperature_init: float = 1.0,
                 num_attn_heads: int = 4,
                 attn_head_dim: int = 8,
                 attn_dropout: float = 0.0,
                 K_color: int = 4,
                 num_rule_slots: int = 8):
        super().__init__()
        self.channels = channels
        D_cpu = algebra_cpu.dim  # 16

        # Cell-to-cell attention (cross-grade features)
        self.cell_attn = CellAttention(
            algebra_cpu, num_attn_heads, attn_head_dim, attn_dropout,
        )

        # Superposition search module (no mother algebra)
        self.search = GeometricSuperpositionSearch(
            algebra_cpu, algebra_ctrl,
            channels, num_hypotheses, top_k, temperature_init,
            K_color, num_rule_slots,
        )

        # Control plane
        self.control = ControlPlane(algebra_ctrl, channels)

        # CPU state normalization
        self.norm = CliffordLayerNorm(algebra_cpu, 1)  # per-cell norm (C=1)

        # Context projection: cpu summary -> ctrl context
        self.context_proj = nn.Linear(D_cpu, channels)

        # Geometric write gate: scalar gate per cell
        self.write_gate = nn.Sequential(
            nn.Linear(D_cpu * 2, 64),  # concat(old, new) -> 64
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cpu_state: torch.Tensor,
                ctrl_cursor: torch.Tensor,
                mask: torch.Tensor = None,
                rule_memory: torch.Tensor = None) -> dict:
        """Execute one GTM step.

        Args:
            cpu_state: [B, N, 16] CPU state in Cl(3,0,1).
            ctrl_cursor: [B, 4] control cursor in Cl(1,1).
            mask: Optional [B, N] validity mask (True=valid).
            rule_memory: Optional [B, M, 16] rule slots from RuleAggregator.

        Returns:
            dict with 'cpu_state', 'ctrl_cursor', 'halt_prob', 'search_info'.
        """
        old_state = cpu_state

        # 1. Cell attention (cross-cell communication)
        attended = self.cell_attn(cpu_state, mask)

        # 2. Superposition search (per-cell transformation via PGA motor)
        new_cpu, search_info = self.search.step(attended, ctrl_cursor, rule_memory)

        # 3. Geometric write gate (NO additive residual)
        gate_input = torch.cat([old_state, new_cpu], dim=-1)  # [B, N, 32]
        gate = torch.sigmoid(self.write_gate(gate_input))  # [B, N, 1]
        new_cpu = gate * new_cpu + (1.0 - gate) * old_state

        # 4. CliffordLayerNorm (per-cell)
        B, N, D = new_cpu.shape
        new_cpu_flat = new_cpu.reshape(B * N, 1, D)
        new_cpu_flat = self.norm(new_cpu_flat)
        new_cpu = new_cpu_flat.reshape(B, N, D)

        # 5. Control plane step
        cpu_summary = new_cpu.mean(dim=1)  # [B, 16]
        cpu_context = self.context_proj(cpu_summary)  # [B, channels]
        new_cursor, direction_logit, halt_prob = self.control.step(
            ctrl_cursor, cpu_context
        )

        return {
            'cpu_state': new_cpu,
            'ctrl_cursor': new_cursor,
            'halt_prob': halt_prob,
            'search_info': search_info,
            'gate_values': gate,
        }
