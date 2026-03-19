# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Single GTM step: SuperpositionSearch + Cross-Grade Attention + ControlPlane."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from layers.primitives.normalization import CliffordLayerNorm
from .superposition import GeometricSuperpositionSearch
from .control_plane import ControlPlane


_GRADE_MAP_16 = torch.zeros(16, dtype=torch.long)
_GRADE_MAP_16[0] = 0
_GRADE_MAP_16[[1, 2, 4, 8]] = 1
_GRADE_MAP_16[[3, 5, 6, 9, 10, 12]] = 2
_GRADE_MAP_16[[7, 11, 13, 14]] = 3
_GRADE_MAP_16[15] = 4


class CellAttention(nn.Module):
    """Cross-grade self-attention over grid cells in Cl(3,0,1)."""

    def __init__(self, algebra_cpu: CliffordAlgebra, num_heads: int = 4,
                 head_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        D = algebra_cpu.dim
        attn_dim = num_heads * head_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(D, attn_dim)
        self.k_proj = nn.Linear(D, attn_dim)
        self.v_gain = nn.ParameterDict({
            f'g{k}': nn.Parameter(torch.ones(1)) for k in range(5)
        })
        self.dropout = nn.Dropout(dropout)
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
        """[B, N, 16] -> [B, N, 16] with optional mask [B, N]."""
        B, N, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(
                (~mask).unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_avg = attn.mean(dim=1)

        attended = torch.bmm(attn_avg, x)
        return self._apply_grade_gains(attended)


class TuringStep(nn.Module):
    """One step of the Geometric Turing Machine."""

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
        D_cpu = algebra_cpu.dim

        self.cell_attn = CellAttention(
            algebra_cpu, num_attn_heads, attn_head_dim, attn_dropout,
        )
        self.search = GeometricSuperpositionSearch(
            algebra_cpu, algebra_ctrl,
            channels, num_hypotheses, top_k, temperature_init,
            K_color, num_rule_slots,
        )
        self.control = ControlPlane(algebra_ctrl, channels)
        self.norm = CliffordLayerNorm(algebra_cpu, 1)
        self.context_proj = nn.Linear(D_cpu, channels)
        # Per-component gate: enables cross-grade mixing.
        # With scalar gate (old), color (g0) and position (g1) always move
        # together.  Per-component gate lets the model selectively update
        # color based on position context and vice versa.
        self.write_gate = nn.Sequential(
            nn.Linear(D_cpu * 2, 64),
            nn.ReLU(),
            nn.Linear(64, D_cpu),
        )

    def set_temperature(self, tau: float):
        self.search.set_temperature(tau)

    def forward(self, cpu_state: torch.Tensor,
                ctrl_cursor: torch.Tensor,
                mask: torch.Tensor = None,
                rule_memory: torch.Tensor = None) -> dict:
        old_state = cpu_state

        attended = self.cell_attn(cpu_state, mask)
        new_cpu, search_info = self.search.step(attended, ctrl_cursor, rule_memory)

        gate_input = torch.cat([old_state, new_cpu], dim=-1)
        gate = torch.sigmoid(self.write_gate(gate_input))
        new_cpu = gate * new_cpu + (1.0 - gate) * old_state

        B, N, D = new_cpu.shape
        new_cpu_flat = new_cpu.reshape(B * N, 1, D)
        new_cpu_flat = self.norm(new_cpu_flat)
        new_cpu = new_cpu_flat.reshape(B, N, D)

        cpu_summary = new_cpu.mean(dim=1)
        cpu_context = self.context_proj(cpu_summary)
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
