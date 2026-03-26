# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""WorldModel + WorldModelStep + CellAttention: the core computation loop.

CellAttention provides spatial context via cross-grade self-attention.
WorldModelStep chains attention, action proposals, search, modulation,
gating, and rotor accumulation. WorldModel wraps the step loop with
log-manifold stability and FIM-based adaptive halt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from layers.primitives.normalization import CliffordLayerNorm
from .search_plane import AlgebraicProjection, AlgebraicLift, SearchPlane
from .action_engine import ActionEngine
from .info_geometry import FIMEvaluator
from .log_manifold import LogManifoldProjector
from .adaptive_halt import FIMAdaptiveHalt


_GRADE_MAP_16 = torch.zeros(16, dtype=torch.long)
_GRADE_MAP_16[0] = 0
_GRADE_MAP_16[[1, 2, 4, 8]] = 1
_GRADE_MAP_16[[3, 5, 6, 9, 10, 12]] = 2
_GRADE_MAP_16[[7, 11, 13, 14]] = 3
_GRADE_MAP_16[15] = 4


class CellAttention(nn.Module):
    """Cross-grade self-attention over grid cells in Cl(3,0,1).

    Multi-head attention where Q/K are projected from the full multivector
    and values are the raw multivector with per-grade learnable gains.
    """

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
                (~mask).unsqueeze(1).unsqueeze(2), float('-inf'),
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_avg = attn.mean(dim=1)

        attended = torch.bmm(attn_avg, x)
        return self._apply_grade_gains(attended)


def _identity_rotor(batch_size: int, dim: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    """Create identity rotor (scalar=1, rest=0)."""
    R = torch.zeros(batch_size, dim, device=device, dtype=dtype)
    R[:, 0] = 1.0
    return R


def _normalize_rotor(algebra: CliffordAlgebra, R: torch.Tensor) -> torch.Tensor:
    """Normalize rotor to stay on the Spin group: R / sqrt(|R R~|_0)."""
    R_rev = algebra.reverse(R)
    sq = algebra.geometric_product(R, R_rev)[..., 0:1].abs().clamp(min=1e-6)
    return R / sq.sqrt()


class WorldModelStep(nn.Module):
    """One step of the World Model computation loop.

    Chains cell attention, action proposals, FIM scoring, search-plane
    evolution, weighted candidate selection, grade-wise modulation,
    gated residual update, and rotor accumulation.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra,
                 algebra_ctrl: CliffordAlgebra,
                 num_hypotheses: int = 8,
                 num_attn_heads: int = 4,
                 attn_head_dim: int = 8,
                 num_rule_slots: int = 8,
                 evolve_hidden: int = 64,
                 gate_init: float = 0.0,
                 use_supervised_fim: bool = True):
        super().__init__()
        D = algebra_cpu.dim  # 16
        self._algebra_cpu = algebra_cpu

        # Attention
        self.cell_attn = CellAttention(
            algebra_cpu, num_attn_heads, attn_head_dim,
        )

        # Algebraic projection/lift
        self.phi = AlgebraicProjection(algebra_cpu)
        self.psi = AlgebraicLift(algebra_cpu)

        # Search plane
        self.search_plane = SearchPlane(
            algebra_ctrl, num_hypotheses, evolve_hidden,
        )

        # Action engine
        self.action_engine = ActionEngine(
            algebra_cpu, num_hypotheses, gate_init,
        )

        # FIM evaluator
        self.fim_evaluator = FIMEvaluator(algebra_cpu)
        self.use_supervised_fim = use_supervised_fim

        self.write_gate = nn.Sequential(
            nn.Linear(D * 2, 64),
            nn.ReLU(),
            nn.Linear(64, D),
        )
        nn.init.constant_(self.write_gate[-1].bias, -3.0)

        self.color_write_gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.constant_(self.color_write_gate[-1].bias, -3.0)

        # Normalization
        self.norm = CliffordLayerNorm(algebra_cpu, 1)

    def set_temperature(self, tau: float):
        self.search_plane.set_temperature(tau)

    def forward(self, state: torch.Tensor, hypotheses: torch.Tensor,
                R_accum: torch.Tensor, mask: torch.Tensor = None,
                rule_memory: torch.Tensor = None,
                fim_prev: torch.Tensor = None,
                targets: torch.Tensor = None,
                step_idx: int = 0, total_steps: int = 12,
                gradient_horizon: int = 2) -> dict:
        """Execute one world model step.

        Args:
            state: Current state (mantissa) [B, N, D].
            hypotheses: Current hypotheses [B, K, 4].
            R_accum: Accumulated rotor [B, D].
            mask: Validity mask [B, N].
            rule_memory: Optional rule slots [B, M, D].
            fim_prev: Previous FIM values [B, K] or None.
            targets: Optional target colors [B, N] for supervised FIM.
            step_idx: Current step index in the loop.
            total_steps: Total number of steps.
            gradient_horizon: Allow gradient for last N steps.

        Returns:
            dict with world_state, hypotheses, R_accum, fim_values, search_info, gate.
        """
        B, N, D = state.shape
        old = state

        attended = self.cell_attn(state, mask)  # [B, N, D]

        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)  # [B, N, 1]
            world_summary = (attended * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            world_summary = attended.mean(dim=1)  # [B, D]

        candidates = self.action_engine.propose_all(
            attended, hypotheses, rule_memory,
        )  # [B, K, N, D]

        if self.training and self.use_supervised_fim and targets is not None:
            fim_values = self.fim_evaluator.supervised_fim(candidates, targets, mask)
        else:
            fim_values = self.fim_evaluator.fim_proxy(candidates, mask)

        search_result = self.search_plane(
            hypotheses, world_summary, fim_values, fim_prev,
        )
        weights = search_result['weights']  # [B, K]
        hypotheses = search_result['hypotheses']

        new_state = torch.einsum('bk,bknd->bnd', weights, candidates)  # [B, N, D]

        modulation = self.psi(hypotheses, weights)  # [B, D]
        new_state = new_state * modulation.unsqueeze(1)

        gate_input = torch.cat([old, new_state], dim=-1)
        gate = torch.sigmoid(self.write_gate(gate_input))  # [B, N, D]

        # Separate color gate for grade-0 — avoids clone() by computing
        # the residual blend per-component before combining
        color_gate_in = torch.stack([old[:, :, 0], new_state[:, :, 0]], dim=-1)
        color_gate = torch.sigmoid(self.color_write_gate(color_gate_in))  # [B, N, 1]

        new_state_spatial = gate[:, :, 1:] * new_state[:, :, 1:] + (1.0 - gate[:, :, 1:]) * old[:, :, 1:]
        new_state_color = color_gate * new_state[:, :, 0:1] + (1.0 - color_gate) * old[:, :, 0:1]
        new_state = torch.cat([new_state_color, new_state_spatial], dim=-1)

        new_state = self.norm(
            new_state.reshape(B * N, 1, D),
        ).reshape(B, N, D)

        # R_accum tracks the accumulated rotor for diagnostics but is not
        # in the loss path, so always detach to save memory.
        self._algebra_cpu.ensure_device(state.device)
        R_t = self.action_engine.get_combined_rotor(weights)  # [B, D]
        R_accum = self._algebra_cpu.geometric_product(R_t, R_accum.detach())
        R_accum = _normalize_rotor(self._algebra_cpu, R_accum)

        return {
            'world_state': new_state,
            'hypotheses': hypotheses,
            'R_accum': R_accum,
            'fim_values': search_result['fim_values'],
            'search_info': search_result,
            'gate': gate,
        }


class WorldModel(nn.Module):
    """Main World Model: chains WorldModelSteps with log-manifold stability.

    Splits input into mantissa/exponent, processes mantissa through T steps,
    then merges back. Supports FIM-based adaptive halt at inference and
    FIM-weighted mixing at training (Phase 3).
    """

    def __init__(self, algebra_cpu: CliffordAlgebra,
                 algebra_ctrl: CliffordAlgebra,
                 num_steps: int = 12,
                 max_steps: int = 24,
                 num_hypotheses: int = 8,
                 num_attn_heads: int = 4,
                 attn_head_dim: int = 8,
                 num_rule_slots: int = 8,
                 evolve_hidden: int = 64,
                 gate_init: float = 0.0,
                 log_manifold_gate_init: float = -5.0,
                 halt_eps: float = 0.01,
                 use_supervised_fim: bool = True,
                 weight_share_steps: bool = False,
                 gradient_horizon: int = 2):
        super().__init__()
        self.num_steps = num_steps
        self.max_steps = max_steps
        self.num_hypotheses = num_hypotheses
        self.gradient_horizon = gradient_horizon
        self._algebra_cpu = algebra_cpu

        D = algebra_cpu.dim

        # Log-manifold projector
        self.log_projector = LogManifoldProjector(algebra_cpu, log_manifold_gate_init)

        # World model steps
        if weight_share_steps:
            shared_step = WorldModelStep(
                algebra_cpu, algebra_ctrl,
                num_hypotheses, num_attn_heads, attn_head_dim,
                num_rule_slots, evolve_hidden, gate_init,
                use_supervised_fim,
            )
            self.steps = nn.ModuleList([shared_step] * num_steps)
        else:
            self.steps = nn.ModuleList([
                WorldModelStep(
                    algebra_cpu, algebra_ctrl,
                    num_hypotheses, num_attn_heads, attn_head_dim,
                    num_rule_slots, evolve_hidden, gate_init,
                    use_supervised_fim,
                )
                for _ in range(num_steps)
            ])

        # Initial hypotheses (base, shared across problems)
        self.hypothesis_init = nn.Parameter(torch.randn(num_hypotheses, 4) * 0.1)

        # Demo-conditioned hypothesis offset: rule_memory -> per-problem initial hypotheses
        # Creates different starting hypotheses for different problems, enabling
        # per-problem adaptation instead of fitting one global average.
        # Zero-initialized so it starts as identity (no offset at init).
        self.hypothesis_projector = nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, num_hypotheses * 4),
        )
        nn.init.zeros_(self.hypothesis_projector[-1].weight)
        nn.init.zeros_(self.hypothesis_projector[-1].bias)

        # FIM adaptive halt
        self.fim_halt = FIMAdaptiveHalt(halt_eps)
        self.use_fim_halt = False
        # Ramp for gradual FIM mixing blend (0=last-step only, 1=full FIM mix)
        self.fim_mix_ramp = 0.0

        # Gated exponent update: enables magnitude learning during the step loop
        self.exponent_update = nn.Sequential(
            nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.exponent_update[-1].weight)
        nn.init.zeros_(self.exponent_update[-1].bias)
        self.exponent_gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5)≈0.007

        # Final norm
        self.final_norm = CliffordLayerNorm(algebra_cpu, 1)

    def set_temperature(self, tau: float):
        for step in self.steps:
            step.set_temperature(tau)

    def forward(self, cpu_state: torch.Tensor,
                mask: torch.Tensor = None,
                rule_memory: torch.Tensor = None,
                targets: torch.Tensor = None,
                return_trace: bool = False) -> dict:
        """Execute the world model loop.

        Args:
            cpu_state: Initial CPU state [B, N, D] from GridCodec.
            mask: Validity mask [B, N].
            rule_memory: Optional rule slots [B, M, D].
            targets: Optional target colors [B, N] for supervised FIM.
            return_trace: Collect per-step diagnostics.

        Returns:
            dict with output, hypotheses, R_accum, step_outputs, etc.
        """
        B, N, D = cpu_state.shape
        device = cpu_state.device
        dtype = cpu_state.dtype

        # Split into mantissa + exponent
        mantissa, exponent = self.log_projector.split(cpu_state)

        # Initialize hypotheses — conditioned on rule_memory if available
        K = self.num_hypotheses
        if rule_memory is not None:
            rule_ctx = rule_memory.mean(dim=1)  # [B, D]
            h_offset = self.hypothesis_projector(rule_ctx).view(B, K, 4)
            hypotheses = self.hypothesis_init.unsqueeze(0).expand(B, -1, -1) + h_offset
        else:
            hypotheses = self.hypothesis_init.unsqueeze(0).expand(B, -1, -1).clone()
        R_accum = _identity_rotor(B, D, device, dtype)
        fim_prev = None

        step_outputs = []
        step_deltas = []
        step_weights = []
        trace = {
            'search_info': [],
            'gate_values': [],
            'fim_values': [],
        } if return_trace else None

        num_active_steps = len(self.steps)
        exp_gate = torch.sigmoid(self.exponent_gate)

        for t, step in enumerate(self.steps):
            result = step(
                mantissa, hypotheses, R_accum, mask,
                rule_memory, fim_prev, targets,
                step_idx=t, total_steps=num_active_steps,
                gradient_horizon=self.gradient_horizon,
            )
            mantissa = result['world_state']
            hypotheses = result['hypotheses']
            R_accum = result['R_accum']
            fim_prev = result['fim_values']

            # Gated exponent update: enables magnitude learning per step
            # tanh bounds delta to [-1, 1] — max exponent shift per step is ~0.007
            exp_delta = torch.tanh(self.exponent_update(mantissa.mean(dim=1, keepdim=True)))
            exponent = exponent + exp_gate * exp_delta

            step_outputs.append(mantissa)

            search_info = result['search_info']
            step_deltas.append(search_info['delta_info'])
            step_weights.append(search_info['weights'])

            if trace is not None:
                trace['search_info'].append({
                    k: v.detach() if torch.is_tensor(v) else v
                    for k, v in search_info.items()
                })
                trace['gate_values'].append(result['gate'].detach())
                trace['fim_values'].append(result['fim_values'].detach())

            # FIM-based halt (inference only)
            if not self.training and t > 0:
                delta = search_info['delta_info']
                w = search_info['weights']
                weighted_delta = (delta * w).sum(dim=-1).mean()
                if weighted_delta < self.fim_halt.halt_eps:
                    break

        # Merge mantissa with exponent
        output = self.log_projector.merge(mantissa, exponent)

        # Final norm
        output = self.final_norm(
            output.reshape(B * N, 1, D),
        ).reshape(B, N, D)

        # FIM-weighted mixing during training: gradually blend between
        # last-step output and FIM-weighted mix using fim_mix_ramp (0->1)
        mixing_weights = None
        if self.training and self.use_fim_halt and len(step_deltas) > 1:
            halt_result = self.fim_halt(step_deltas, step_weights)
            mixing_weights = halt_result['mixing_weights']  # [B, T]

            stacked = torch.stack(step_outputs, dim=1)  # [B, T, N, D]
            mixed_mantissa = torch.einsum('bt,btnd->bnd', mixing_weights, stacked)

            # Blend: ramp=0 uses last-step, ramp=1 uses full FIM mix
            ramp = self.fim_mix_ramp
            blended_mantissa = (1.0 - ramp) * mantissa + ramp * mixed_mantissa

            output = self.log_projector.merge(blended_mantissa, exponent)
            output = self.final_norm(
                output.reshape(B * N, 1, D),
            ).reshape(B, N, D)

        return {
            'output': output,
            'hypotheses': hypotheses,
            'R_accum': R_accum,
            'step_outputs': step_outputs,
            'step_deltas': step_deltas,
            'step_weights': step_weights,
            'mixing_weights': mixing_weights,
            'trace': trace,
        }
