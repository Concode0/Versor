# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Turing Machine execution engine — ARC-AGI v4.

Chains TuringSteps with dual-state (cpu_state + ctrl_cursor) threading.
Supports both fixed-step and adaptive computation (PonderNet) modes.
Optionally threads rule_memory from Phase 1 to each step.
"""

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers.primitives.normalization import CliffordLayerNorm
from .turing_step import TuringStep
from .adaptive_halt import AdaptiveHalt


class TuringVM(nn.Module):
    """Geometric Turing Machine execution engine.

    Chains N TuringSteps with dual-state (cpu_state + ctrl_cursor).
    Supports both fixed-step and adaptive computation modes.
    Threads rule_memory to each step when provided.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra,
                 algebra_ctrl: CliffordAlgebra,
                 channels: int,
                 num_steps: int = 8,
                 max_steps: int = 20,
                 num_hypotheses: int = 4,
                 top_k: int = 1,
                 temperature_init: float = 1.0,
                 use_act: bool = False,
                 lambda_p: float = 0.5,
                 num_attn_heads: int = 4,
                 attn_head_dim: int = 8,
                 K_color: int = 4,
                 num_rule_slots: int = 8):
        super().__init__()
        self.channels = channels
        self.num_steps = num_steps
        self.max_steps = max_steps
        self.use_act = use_act

        # Create steps up to max_steps (ACT) or num_steps (fixed)
        effective_steps = max_steps if use_act else num_steps
        self.steps = nn.ModuleList([
            TuringStep(
                algebra_cpu, algebra_ctrl,
                channels, num_hypotheses, top_k, temperature_init,
                num_attn_heads, attn_head_dim, 0.0,
                K_color, num_rule_slots,
            )
            for _ in range(effective_steps)
        ])

        # Adaptive halt controller
        self.adaptive_halt = AdaptiveHalt(lambda_p, max_steps) if use_act else None

        # Final normalization on CPU state
        self.final_norm = CliffordLayerNorm(algebra_cpu, 1)

    def forward(self, cpu_state: torch.Tensor, ctrl_cursor: torch.Tensor,
                mask: torch.Tensor = None,
                return_trace: bool = False,
                rule_memory: torch.Tensor = None) -> tuple:
        """Execute the GTM program.

        Args:
            cpu_state: Initial CPU state [B, N, 16].
            ctrl_cursor: Initial control cursor [B, 4].
            mask: Optional validity mask [B, N] (True=valid).
            return_trace: If True, collect per-step diagnostics.
            rule_memory: Optional [B, M, 16] rule slots from RuleAggregator.

        Returns:
            Tuple of (cpu_state, ctrl_cursor, act_info or None, trace or None).
        """
        trace = {
            'search_scores': [],
            'search_weights': [],
            'halt_probs': [],
            'cursors': [],
            'gate_values': [],
        } if return_trace else None

        if self.use_act:
            return self._forward_act(cpu_state, ctrl_cursor, mask, trace, rule_memory)
        else:
            return self._forward_fixed(cpu_state, ctrl_cursor, mask, trace, rule_memory)

    def _forward_fixed(self, cpu_state, ctrl_cursor, mask, trace, rule_memory):
        """Fixed-step execution."""
        for i in range(self.num_steps):
            result = self.steps[i](cpu_state, ctrl_cursor, mask, rule_memory)
            cpu_state = result['cpu_state']
            ctrl_cursor = result['ctrl_cursor']

            if trace is not None:
                trace['search_scores'].append(result['search_info']['scores'].detach())
                trace['search_weights'].append(result['search_info']['weights'].detach())
                trace['halt_probs'].append(result['halt_prob'].detach())
                trace['cursors'].append(ctrl_cursor.detach())
                trace['gate_values'].append(result['gate_values'].detach())

        # Final norm
        B, N, D = cpu_state.shape
        cpu_state = self.final_norm(
            cpu_state.reshape(B * N, 1, D)
        ).reshape(B, N, D)

        return cpu_state, ctrl_cursor, None, trace

    def _forward_act(self, cpu_state, ctrl_cursor, mask, trace, rule_memory):
        """Adaptive computation with PonderNet halting."""
        per_step_outputs = []
        halt_probs = []

        for i, step in enumerate(self.steps):
            result = step(cpu_state, ctrl_cursor, mask, rule_memory)
            cpu_state = result['cpu_state']
            ctrl_cursor = result['ctrl_cursor']

            per_step_outputs.append(cpu_state)
            halt_probs.append(result['halt_prob'])

            if trace is not None:
                trace['search_scores'].append(result['search_info']['scores'].detach())
                trace['search_weights'].append(result['search_info']['weights'].detach())
                trace['halt_probs'].append(result['halt_prob'].detach())
                trace['cursors'].append(ctrl_cursor.detach())
                trace['gate_values'].append(result['gate_values'].detach())

        # Compute ACT mixing weights
        act_result = self.adaptive_halt(halt_probs)
        weights = act_result['weights']  # [B, T]

        # Weighted sum of per-step CPU states via einsum (no Python loop)
        stacked = torch.stack(per_step_outputs, dim=1)  # [B, T, N, D]
        output = torch.einsum('bt,btnd->bnd', weights, stacked)

        # Final norm
        B, N, D = output.shape
        output = self.final_norm(
            output.reshape(B * N, 1, D)
        ).reshape(B, N, D)

        act_info = {
            'kl_loss': act_result['kl_loss'],
            'expected_steps': act_result['expected_steps'],
            'weights': act_result['weights'],
        }

        # ctrl_cursor is from last step (not mixed — control is sequential)
        return output, ctrl_cursor, act_info, trace
