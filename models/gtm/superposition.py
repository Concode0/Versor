# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Superposition Search: score, dispatch, execute, select."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from .cpu import GeometricCPU


class GeometricSuperpositionSearch(nn.Module):
    """Scores K hypotheses via CPU grade norms, executes PGA motor transforms
    in parallel, and selects via Gumbel-Softmax. Instruction templates are
    optionally modulated by rule memory.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra,
                 algebra_ctrl: CliffordAlgebra,
                 channels: int,
                 num_hypotheses: int = 4,
                 top_k: int = 1,
                 temperature_init: float = 1.0,
                 K_color: int = 4,
                 num_rule_slots: int = 8):
        super().__init__()
        self.algebra_cpu = algebra_cpu
        self.algebra_ctrl = algebra_ctrl
        self.channels = channels
        self.num_hypotheses = num_hypotheses
        self.top_k = top_k

        D_cpu = algebra_cpu.dim

        self.pga_cpu = GeometricCPU(algebra_cpu, K_color)
        self.instruction_templates = nn.Parameter(
            torch.randn(num_hypotheses, D_cpu) * 0.1
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(algebra_cpu.num_grades + algebra_ctrl.dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_hypotheses),
        )
        # Per-cell routing: each cell scores hypotheses independently
        self.cell_router = nn.Linear(D_cpu, num_hypotheses)
        # Rule memory bias on hypothesis scores
        self.rule_score_proj = nn.Linear(D_cpu, num_hypotheses)

        # Small-weight init so initial behavior ≈ old global-only scoring
        nn.init.normal_(self.cell_router.weight, std=0.01)
        nn.init.zeros_(self.cell_router.bias)
        nn.init.normal_(self.rule_score_proj.weight, std=0.01)
        nn.init.zeros_(self.rule_score_proj.bias)

        self.rule_proj = nn.Linear(D_cpu, num_hypotheses * D_cpu)
        self.register_buffer('_temperature', torch.tensor(float(temperature_init)))

    def set_temperature(self, tau: float):
        """Set Gumbel-Softmax temperature (called by external annealing schedule)."""
        self._temperature.fill_(tau)

    def step(self, cpu_state: torch.Tensor,
             ctrl_cursor: torch.Tensor,
             rule_memory: torch.Tensor = None) -> tuple:
        """One search step. Returns (new_cpu_state, search_info)."""
        B, N, D_cpu = cpu_state.shape
        device = cpu_state.device
        K = self.num_hypotheses

        cpu_summary = cpu_state.mean(dim=1)
        self.algebra_cpu.ensure_device(device)
        grade_norms = self.algebra_cpu.get_grade_norms(cpu_summary)

        # Per-cell logits + global bias from cursor/grade norms
        cell_logits = self.cell_router(cpu_state)                          # [B, N, K]
        global_bias = self.score_mlp(
            torch.cat([grade_norms, ctrl_cursor], dim=-1)
        )                                                                  # [B, K]
        scores = cell_logits + global_bias.unsqueeze(1)                    # [B, N, K]

        templates = self.instruction_templates.unsqueeze(0).expand(B, -1, -1)
        if rule_memory is not None:
            rule_summary = rule_memory.mean(dim=1)
            rule_modulation = self.rule_proj(rule_summary).view(B, K, D_cpu)
            templates = templates + rule_modulation
            # Rule memory biases scoring (which instructions cells prefer)
            rule_score_bias = self.rule_score_proj(rule_summary)           # [B, K]
            scores = scores + rule_score_bias.unsqueeze(1)                 # [B, N, K]

        outcomes = self.pga_cpu.execute_all(cpu_state, templates)          # [B, K, N, D]

        tau = self._temperature.clamp(0.1, 5.0)
        weights = F.gumbel_softmax(
            scores.reshape(B * N, K), tau=tau, hard=False
        ).reshape(B, N, K)                                                 # [B, N, K]
        new_cpu_state = torch.einsum('bnk,bknd->bnd', weights, outcomes)

        return new_cpu_state, {
            'scores': scores,
            'weights': weights,
            'temperature': tau.detach(),
        }
