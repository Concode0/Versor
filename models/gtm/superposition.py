# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Superposition Search — simplified scoring via CPU grade norms.

Scores K instruction hypotheses using CPU state grade norms + ctrl_cursor,
dispatches trainable instruction templates (optionally modulated by rule memory)
to the PGA CPU, executes K outcomes in parallel, and selects via Gumbel-Softmax.

Mother algebra is no longer needed — scoring uses CPU grade norms directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from .cpu import GeometricCPU


class GeometricSuperpositionSearch(nn.Module):
    """Geometric Superposition Search over CPU Cl(3,0,1).

    Trainable parameters:
        instruction_templates: [K, 16] full Cl(3,0,1) multivectors
        score_mlp: CPU grade norms + ctrl_cursor -> K scores
        rule_proj: rule_memory -> per-template modulation (if rule_memory provided)
        log_temperature: Gumbel-Softmax temperature (learnable)
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

        D_cpu = algebra_cpu.dim  # 16

        # CPU engine (has ColorUnit params)
        self.pga_cpu = GeometricCPU(algebra_cpu, K_color)

        # Trainable instruction templates — full Cl(3,0,1) multivectors
        self.instruction_templates = nn.Parameter(
            torch.randn(num_hypotheses, D_cpu) * 0.01
        )

        # Scoring MLP: CPU grade norms + ctrl_cursor -> K scores
        cpu_grades = algebra_cpu.num_grades  # 5 for Cl(3,0,1)
        self.score_mlp = nn.Sequential(
            nn.Linear(cpu_grades + algebra_ctrl.dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_hypotheses),
        )

        # Rule-conditioned instruction modulation
        self.rule_proj = nn.Linear(D_cpu, num_hypotheses * D_cpu)

        # Gumbel temperature (learnable)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(torch.tensor(temperature_init).log()))
        )

    def step(self, cpu_state: torch.Tensor,
             ctrl_cursor: torch.Tensor,
             rule_memory: torch.Tensor = None) -> tuple:
        """One superposition search step.

        Args:
            cpu_state: [B, N, 16] CPU state in Cl(3,0,1).
            ctrl_cursor: [B, 4] control cursor in Cl(1,1).
            rule_memory: Optional [B, M, 16] rule slots from RuleAggregator.

        Returns:
            Tuple of (new_cpu_state [B, N, 16], search_info dict).
        """
        B, N, D_cpu = cpu_state.shape
        device = cpu_state.device
        K = self.num_hypotheses

        # STEP 1 — SCORE: CPU grade norms + ctrl_cursor
        cpu_summary = cpu_state.mean(dim=1)  # [B, 16]
        self.algebra_cpu.ensure_device(device)
        cpu_grade_norms = self.algebra_cpu.get_grade_norms(cpu_summary)  # [B, 5]
        score_input = torch.cat([cpu_grade_norms, ctrl_cursor], dim=-1)  # [B, 9]
        scores = self.score_mlp(score_input)  # [B, K]

        # STEP 2 — DISPATCH: templates optionally modulated by rule memory
        templates = self.instruction_templates.unsqueeze(0).expand(B, -1, -1)  # [B, K, 16]

        if rule_memory is not None:
            rule_summary = rule_memory.mean(dim=1)  # [B, 16]
            rule_features = self.rule_proj(rule_summary)  # [B, K * 16]
            rule_modulation = rule_features.view(B, K, D_cpu)  # [B, K, 16]
            templates = templates + rule_modulation

        # Score-dependent modulation
        instructions = scores.unsqueeze(-1) * templates  # [B, K, 16]

        # STEP 3 — EXECUTE: CPU applies PGA Motor + ColorUnit, K× batched
        outcomes = self.pga_cpu.execute_all(cpu_state, instructions)  # [B, K, N, 16]

        # STEP 4 — SELECT: Gumbel-Softmax, differentiable discrete selection
        tau = self.log_temperature.exp().clamp(0.1, 5.0)
        weights = F.gumbel_softmax(scores, tau=tau, hard=False)  # [B, K]

        # Weighted sum via einsum (no Python loop)
        new_cpu_state = torch.einsum('bk,bknd->bnd', weights, outcomes)

        search_info = {
            'scores': scores,
            'weights': weights,
            'temperature': tau.detach(),
        }

        return new_cpu_state, search_info
