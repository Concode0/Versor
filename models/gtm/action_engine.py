# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""ActionEngine: generalized PGA Cl(3,0,1) action proposer.

Combines continuous motor transforms (rotation + translation via sandwich
product) with discrete color operations (DiscreteActionHead). A learnable
per-component gate controls the blend between paths. Proposes K candidate
states for all hypotheses in parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra


class DiscreteActionHead(nn.Module):
    """Discrete color update conditioned on spatial context.

    Predicts a 10-class color distribution and converts to grade-0 via
    differentiable soft-argmax.  This gives much richer color control than
    a single additive delta, since the model can target any of 10 colors
    in one shot.

    Operates on grade-0 (motor-invariant) using grade-1 spatial features
    as context. Bypasses the motor sandwich invariance limitation by
    directly setting the scalar component.
    """

    # Grade-1 spatial indices in Cl(3,0,1): e0(1), e1(2), e2(4), e3(8)
    _SPATIAL_IDX = [1, 2, 4, 8]
    _NUM_COLORS = 10

    def __init__(self):
        super().__init__()
        self.spatial_proj = nn.Linear(4, 32)
        self.color_mlp = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, self._NUM_COLORS),
        )
        # Anchors: 10 evenly-spaced grade-0 values for each color
        self.register_buffer(
            '_color_anchors',
            torch.arange(self._NUM_COLORS, dtype=torch.float32) / (self._NUM_COLORS - 1),
        )

    def forward(self, state: torch.Tensor,
                instr: torch.Tensor) -> torch.Tensor:
        """Apply discrete color update via soft color selection.

        Args:
            state: Cell states [L, N, 16].
            instr: Instructions [L, 16].

        Returns:
            Updated state [L, N, 16] with modified grade-0.
        """
        L, N, D = state.shape
        spatial = state[:, :, self._SPATIAL_IDX]  # [L, N, 4]
        feat = F.relu(self.spatial_proj(spatial))  # [L, N, 32]
        ctx = torch.cat([feat, instr.unsqueeze(1).expand(-1, N, -1)], dim=-1)
        color_logits = self.color_mlp(ctx)  # [L, N, 10]

        # Soft color selection: differentiable weighted sum of anchor values
        color_probs = F.softmax(color_logits, dim=-1)  # [L, N, 10]
        new_color = (color_probs * self._color_anchors).sum(dim=-1)  # [L, N]

        out = state.clone()
        out[:, :, 0] = new_color
        return out


class ActionEngine(nn.Module):
    """Generalized action proposer combining continuous motors and discrete updates.

    For each of K hypotheses, modulates instruction templates with rule memory,
    applies both continuous motor transforms and discrete color updates, then
    blends via a learnable per-component gate.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra,
                 num_hypotheses: int = 8,
                 gate_init: float = 0.0):
        super().__init__()
        assert algebra_cpu.p == 3 and algebra_cpu.r == 1, \
            f"ActionEngine requires Cl(3,0,1), got Cl({algebra_cpu.p},{algebra_cpu.q},{algebra_cpu.r})"

        self.algebra = algebra_cpu
        D = algebra_cpu.dim  # 16
        K = num_hypotheses
        self.num_hypotheses = K

        # Instruction templates
        self.instruction_templates = nn.Parameter(torch.randn(K, D) * 0.1)

        # Discrete action head
        self.discrete_head = DiscreteActionHead()

        # Per-component gate: sigmoid(gate_init=0) = 0.5 balanced start
        # Grade-0 biased toward discrete (motor can't change scalars)
        gate_vals = torch.full((D,), gate_init)
        gate_vals[0] = -2.0  # sigmoid(-2) ≈ 0.12 → mostly discrete for color
        self.action_gate = nn.Parameter(gate_vals)

        # Rule memory modulation
        self.rule_proj = nn.Linear(D, K * D)

    def _motor_transform(self, state: torch.Tensor,
                         instruction: torch.Tensor) -> torch.Tensor:
        """Apply motor sandwich product: R x R~ where R = exp(-B/2).

        Pure geometric transform — no color remapping. The discrete head
        handles color updates separately.

        Args:
            state: [L, N, D]
            instruction: [L, D]

        Returns:
            Transformed state [L, N, D].
        """
        L, N, D = state.shape
        self.algebra.ensure_device(state.device)

        bv = self.algebra.grade_projection(instruction, 2)
        M = self.algebra.exp(-0.5 * bv)
        M_rev = self.algebra.reverse(M)

        M_exp = M.unsqueeze(1).expand(L, N, D).reshape(L * N, D)
        M_rev_exp = M_rev.unsqueeze(1).expand(L, N, D).reshape(L * N, D)
        state_flat = state.reshape(L * N, 1, D)

        out = self.algebra.sandwich_product(
            M_exp, state_flat, M_rev_exp,
        ).reshape(L, N, D)

        return out

    def propose_all(self, state: torch.Tensor,
                    hypotheses: torch.Tensor,
                    rule_memory: torch.Tensor = None) -> torch.Tensor:
        """Propose K candidate states for all hypotheses.

        Args:
            state: Attended cell states [B, N, D].
            hypotheses: Current hypothesis states [B, K, 4] (reserved for
                future hypothesis-conditioned actions).
            rule_memory: Optional rule slots [B, M, D].

        Returns:
            Candidate states [B, K, N, D].
        """
        B, N, D = state.shape
        K = self.num_hypotheses
        self.algebra.ensure_device(state.device)

        templates = self.instruction_templates.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        if rule_memory is not None:
            rule_mod = self.rule_proj(rule_memory.mean(dim=1)).view(B, K, D)
            templates = templates + rule_mod

        # Batch all K hypotheses: [B*K, N, D]
        state_exp = state.unsqueeze(1).expand(B, K, N, D).reshape(B * K, N, D)
        instr_flat = templates.reshape(B * K, D)

        # Continuous motor transform (no ColorUnit — pure geometric)
        continuous = self._motor_transform(state_exp, instr_flat)  # [B*K, N, D]

        # Discrete color update
        discrete = self.discrete_head(state_exp, instr_flat)  # [B*K, N, D]

        # Blend via per-component gate
        gate = torch.sigmoid(self.action_gate)  # [D]
        result = gate * continuous + (1.0 - gate) * discrete

        return result.reshape(B, K, N, D)

    def get_combined_rotor(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted combination of rotors from instruction templates.

        Uses weighted bivector averaging in Lie algebra (log-space) then
        a single exp map, which is more numerically stable than weighting
        post-exp rotors.

        Args:
            weights: Hypothesis attention weights [B, K].

        Returns:
            Combined rotor [B, D].
        """
        self.algebra.ensure_device(weights.device)
        # Weighted sum of bivectors (Lie algebra averaging)
        bv = self.algebra.grade_projection(self.instruction_templates, 2)  # [K, D]
        combined_bv = torch.einsum('bk,kd->bd', weights, bv)  # [B, D]
        # Single exp map from the averaged bivector
        return self.algebra.exp(-0.5 * combined_bv)
