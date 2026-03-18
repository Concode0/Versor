# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Rule Memory Bank — cross-attention aggregator for demo→test information flow.

Compresses Phase 1 (demo) CPU state into M learnable rule slots via
cross-attention. This replaces the 4-float ctrl_cursor bottleneck as the
primary information bridge between demo and test phases.

Information capacity: M=8 slots * 16 dims = 128 floats (vs 4 floats before).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleAggregator(nn.Module):
    """Cross-attention from M learnable queries to demo cpu_state.

    Compresses Phase 1 output into M rule slots that encode the
    transformation rule learned from demo pairs.
    """

    def __init__(self, d_cpu: int = 16, num_slots: int = 8, num_heads: int = 4):
        super().__init__()
        self.d_cpu = d_cpu
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = d_cpu // num_heads
        assert d_cpu % num_heads == 0, f"d_cpu={d_cpu} must be divisible by num_heads={num_heads}"

        self.scale = self.head_dim ** -0.5

        # Learnable query templates
        self.query_templates = nn.Parameter(torch.randn(num_slots, d_cpu) * 0.02)

        # Projections for cross-attention
        self.q_proj = nn.Linear(d_cpu, d_cpu)
        self.k_proj = nn.Linear(d_cpu, d_cpu)
        # V = raw demo state (no projection — preserves geometric structure)

    def forward(self, demo_cpu_state: torch.Tensor,
                demo_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate demo state into rule memory slots.

        Args:
            demo_cpu_state: [B, N_demo, 16] CPU state after Phase 1.
            demo_mask: [B, N_demo] bool (True=valid).

        Returns:
            rule_memory: [B, M, 16] compressed rule representation.
        """
        B, N_demo, D = demo_cpu_state.shape
        M = self.num_slots
        H = self.num_heads
        hd = self.head_dim

        # Query from learnable templates: [M, D] -> [B, M, D]
        Q = self.q_proj(self.query_templates).unsqueeze(0).expand(B, -1, -1)
        # Key from demo state: [B, N_demo, D]
        K = self.k_proj(demo_cpu_state)

        # Multi-head reshape
        Q = Q.reshape(B, M, H, hd).transpose(1, 2)      # [B, H, M, hd]
        K = K.reshape(B, N_demo, H, hd).transpose(1, 2)  # [B, H, N_demo, hd]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, M, N_demo]

        # Mask invalid demo cells
        if demo_mask is not None:
            pad_mask = ~demo_mask  # [B, N_demo]
            scores = scores.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(scores, dim=-1)  # [B, H, M, N_demo]
        attn_avg = attn.mean(dim=1)       # [B, M, N_demo]

        # Values: raw demo state (preserves geometric structure)
        rule_memory = torch.bmm(attn_avg, demo_cpu_state)  # [B, M, 16]

        return rule_memory
