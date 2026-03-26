# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Rule memory bank: cross-attention aggregator for demo-to-test information flow."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleAggregator(nn.Module):
    """Compresses demo CPU state into M rule slots via cross-attention.

    Learnable query templates attend over demo cells; values are raw
    multivectors to preserve geometric structure.
    """

    def __init__(self, d_cpu: int = 16, num_slots: int = 8, num_heads: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = d_cpu // num_heads
        assert d_cpu % num_heads == 0

        self.scale = self.head_dim ** -0.5
        self.query_templates = nn.Parameter(torch.randn(num_slots, d_cpu) * 0.02)
        self.q_proj = nn.Linear(d_cpu, d_cpu)
        self.k_proj = nn.Linear(d_cpu, d_cpu)

    def forward(self, demo_cpu_state: torch.Tensor,
                demo_mask: torch.Tensor) -> torch.Tensor:
        """[B, N_demo, 16] -> [B, M, 16] rule memory slots."""
        B, N_demo, D = demo_cpu_state.shape
        M, H, hd = self.num_slots, self.num_heads, self.head_dim

        Q = self.q_proj(self.query_templates).unsqueeze(0).expand(B, -1, -1)
        K = self.k_proj(demo_cpu_state)

        Q = Q.reshape(B, M, H, hd).transpose(1, 2)
        K = K.reshape(B, N_demo, H, hd).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if demo_mask is not None:
            scores = scores.masked_fill(
                (~demo_mask).unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = F.softmax(scores, dim=-1).mean(dim=1)
        return torch.bmm(attn, demo_cpu_state)
