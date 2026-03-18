# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Grade-aware projections between LLM hidden states and Clifford multivectors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.primitives.normalization import CliffordLayerNorm


class GradeAwareProjectionIn(nn.Module):
    """Project LLM hidden states into grade-1 multivectors.

    Linear(llm_dim -> channels * g1_dim) then scatter into the grade-1
    subspace of the full multivector, followed by CliffordLayerNorm.
    """

    def __init__(self, algebra, llm_dim, channels):
        super().__init__()
        g1_mask = algebra.grade_masks[1]
        g1_idx = g1_mask.nonzero(as_tuple=False).squeeze(-1)
        self.register_buffer('g1_idx', g1_idx)
        self.g1_dim = len(g1_idx)
        self.channels = channels
        self.algebra_dim = algebra.dim
        self.linear = nn.Linear(llm_dim, channels * self.g1_dim)
        self.norm = CliffordLayerNorm(algebra, channels)

    def forward(self, x):
        B, L, _ = x.shape
        proj = self.linear(x)  # [B, L, C*g1_dim]
        proj = proj.reshape(B, L, self.channels, self.g1_dim)
        mv = torch.zeros(B, L, self.channels, self.algebra_dim,
                         device=x.device, dtype=x.dtype)
        g1_idx = self.g1_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        g1_idx = g1_idx.expand(B, L, self.channels, -1)
        mv.scatter_(3, g1_idx, proj)
        mv_flat = mv.reshape(B * L, self.channels, self.algebra_dim)
        mv_flat = self.norm(mv_flat)
        return mv_flat.reshape(B, L, self.channels, self.algebra_dim)


class GradeWeightedProjectionOut(nn.Module):
    """Project multivectors back to LLM dim with learned grade weights.

    Per-grade linear projections weighted by a learnable softmax over grades.
    """

    def __init__(self, algebra, channels, llm_dim, task_type='generic'):
        super().__init__()
        self.channels = channels
        self.llm_dim = llm_dim
        self.num_grades = algebra.num_grades
        self.grade_weights = nn.Parameter(torch.zeros(algebra.num_grades))
        self.grade_projections = nn.ModuleList()
        for g in range(algebra.num_grades):
            g_mask = algebra.grade_masks[g]
            g_dim = int(g_mask.sum().item())
            g_idx = g_mask.nonzero(as_tuple=False).squeeze(-1)
            self.register_buffer(f'_grade_idx_{g}', g_idx)
            self.grade_projections.append(
                nn.Linear(channels * g_dim, llm_dim)
            )
        self.layer_norm = nn.LayerNorm(llm_dim)

    def forward(self, mv):
        B, L, C, D = mv.shape
        weights = F.softmax(self.grade_weights, dim=0)
        out = torch.zeros(B, L, self.llm_dim, device=mv.device, dtype=mv.dtype)
        for g in range(self.num_grades):
            idx = getattr(self, f'_grade_idx_{g}')
            g_vals = mv[..., idx]  # [B, L, C, g_dim]
            g_flat = g_vals.reshape(B, L, -1)
            out = out + weights[g] * self.grade_projections[g](g_flat)
        return self.layer_norm(out)
