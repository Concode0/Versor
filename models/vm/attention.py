# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Grade-masked multihead attention for the Geometric Turing Machine.

Operates on grade-1 + grade-2 components (the "heap") of Clifford multivectors,
leaving other grades untouched. This focuses attention on the most computation-
relevant subspaces: vectors (data bus) and bivectors (instructions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradeMaskedAttention(nn.Module):
    """Grade-masked multihead attention on Clifford multivectors.

    Extracts grade-1 + grade-2 basis elements, applies standard MHA,
    then scatters the result back into the full multivector.
    """

    def __init__(self, algebra, channels, num_heads=4, dropout=0.0):
        super().__init__()
        # Select grade-1 + grade-2 basis elements
        g1_mask = algebra.grade_masks[1]
        g2_mask = algebra.grade_masks[2]
        heap_mask = g1_mask | g2_mask
        heap_idx = heap_mask.nonzero(as_tuple=False).squeeze(-1)
        self.register_buffer('heap_idx', heap_idx)
        self.heap_dim = len(heap_idx)
        self.channels = channels
        self.num_heads = num_heads
        self.algebra_dim = algebra.dim

        proj_dim = channels * self.heap_dim
        self.q_proj = nn.Linear(proj_dim, proj_dim)
        self.k_proj = nn.Linear(proj_dim, proj_dim)
        self.v_proj = nn.Linear(proj_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = proj_dim // num_heads

    def forward(self, mv, key_padding_mask=None):
        B, L, C, D = mv.shape
        # Extract heap components
        heap = mv[..., self.heap_idx]  # [B, L, C, heap_dim]
        heap_flat = heap.reshape(B, L, -1)  # [B, L, C*heap_dim]

        Q = self.q_proj(heap_flat).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(heap_flat).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(heap_flat).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(B, L, -1)
        out_flat = self.out_proj(context)
        out_heap = out_flat.reshape(B, L, C, self.heap_dim)

        # Write back to full multivector
        result = mv.clone()
        heap_idx_exp = self.heap_idx.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, L, C, -1)
        result.scatter_(3, heap_idx_exp, out_heap)
        return result
