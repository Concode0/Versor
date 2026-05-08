# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric reasoning layers for SLM experiments."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from core.module import CliffordModule
from layers.blocks.multi_rotor_ffn import MultiRotorFFN
from layers.primitives.linear import CliffordLinear
from layers.primitives.normalization import CliffordLayerNorm


class CausalGeometricAttention(CliffordModule):
    """Causal self-attention scored by scalar and bivector product evidence.

    This is intentionally model-local rather than using ``GeometricTransformerBlock``,
    so SLM-specific attention changes can happen inside ``models/slm``.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_heads: int = 4,
        bivector_weight: float = 0.5,
        dropout: float = 0.0,
        block_size: int = 128,
    ):
        super().__init__(algebra)
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        if block_size <= 0:
            raise ValueError(f"block_size ({block_size}) must be > 0")

        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.bivector_weight = bivector_weight
        self.block_size = block_size

        self.q_proj = CliffordLinear(algebra, channels, channels)
        self.k_proj = CliffordLinear(algebra, channels, channels)
        self.v_proj = CliffordLinear(algebra, channels, channels)
        self.out_proj = CliffordLinear(algebra, channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self._precompute_score_tables()

    def _precompute_score_tables(self):
        """Precompute bilinear tables for ``Q * reverse(K)`` attention scores."""
        algebra = self.algebra
        D = algebra.dim

        metric_rev = algebra.gp_signs[:, 0].float() * algebra.rev_signs.float()
        self.register_buffer("_metric_rev", metric_rev)

        g2_blades = [idx for idx in range(D) if bin(idx).count("1") == 2]
        self.n_g2 = len(g2_blades)

        if g2_blades:
            a_idx = torch.arange(D, device=algebra.device)
            r_vals = torch.tensor(g2_blades, dtype=torch.long, device=algebra.device)
            b_idx = a_idx.unsqueeze(0) ^ r_vals.unsqueeze(1)
            rev_b = algebra.rev_signs.float()[b_idx]
            gp_ar = algebra.gp_signs[:, r_vals].float().T
            g2_sign = rev_b * gp_ar
        else:
            b_idx = torch.zeros(0, D, dtype=torch.long, device=algebra.device)
            g2_sign = torch.zeros(0, D, device=algebra.device)

        self.register_buffer("_g2_b_idx", b_idx)
        self.register_buffer("_g2_sign", g2_sign)

    def _compute_score(self, q_head: torch.Tensor, k_head: torch.Tensor, k_g2: torch.Tensor) -> torch.Tensor:
        """Compute grade-0 plus grade-2 attention scores without materializing GP tensors."""
        B, H, Lq, Hc, D = q_head.shape
        Lk = k_head.shape[2]

        metric_rev = self._metric_rev.to(dtype=q_head.dtype)
        q_flat = (q_head * metric_rev).reshape(B, H, Lq, Hc * D)
        k_flat = k_head.reshape(B, H, Lk, Hc * D)
        score_g0 = torch.matmul(q_flat, k_flat.transpose(-2, -1))

        if self.n_g2 > 0:
            q_2d = q_head.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lq, D)
            k_g2_2d = k_g2.permute(0, 1, 3, 2, 4, 5).reshape(B * H * Hc, Lk * self.n_g2, D)
            comp = torch.bmm(q_2d, k_g2_2d.transpose(-2, -1))
            comp_sq = comp.reshape(B * H * Hc, Lq, Lk, self.n_g2).pow(2).sum(-1)
            score_g2 = comp_sq.reshape(B, H, Hc, Lq, Lk).sum(2).clamp(min=self.algebra.eps).sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        return (score_g0 + self.bivector_weight * score_g2) / math.sqrt(Hc * D)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Apply causal geometric attention.

        Args:
            x: Token multivectors ``[B, L, C, D]``.
            key_padding_mask: Optional bool mask ``[B, L]`` where ``True`` means padded.
        """
        B, L, C, D = x.shape
        H = self.num_heads
        Hc = self.head_channels

        flat = x.reshape(B * L, C, D)
        Q = self.q_proj(flat).reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)
        K = self.k_proj(flat).reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)
        V = self.v_proj(flat).reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)

        causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)

        g2_sign = self._g2_sign.to(dtype=K.dtype)
        K_g2 = K[..., self._g2_b_idx] * g2_sign

        output_chunks = []
        for q_start in range(0, L, self.block_size):
            q_end = min(q_start + self.block_size, L)
            Q_block = Q[:, :, q_start:q_end]
            scores = self._compute_score(Q_block, K, K_g2)

            mask_block = causal_mask[q_start:q_end, :]
            scores = scores.masked_fill(mask_block.unsqueeze(0).unsqueeze(0), float("-inf"))

            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

            weights = F.softmax(scores, dim=-1)
            if self.dropout is not None:
                weights = self.dropout(weights)

            output_chunks.append(torch.einsum("bhij,bhjcd->bhicd", weights, V))

        out = torch.cat(output_chunks, dim=2).permute(0, 2, 1, 3, 4).reshape(B, L, C, D)
        out = self.out_proj(out.reshape(B * L, C, D)).reshape(B, L, C, D)

        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        return out


class GeometricSLMBlock(CliffordModule):
    """A model-local causal block with geometric attention and multi-rotor FFN."""

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_heads: int = 4,
        num_rotors: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        bivector_weight: float = 0.5,
        attn_block_size: int = 128,
    ):
        super().__init__(algebra)
        self.norm1 = CliffordLayerNorm(algebra, channels)
        self.attn = CausalGeometricAttention(
            algebra=algebra,
            channels=channels,
            num_heads=num_heads,
            bivector_weight=bivector_weight,
            dropout=dropout,
            block_size=attn_block_size,
        )
        self.norm2 = CliffordLayerNorm(algebra, channels)
        self.ffn = MultiRotorFFN(algebra, channels, ffn_mult=ffn_mult, num_rotors=num_rotors)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, L, C, D = x.shape

        h = self.norm1(x.reshape(B * L, C, D)).reshape(B, L, C, D)
        h = self.attn(h, key_padding_mask=key_padding_mask)
        if self.dropout is not None:
            h = self.dropout(h)
        x = x + h

        h = self.norm2(x.reshape(B * L, C, D)).reshape(B, L, C, D)
        h = self.ffn(h.reshape(B * L, C, D)).reshape(B, L, C, D)
        if self.dropout is not None:
            h = self.dropout(h)
        return x + h
