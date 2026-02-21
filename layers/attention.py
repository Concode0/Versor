# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from layers.base import CliffordModule
from layers.linear import CliffordLinear


# Memory-bounded block size for chunked attention computation
_BLOCK_SIZE = 64


class GeometricProductAttention(CliffordModule):
    """Multi-head attention using geometric product scoring.

    Standard attention: score(Q, K) = <Q, K> / sqrt(d)  (scalar only)

    GA attention:
        product = Q_c * reverse(K_c)    (geometric product per head-channel)
        score   = (<product>_0 + λ * ||<product>_2||_F) / sqrt(H_c * dim)

    The grade-0 (scalar) part measures alignment (like dot product).
    The grade-2 (bivector) part measures relative orientation — novel.

    Memory: naive [B, H, L, L, H_c, D] is too large. We chunk over L_q
    in blocks of BLOCK_SIZE to bound peak VRAM.

    Attributes:
        num_heads (int): Number of attention heads.
        head_channels (int): Channels per head.
        causal (bool): If True, apply autoregressive causal mask.
        bivector_weight (float): λ — weight of bivector score component.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_heads: int,
        causal: bool = True,
        bivector_weight: float = 0.5,
        dropout: float = 0.0,
    ):
        """Sets up geometric product attention.

        Args:
            algebra: Clifford algebra instance.
            channels: Total number of multivector channels.
            num_heads: Number of attention heads.
            causal: Apply causal mask for autoregressive generation.
            bivector_weight: λ weight on bivector score component.
            dropout: Dropout rate on attention weights.
        """
        super().__init__(algebra)
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.causal = causal
        self.bivector_weight = bivector_weight

        # Q, K, V projections operate on [B*L, channels, dim]
        self.q_proj = CliffordLinear(algebra, channels, channels)
        self.k_proj = CliffordLinear(algebra, channels, channels)
        self.v_proj = CliffordLinear(algebra, channels, channels)
        self.out_proj = CliffordLinear(algebra, channels, channels)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Precompute bilinear score tables (replaces pairwise geometric product)
        self._precompute_score_tables()

    def _precompute_score_tables(self):
        """Precomputes lookup tables for efficient attention scoring.

        Replaces the O(L²) full pairwise geometric product with direct bilinear
        forms for grade-0 and grade-2 components of Q * reverse(K):

        Grade-0:  <Q * rev(K)>_0 = Σ_a Q[a] * K[a] * metric_rev[a]
                  → simple weighted dot product, no pairwise expansion needed.

        Grade-2:  <Q * rev(K)>_r = Σ_a Q[a] * K[a^r] * g2_sign[r, a]
                  → precompute K_g2 once, then batched matmul.

        Memory: ~4 MB peak vs ~256 MB for the naive B_gathered approach.
        """
        alg = self.algebra
        D = alg.dim

        # Grade-0 metric: metric_rev[a] = gp_signs[a, 0] * rev_signs[a]
        # gp_signs[a, 0] is the sign when A[a] * B[a] contributes to output blade 0
        metric_rev = alg.gp_signs[:, 0].float() * alg.rev_signs.float()
        self.register_buffer('_metric_rev', metric_rev)  # [D]

        # Grade-2 tables: for each grade-2 blade r, for each A-blade a:
        #   B-blade  = a XOR r
        #   sign     = rev_sign[a^r] * gp_signs[a, r]
        g2_blades = [i for i in range(D) if bin(i).count('1') == 2]
        n_g2 = len(g2_blades)
        self.n_g2 = n_g2

        if n_g2 > 0:
            a_idx = torch.arange(D, device=alg.device)
            r_vals = torch.tensor(g2_blades, dtype=torch.long, device=alg.device)  # [n_g2]

            # b_idx[r, a] = a XOR r_vals[r]
            b_idx = a_idx.unsqueeze(0) ^ r_vals.unsqueeze(1)  # [n_g2, D]

            # rev_sign at the B-blade position
            rev_b = alg.rev_signs.float()[b_idx]  # [n_g2, D]

            # gp_signs[a, r_val]: sign when A[a] pairs with B[a^r] to give output r
            # alg.gp_signs[:, r_vals] → [D, n_g2]; transpose → [n_g2, D]
            gp_ar = alg.gp_signs[:, r_vals].float().T  # [n_g2, D]

            g2_sign = rev_b * gp_ar  # [n_g2, D]
        else:
            b_idx = torch.zeros(0, D, dtype=torch.long, device=alg.device)
            g2_sign = torch.zeros(0, D, device=alg.device)

        self.register_buffer('_g2_b_idx', b_idx)   # [n_g2, D] long
        self.register_buffer('_g2_sign', g2_sign)  # [n_g2, D] float

    def _compute_score(
        self,
        q_head: torch.Tensor,
        k_head: torch.Tensor,
        k_g2: torch.Tensor,
    ) -> torch.Tensor:
        """Computes GA attention score using precomputed bilinear form tables.

        Avoids the O(B·H·Lq·Lk·Hc·D·BLOCK) memory of the full pairwise
        geometric product. Instead:

          Grade-0: score_g0 = Q_weighted @ K^T  (weighted dot product, peak ~1 MB)
          Grade-2: batched matmul via precomputed k_g2            (peak ~4 MB)

        Args:
            q_head: Query block [B, H, Lq, Hc, D]
            k_head: Keys        [B, H, Lk, Hc, D]
            k_g2:   Precomputed [B, H, Lk, Hc, n_g2, D]
                    k_g2[b,h,j,c,r,d] = K[b,h,j,c, d^r] * g2_sign[r, d]

        Returns:
            scores: [B, H, Lq, Lk]
        """
        B, H, Lq, Hc, D = q_head.shape
        Lk = k_head.shape[2]
        n_g2 = self.n_g2

        # ── Grade-0 score ────────────────────────────────────────────────────
        # <Q * rev(K)>_0 = Σ_c Σ_d  Q[c,d] * K[c,d] * metric_rev[d]
        # Implemented as a batched matrix multiply: [B,H,Lq,Hc*D] @ [B,H,Hc*D,Lk]
        q_weighted = q_head * self._metric_rev          # [B, H, Lq, Hc, D]
        q_flat = q_weighted.reshape(B, H, Lq, Hc * D)  # [B, H, Lq, Hc*D]
        k_flat = k_head.reshape(B, H, Lk, Hc * D)      # [B, H, Lk, Hc*D]
        score_g0 = torch.matmul(q_flat, k_flat.transpose(-2, -1))  # [B, H, Lq, Lk]

        # ── Grade-2 score ────────────────────────────────────────────────────
        # ||<Q * rev(K)>_2||_F = sqrt(Σ_c Σ_r (Σ_d Q[c,d]*k_g2[j,c,r,d])^2)
        # Batched matmul merging (B, H, Hc) into one batch dimension:
        #   q_2d:     [B*H*Hc, Lq, D]
        #   k_g2_2d:  [B*H*Hc, Lk*n_g2, D]   (Lk and n_g2 merged, n_g2 varies fast)
        #   comp:     [B*H*Hc, Lq, Lk*n_g2]
        # Peak ~4 MB vs ~256 MB for the naive B_gathered approach.
        if n_g2 > 0:
            q_2d = q_head.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lq, D)
            # k_g2: [B, H, Lk, Hc, n_g2, D] → permute to [B, H, Hc, Lk, n_g2, D]
            k_g2_t = k_g2.permute(0, 1, 3, 2, 4, 5)
            k_g2_2d = k_g2_t.reshape(B * H * Hc, Lk * n_g2, D)
            # [B*H*Hc, Lq, D] @ [B*H*Hc, D, Lk*n_g2] → [B*H*Hc, Lq, Lk*n_g2]
            comp = torch.bmm(q_2d, k_g2_2d.transpose(-2, -1))
            # Sum squared components over n_g2, then sum over Hc → [B, H, Lq, Lk]
            comp_sq = comp.reshape(B * H * Hc, Lq, Lk, n_g2).pow(2).sum(-1)  # [B*H*Hc, Lq, Lk]
            score_g2_sq = comp_sq.reshape(B, H, Hc, Lq, Lk).sum(2)           # [B, H, Lq, Lk]
            score_g2 = score_g2_sq.sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        # Combined score
        scale = math.sqrt(self.head_channels * self.algebra.dim)
        return (score_g0 + self.bivector_weight * score_g2) / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes geometric product attention.

        Args:
            x: Input multivectors [B, L, C, D].

        Returns:
            Output multivectors [B, L, C, D].
        """
        B, L, C, D = x.shape

        # Project Q, K, V (CliffordLinear expects [B, C, D])
        x_flat = x.reshape(B * L, C, D)
        Q = self.q_proj(x_flat).reshape(B, L, C, D)
        K = self.k_proj(x_flat).reshape(B, L, C, D)
        V = self.v_proj(x_flat).reshape(B, L, C, D)

        H = self.num_heads
        Hc = self.head_channels

        # Reshape to [B, H, L, Hc, D]
        Q = Q.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)  # [B, H, L, Hc, D]
        K = K.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)
        V = V.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)

        # Build causal mask once [L, L]
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
            )  # True = masked (future)
        else:
            causal_mask = None

        # Precompute K_g2 once for all query blocks — much cheaper than recomputing
        # k_g2[b,h,j,c,r,d] = K[b,h,j,c, d^r_val] * g2_sign[r, d]
        # Shape: [B, H, L, Hc, n_g2, D]  ≈ 768 KB for the small MPS config
        K_g2 = K[..., self._g2_b_idx] * self._g2_sign  # [B, H, L, Hc, n_g2, D]

        # Chunked attention over query positions to bound memory
        output_chunks = []
        for q_start in range(0, L, _BLOCK_SIZE):
            q_end = min(q_start + _BLOCK_SIZE, L)

            Q_block = Q[:, :, q_start:q_end]  # [B, H, Lq, Hc, D]

            # Compute scores: [B, H, Lq, L]
            scores = self._compute_score(Q_block, K, K_g2)

            # Apply causal mask
            if causal_mask is not None:
                mask_block = causal_mask[q_start:q_end, :]  # [Lq, L]
                scores = scores.masked_fill(
                    mask_block.unsqueeze(0).unsqueeze(0), float('-inf')
                )

            # Softmax + dropout
            attn_weights = F.softmax(scores, dim=-1)  # [B, H, Lq, L]
            if self.attn_dropout is not None:
                attn_weights = self.attn_dropout(attn_weights)

            # Aggregate values: sum_k attn[b,h,i,k] * V[b,h,k,Hc,D]
            # attn_weights: [B, H, Lq, L]
            # V:            [B, H, L,  Hc, D]
            # out:          [B, H, Lq, Hc, D]
            out_block = torch.einsum('bhij,bhjcd->bhicd', attn_weights, V)
            output_chunks.append(out_block)

        # Reassemble: [B, H, L, Hc, D]
        output = torch.cat(output_chunks, dim=2)

        # Merge heads back: [B, L, C, D]
        output = output.permute(0, 2, 1, 3, 4).reshape(B, L, C, D)

        # Output projection
        output = self.out_proj(output.reshape(B * L, C, D)).reshape(B, L, C, D)

        return output
