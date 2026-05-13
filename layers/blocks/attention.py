# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.foundation.basis import normalize_grades, reverse_sign
from core.foundation.module import CliffordModule

from ..grade import check_multivector_lanes, lane_count, resolve_layer_layout
from ..primitives.linear import CliffordLinear

# Memory-bounded block size for chunked attention computation
_BLOCK_SIZE = 64
_G2_BLADE_CHUNK_SIZE = 16
_SCORE_PRECOMPUTE_LIMIT = 8_000_000


class GeometricProductAttention(CliffordModule):
    """Multi-head attention using geometric product scoring.

    Standard attention: score(Q, K) = <Q, K> / sqrt(d)  (scalar only)

    GA attention:
        product = Q_c * reverse(K_c)    (geometric product per head-channel)
        score   = (<product>_0 + lambda_ * ||<product>_2||_F) / sqrt(H_c * dim)

    The grade-0 (scalar) part measures alignment (like dot product).
    The grade-2 (bivector) part measures relative orientation - novel.

    Memory: naive [B, H, L, L, H_c, D] is too large. We chunk over L_q
    in blocks of BLOCK_SIZE to bound peak VRAM.

    Attributes:
        num_heads (int): Number of attention heads.
        head_channels (int): Channels per head.
        causal (bool): If True, apply autoregressive causal mask.
        bivector_weight (float): lambda_ - weight of bivector score component.
    """

    def __init__(
        self,
        algebra,
        channels: int,
        num_heads: int,
        causal: bool = True,
        bivector_weight: float = 0.5,
        dropout: float = 0.0,
        feature_grades=None,
        score_grades=None,
        score_blade_chunk_size: int = _G2_BLADE_CHUNK_SIZE,
        score_precompute_limit: int = _SCORE_PRECOMPUTE_LIMIT,
    ):
        """Sets up geometric product attention.

        Args:
            algebra: Clifford algebra instance.
            channels: Total number of multivector channels.
            num_heads: Number of attention heads.
            causal: Apply causal mask for autoregressive generation.
            bivector_weight: lambda_ weight on bivector score component.
            dropout: Dropout rate on attention weights.
            feature_grades: Optional active grades carried by this attention layer.
                When set, projections and attention values use compact lanes.
            score_grades: Optional declared grades for compact planned scoring.
                ``None`` preserves exact dense scoring over all basis lanes unless
                ``feature_grades`` is set, in which case those feature grades are
                also used for scoring.
            score_blade_chunk_size: Grade-2 output blades processed per dense
                chunk when exact dense scoring is used.
            score_precompute_limit: Maximum temporary ``K_g2`` elements allowed
                before exact dense scoring switches to chunked grade-2 blades.
        """
        super().__init__(algebra)
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.channels = channels
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.causal = causal
        self.bivector_weight = bivector_weight
        self.feature_layout = resolve_layer_layout(algebra, feature_grades)
        self.feature_dim = lane_count(algebra, self.feature_layout)
        if score_grades is None and feature_grades is not None:
            score_grades = feature_grades
        self.score_grades = None if score_grades is None else normalize_grades(score_grades, algebra.n)
        self.score_blade_chunk_size = max(1, int(score_blade_chunk_size))
        self.score_precompute_limit = max(0, int(score_precompute_limit))

        # Q, K, V projections operate on [B*L, channels, dim]
        self.q_proj = CliffordLinear(algebra, channels, channels, grades=feature_grades)
        self.k_proj = CliffordLinear(algebra, channels, channels, grades=feature_grades)
        self.v_proj = CliffordLinear(algebra, channels, channels, grades=feature_grades)
        self.out_proj = CliffordLinear(algebra, channels, channels, grades=feature_grades)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Precompute bilinear score routes (replaces pairwise geometric product)
        self._precompute_score_tables()

    def _precompute_score_tables(self):
        """Precompute exact dense or compact-planned attention score routes.

        Dense scoring is exact for existing callers and chunks grade-2 blades
        instead of materializing ``[B, H, L, Hc, n_g2, D]``. Compact scoring
        uses static grade product plans when ``score_grades`` declares a layout.
        """
        alg = self.algebra
        D = alg.dim

        self._score_layout = None
        self._score_scalar_product = None
        self._score_bivector_product = None
        self._score_scale_dim = self.feature_dim if self.feature_layout is not None else D
        if self.score_grades is not None:
            self.n_g2 = alg.n * (alg.n - 1) // 2
            self._score_layout = alg.planner.layout(self.score_grades)
            layout_indices = self._score_input_positions(device=alg.device)
            rev_signs = torch.tensor(
                [reverse_sign(index) for index in self._score_layout.basis_indices],
                dtype=torch.float32,
                device=alg.device,
            )
            self.register_buffer("_score_layout_indices", layout_indices)
            self.register_buffer("_score_rev_signs", rev_signs)
            self._score_scalar_product = alg.planner.product_executor(
                op="gp",
                left_grades=self.score_grades,
                right_grades=self.score_grades,
                output_grades=(0,),
                device=alg.device,
                dtype=alg.dtype,
                cache=False,
            )
            if self.n_g2 > 0:
                self._score_bivector_product = alg.planner.product_executor(
                    op="gp",
                    left_grades=self.score_grades,
                    right_grades=self.score_grades,
                    output_grades=(2,),
                    device=alg.device,
                    dtype=alg.dtype,
                    cache=False,
                )
            return

        if not hasattr(alg, "gp_signs") or not hasattr(alg, "rev_signs"):
            raise ValueError("Dense attention scoring requires CliffordAlgebra; pass score_grades for AlgebraContext.")

        # Grade-0 metric: metric_rev[a] = gp_signs[a, 0] * rev_signs[a]
        # gp_signs[a, 0] is the sign when A[a] * B[a] contributes to output blade 0
        metric_rev = alg.gp_signs[:, 0].float() * alg.rev_signs.float()
        self.register_buffer("_metric_rev", metric_rev)  # [D]

        g2_blades = [i for i in range(D) if bin(i).count("1") == 2]
        self.n_g2 = len(g2_blades)
        self.register_buffer("_g2_blades", torch.tensor(g2_blades, dtype=torch.long, device=alg.device))
        self.register_buffer("_basis_indices", torch.arange(D, dtype=torch.long, device=alg.device))

        self.register_buffer("_score_layout_indices", torch.zeros(0, dtype=torch.long, device=alg.device))
        self.register_buffer("_score_rev_signs", torch.zeros(0, device=alg.device))

    def _compute_score(
        self,
        q_head: torch.Tensor,
        k_head: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GA attention scores for one query block."""
        if self._score_layout is not None:
            return self._compute_score_compact(q_head, k_head)
        return self._compute_score_dense(q_head, k_head)

    def _compute_score_dense(self, q_head: torch.Tensor, k_head: torch.Tensor) -> torch.Tensor:
        """Exact dense score with automatic full/prechunked grade-2 routing."""
        B, H, Lq, Hc, D = q_head.shape
        Lk = k_head.shape[2]
        n_g2 = self.n_g2

        # == Grade-0 score ====================================================
        # <Q * rev(K)>_0 = Sum_c Sum_d  Q[c,d] * K[c,d] * metric_rev[d]
        # Implemented as a batched matrix multiply: [B,H,Lq,Hc*D] @ [B,H,Hc*D,Lk]
        q_weighted = q_head * self._metric_rev  # [B, H, Lq, Hc, D]
        q_flat = q_weighted.reshape(B, H, Lq, Hc * D)  # [B, H, Lq, Hc*D]
        k_flat = k_head.reshape(B, H, Lk, Hc * D)  # [B, H, Lk, Hc*D]
        score_g0 = torch.matmul(q_flat, k_flat.transpose(-2, -1))  # [B, H, Lq, Lk]

        # == Grade-2 score ====================================================
        # ||<Q * rev(K)>_2||_F = sqrt(Sum_c Sum_r (Sum_d Q[c,d]*k_g2[j,c,r,d])^2)
        if n_g2 > 0:
            q_2d = q_head.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lq, D)

            full_k_g2_elements = B * H * Lk * Hc * n_g2 * D
            if full_k_g2_elements <= self.score_precompute_limit:
                score_g2_sq = self._dense_score_g2_precomputed(q_2d, k_head, B, H, Hc, Lq, Lk, D, n_g2)
            else:
                k_2d = k_head.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lk, D)
                score_g2_sq = self._dense_score_g2_chunked(q_2d, k_2d, B, H, Hc, Lq, Lk, D, n_g2)
            score_g2 = score_g2_sq.sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        # Combined score
        scale = math.sqrt(self.head_channels * self._score_scale_dim)
        return (score_g0 + self.bivector_weight * score_g2) / scale

    def _dense_score_g2_precomputed(self, q_2d, k_head, B, H, Hc, Lq, Lk, D, n_g2):
        """Dense grade-2 score using one full shifted-key materialization."""
        r_vals = self._g2_blades
        b_idx = self._basis_indices.unsqueeze(0) ^ r_vals.unsqueeze(1)
        rev_b = self.algebra.rev_signs[b_idx].to(dtype=k_head.dtype)
        gp_ar = self.algebra.gp_signs[:, r_vals].T.to(dtype=k_head.dtype)
        g2_sign = rev_b * gp_ar

        k_g2 = k_head[..., b_idx] * g2_sign
        k_g2_2d = k_g2.permute(0, 1, 3, 2, 4, 5).reshape(B * H * Hc, Lk * n_g2, D)
        comp = torch.bmm(q_2d, k_g2_2d.transpose(-2, -1))
        comp_sq = comp.reshape(B * H * Hc, Lq, Lk, n_g2).pow(2).sum(-1)
        return comp_sq.reshape(B, H, Hc, Lq, Lk).sum(2)

    def _dense_score_g2_chunked(self, q_2d, k_2d, B, H, Hc, Lq, Lk, D, n_g2):
        """Dense grade-2 score using bounded output-blade chunks."""
        score_g2_sq = q_2d.new_zeros(B, H, Lq, Lk)
        for start in range(0, n_g2, self.score_blade_chunk_size):
            end = min(start + self.score_blade_chunk_size, n_g2)
            r_vals = self._g2_blades[start:end]
            b_idx = self._basis_indices.unsqueeze(0) ^ r_vals.unsqueeze(1)
            rev_b = self.algebra.rev_signs[b_idx].to(dtype=k_2d.dtype)
            gp_ar = self.algebra.gp_signs[:, r_vals].T.to(dtype=k_2d.dtype)
            g2_sign = rev_b * gp_ar

            k_shifted = torch.index_select(k_2d, -1, b_idx.reshape(-1))
            k_shifted = k_shifted * g2_sign.reshape(-1)
            k_g2_2d = k_shifted.reshape(B * H * Hc, Lk * (end - start), D)
            comp = torch.bmm(q_2d, k_g2_2d.transpose(-2, -1))
            comp_sq = comp.reshape(B * H * Hc, Lq, Lk, end - start).pow(2).sum(-1)
            score_g2_sq = score_g2_sq + comp_sq.reshape(B, H, Hc, Lq, Lk).sum(2)
        return score_g2_sq

    def _compute_score_compact(self, q_head: torch.Tensor, k_head: torch.Tensor) -> torch.Tensor:
        """Declared-grade score using static compact product plans."""
        B, H, Lq, Hc, _ = q_head.shape
        Lk = k_head.shape[2]

        q_values = torch.index_select(q_head, -1, self._score_layout_indices)
        k_values = torch.index_select(k_head, -1, self._score_layout_indices) * self._score_rev_signs.to(
            dtype=k_head.dtype
        )

        q_2d = q_values.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lq, self._score_layout.dim)
        k_2d = k_values.permute(0, 1, 3, 2, 4).reshape(B * H * Hc, Lk, self._score_layout.dim)

        scalar = self._score_scalar_product.forward_pairwise_compact(q_2d, k_2d).squeeze(-1)
        score_g0 = scalar.reshape(B, H, Hc, Lq, Lk).sum(2)

        if self._score_bivector_product is not None:
            bivectors = self._score_bivector_product.forward_pairwise_compact(q_2d, k_2d)
            score_g2_sq = bivectors.pow(2).sum(-1).reshape(B, H, Hc, Lq, Lk).sum(2)
            score_g2 = score_g2_sq.sqrt()
        else:
            score_g2 = torch.zeros_like(score_g0)

        scale = math.sqrt(self.head_channels * self._score_scale_dim)
        return (score_g0 + self.bivector_weight * score_g2) / scale

    def _score_input_positions(self, *, device) -> torch.Tensor:
        """Return score-lane positions in dense or declared feature storage."""
        if self.feature_layout is None:
            return self._score_layout.indices_tensor(device=device)
        position_by_basis = {index: position for position, index in enumerate(self.feature_layout.basis_indices)}
        missing = tuple(index for index in self._score_layout.basis_indices if index not in position_by_basis)
        if missing:
            raise ValueError("score_grades must be contained in feature_grades for compact attention")
        positions = [position_by_basis[index] for index in self._score_layout.basis_indices]
        return torch.tensor(positions, dtype=torch.long, device=device)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Computes geometric product attention.

        Args:
            x: Input multivectors [B, L, C, D].
            key_padding_mask: Optional [B, L] bool mask where True = padded (ignored).

        Returns:
            Output multivectors [B, L, C, D].
        """
        check_multivector_lanes(x, self.algebra, self.feature_layout, "GeometricProductAttention input")
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

        # Chunked attention over query positions to bound memory
        output_chunks = []
        for q_start in range(0, L, _BLOCK_SIZE):
            q_end = min(q_start + _BLOCK_SIZE, L)

            Q_block = Q[:, :, q_start:q_end]  # [B, H, Lq, Hc, D]

            # Compute scores: [B, H, Lq, L]
            scores = self._compute_score(Q_block, K)

            # Apply causal mask
            if causal_mask is not None:
                mask_block = causal_mask[q_start:q_end, :]  # [Lq, L]
                scores = scores.masked_fill(mask_block.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Apply key padding mask: True = padded -> -inf
            if key_padding_mask is not None:
                # key_padding_mask: [B, L] -> [B, 1, 1, L]
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

            # Softmax + dropout
            attn_weights = F.softmax(scores, dim=-1)  # [B, H, Lq, L]
            if self.attn_dropout is not None:
                attn_weights = self.attn_dropout(attn_weights)

            # Aggregate values: sum_k attn[b,h,i,k] * V[b,h,k,Hc,D]
            # attn_weights: [B, H, Lq, L]
            # V:            [B, H, L,  Hc, D]
            # out:          [B, H, Lq, Hc, D]
            out_block = torch.einsum("bhij,bhjcd->bhicd", attn_weights, V)
            output_chunks.append(out_block)

        # Reassemble: [B, H, L, Hc, D]
        output = torch.cat(output_chunks, dim=2)

        # Merge heads back: [B, L, C, D]
        output = output.permute(0, 2, 1, 3, 4).reshape(B, L, C, D)

        # Output projection
        output = self.out_proj(output.reshape(B * L, C, D)).reshape(B, L, C, D)

        return output
