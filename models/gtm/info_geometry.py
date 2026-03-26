# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""FIMEvaluator: target-free Fisher Information proxy via grade-wise variance.

Scores hypotheses by measuring how much each deviates from the mean
across hypotheses, weighted by learnable per-grade importance. Higher
FIM proxy = more informative hypothesis (more distinctive structure).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm
from layers.primitives.base import CliffordModule


class FIMEvaluator(CliffordModule):
    """Fisher Information Matrix proxy using grade-wise variance across hypotheses.

    For each hypothesis k, measures how its grade-wise structure deviates from
    the mean across all hypotheses. Learnable grade_weights control per-grade
    importance. Optionally provides supervised FIM using target comparison.
    """

    def __init__(self, algebra: CliffordAlgebra):
        super().__init__(algebra)
        self.grade_weights = nn.Parameter(torch.zeros(algebra.num_grades))

    def fim_proxy(self, states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Unsupervised FIM proxy: grade-wise variance across hypotheses.

        Args:
            states: Candidate states [B, K, N, D].
            mask: Optional validity mask [B, N] (True=valid).

        Returns:
            FIM proxy values [B, K].
        """
        B, K, N, D = states.shape

        # Mean across hypotheses per position
        mean_state = states.mean(dim=1, keepdim=True)  # [B, 1, N, D]

        # Per-grade deviation for each hypothesis
        diff = states - mean_state  # [B, K, N, D]
        weights = F.softplus(self.grade_weights)  # [num_grades], positive

        # Compute per-grade squared norm of deviation
        fim = torch.zeros(B, K, device=states.device, dtype=states.dtype)
        for g in range(self.algebra.num_grades):
            diff_g = self.algebra.grade_projection(
                diff.reshape(B * K * N, D), g
            ).reshape(B, K, N, D)
            # Hermitian norm squared per position
            sq = (diff_g ** 2).sum(dim=-1)  # [B, K, N]
            if mask is not None:
                sq = sq * mask.unsqueeze(1).float()
            fim = fim + weights[g] * sq.mean(dim=-1)  # [B, K]

        return fim

    def information_gain(self, fim_cur: torch.Tensor,
                         fim_prev: torch.Tensor) -> torch.Tensor:
        """Compute information gain between successive FIM evaluations.

        Args:
            fim_cur: Current FIM values [B, K].
            fim_prev: Previous FIM values [B, K].

        Returns:
            Information gain [B, K].
        """
        return fim_cur - fim_prev

    def supervised_fim(self, states: torch.Tensor,
                       targets: torch.Tensor,
                       mask: torch.Tensor = None) -> torch.Tensor:
        """Training-only: compare grade-0 of each candidate to target encoding.

        Provides stronger signal than the unsupervised proxy by directly
        measuring how well each hypothesis's scalar component matches targets.

        Args:
            states: Candidate states [B, K, N, D].
            targets: Target color indices [B, N] (long, 0-9).
            mask: Optional validity mask [B, N].

        Returns:
            Supervised FIM values [B, K] (higher = better match).
        """
        B, K, N, D = states.shape
        # Grade-0 of candidates = predicted color signal
        pred_color = states[:, :, :, 0]  # [B, K, N]

        # Target as normalized float
        target_color = targets.float() / 9.0  # [B, N]

        # Negative squared error (higher = better)
        sq_err = -(pred_color - target_color.unsqueeze(1)) ** 2  # [B, K, N]
        if mask is not None:
            sq_err = sq_err * mask.unsqueeze(1).float()
            denom = mask.float().sum(dim=-1).clamp(min=1.0)  # [B]
            return sq_err.sum(dim=-1) / denom.unsqueeze(1)  # [B, K]

        return sq_err.mean(dim=-1)  # [B, K]
