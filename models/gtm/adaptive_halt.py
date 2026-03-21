# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""FIM-based adaptive computation halt.

Steps that produce more information gain get higher mixing weight.
At inference, halts when weighted information gain drops below a threshold.
"""

import torch
import torch.nn as nn


class FIMAdaptiveHalt(nn.Module):
    """FIM-based adaptive computation time controller.

    Computes mixing weights from per-step conviction-weighted information gain.
    Steps with higher info gain get proportionally more weight.
    """

    def __init__(self, halt_eps: float = 0.01):
        super().__init__()
        self.halt_eps = halt_eps

    def forward(self, delta_infos: list, weights_list: list) -> dict:
        """Compute mixing weights from per-step FIM information gains.

        Args:
            delta_infos: List of T tensors, each [B, K] (per-hypothesis info gain).
            weights_list: List of T tensors, each [B, K] (hypothesis attention weights).

        Returns:
            dict with:
                'mixing_weights': [B, T] mixing weights for per-step outputs
                'expected_steps': [B] expected computation depth
        """
        T = len(delta_infos)
        device = delta_infos[0].device
        dtype = delta_infos[0].dtype

        # Per-step conviction-weighted info gain
        per_step = []
        for d, w in zip(delta_infos, weights_list):
            per_step.append((d * w).sum(dim=-1))  # [B]

        info_stack = torch.stack(per_step, dim=1)  # [B, T]

        # Mixing weights: proportional to positive info gain
        info_pos = info_stack.clamp(min=1e-8)
        mixing = info_pos / info_pos.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, T]

        # Expected steps (1-indexed)
        steps = torch.arange(1, T + 1, device=device, dtype=dtype)
        expected = (mixing * steps.unsqueeze(0)).sum(dim=1)  # [B]

        return {
            'mixing_weights': mixing,
            'expected_steps': expected,
        }
