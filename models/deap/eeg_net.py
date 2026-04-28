# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric EEG Emotion Classification Network (DEAP).

Combines the Mother pattern (Procrustes alignment, entropy-gated attention)
with the Neutral pattern (GeometricNeutralizer) for robust emotion prediction.

Core insight: emotional states are pushed into Grade-0 (scalar), which is
invariant under rotor sandwich products (R*s*R~ = s). Grade-2 bivectors
capture inter-region phase coupling; Grade-4 pseudoscalar captures global
brain state. GeometricNeutralizer orthogonalizes Grade-0 from Grade-2
artifacts before pooling, then MultiTargetPhaseShiftHead mixes Grade-0
(immediate) and Grade-4 (long-range) for VADL prediction.
"""

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from layers import (
    MotherEmbedding,
    GeometricTransformerBlock,
    CliffordLayerNorm,
    GeometricNeutralizer,
)
from layers.primitives.base import CliffordModule


class MultiTargetPhaseShiftHead(CliffordModule):
    """Maps full multivector geometry to target distributions.

    Projects the flattened multivector (all channels x all blade dimensions)
    to ``num_targets`` outputs, then applies a learnable scale and bias so
    each target can independently shift its prediction range.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, num_targets: int = 4):
        super().__init__(algebra)
        self.channels = channels
        self.num_targets = num_targets
        self.proj = nn.Linear(channels * algebra.dim, num_targets)
        self.log_scale = nn.Parameter(torch.zeros(num_targets))
        self.bias = nn.Parameter(torch.zeros(num_targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        flat_mv = x.reshape(B, -1)  # [B, channels * dim]
        raw_logits = self.proj(flat_mv)  # [B, num_targets]
        return raw_logits * self.log_scale.exp() + self.bias


class EEGNet(CliffordModule):
    """Geometric EEG Emotion Classification Network.

    Architecture:
        MotherEmbedding (per region) -> stack -> GeometricTransformerBlock x N
        -> CliffordLayerNorm -> GeometricNeutralizer -> mean pool
        -> MultiTargetPhaseShiftHead -> [B, num_targets]

    The GeometricNeutralizer is applied **per-token before pooling** so each
    brain region's Grade-0 is cleaned of its own Grade-2 artifacts independently.
    """

    def __init__(self, group_sizes, profiles=None, device=None, config=None):
        """Initialize EEGNet.

        Args:
            group_sizes: Dict mapping region name to input feature dim
                         (e.g. ``{'frontal': 52, 'central': 28, ...}``).
            profiles: Optional dict ``{region: {'U': float, 'V': Tensor}}``
                      from the profiler for MotherEmbedding alignment.
            device: Torch device.
            config: Hydra DictConfig or plain dict with model hyperparameters.
        """
        p, q = 3, 1
        if config is not None:
            if hasattr(config, 'algebra'):
                p = config.algebra.get('p', 3)
                q = config.algebra.get('q', 1)
            elif isinstance(config, dict):
                p = config.get('p', 3)
                q = config.get('q', 1)

        algebra = CliffordAlgebra(p, q, device=device)
        super().__init__(algebra)

        if config is not None and hasattr(config, 'model'):
            m = config.model
            channels = m.get('channels', 16)
            num_layers = m.get('num_layers', 3)
            num_heads = m.get('num_heads', 4)
            num_rotors = m.get('num_rotors', 8)
            eta = m.get('eta_gating', 1.5)
            H_base = m.get('H_base', 0.5)
            dropout = m.get('dropout', 0.1)
            num_targets = m.get('num_targets', 4)
        elif isinstance(config, dict):
            channels = config.get('channels', 16)
            num_layers = config.get('num_layers', 3)
            num_heads = config.get('num_heads', 4)
            num_rotors = config.get('num_rotors', 8)
            eta = config.get('eta_gating', 1.5)
            H_base = config.get('H_base', 0.5)
            dropout = config.get('dropout', 0.1)
            num_targets = config.get('num_targets', 4)
        else:
            channels, num_layers, num_heads, num_rotors = 16, 3, 4, 8
            eta, H_base, dropout, num_targets = 1.5, 0.5, 0.1, 4

        self.channels = channels
        self.group_names = sorted(group_sizes.keys())

        self.embeddings = nn.ModuleDict()
        for name, size in group_sizes.items():
            U = profiles[name]['U'] if profiles and name in profiles else 0.0
            V = profiles[name]['V'] if profiles and name in profiles else None
            self.embeddings[name] = MotherEmbedding(self.algebra, size, channels, U, V)

        self.blocks = nn.ModuleList(
            [
                GeometricTransformerBlock(
                    self.algebra,
                    channels,
                    num_heads,
                    num_rotors,
                    dropout=dropout,
                    use_entropy_gating=True,
                    eta=eta,
                    H_base=H_base,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = CliffordLayerNorm(self.algebra, channels)
        self.neutralizer = GeometricNeutralizer(self.algebra, channels)
        self.head = MultiTargetPhaseShiftHead(self.algebra, channels, num_targets)

    def forward(self, group_data, return_diagnostics=False):
        tokens = [self.embeddings[name](group_data[name]) for name in self.group_names]
        x = torch.stack(tokens, dim=1)  # [B, L, C, D]

        all_H, all_lambda = [], []
        for block in self.blocks:
            if return_diagnostics:
                x, H, lam = block(x, return_state=True)
                all_H.append(H)
                all_lambda.append(lam)
            else:
                x = block(x)

        B, L, C, D = x.shape
        x = self.neutralizer(self.final_norm(x.reshape(B * L, C, D)))
        x = x.reshape(B, L, C, D).mean(dim=1)  # [B, C, D]
        preds = self.head(x)

        if return_diagnostics:
            return preds, torch.stack(all_H).mean(dim=0), torch.stack(all_lambda).mean(dim=0)

        return preds
