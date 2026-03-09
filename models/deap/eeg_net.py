# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Geometric EEG Emotion Classification Network (DEAP).

Combines the Mother pattern (Procrustes alignment, entropy-gated attention)
with the Neutral pattern (GeometricNeutralizer) for robust emotion prediction.

Core insight: emotional states are pushed into Grade-0 (scalar), which is
invariant under rotor sandwich products (R·s·R~ = s). Grade-2 bivectors
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
    """Phase-Shift Head producing multiple output dimensions.

    Extends PhaseShiftHead to output ``num_targets`` predictions (e.g. 4 for
    VADL) by learning a separate phase angle per target dimension.

    Each target mixes Grade-0 (scalar component) and Grade-4 (high-grade
    component) via a per-channel phase rotation, then aggregates channels
    through a **learned per-target projection** (not uniform mean).

    The per-target bias decouples baseline prediction from the phase
    computation, preventing the phase angles from collapsing to encode
    only the label mean.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, num_targets: int = 4):
        super().__init__(algebra)
        self.channels = channels
        self.num_targets = num_targets
        self.theta = nn.Parameter(torch.randn(1, channels, num_targets) * 0.1)
        self.proj = nn.Parameter(
            torch.full((num_targets, channels), 1.0 / channels)
        )
        self.bias = nn.Parameter(torch.zeros(num_targets))

        mask_g4 = self.algebra.grade_masks[4]
        if mask_g4.sum() > 0:
            self.register_buffer('g4_idx', mask_g4.nonzero(as_tuple=True)[0])
        else:
            self.g4_idx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G0 = x[..., 0:1]  # [B, C, 1]

        if self.g4_idx is not None and len(self.g4_idx) > 0:
            G4 = x[..., self.g4_idx]
        else:
            G4 = torch.zeros_like(G0)

        phase = G0 * torch.cos(self.theta) - G4 * torch.sin(self.theta)  # [B, C, num_targets]
        return torch.einsum('bct,tc->bt', phase, self.proj) + self.bias


class EEGNet(nn.Module):
    """Geometric EEG Emotion Classification Network.

    Architecture:
        MotherEmbedding (per region) → stack → GeometricTransformerBlock × N
        → CliffordLayerNorm → GeometricNeutralizer → mean pool
        → MultiTargetPhaseShiftHead → [B, num_targets]

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
        super().__init__()

        p, q = 3, 1
        if config is not None:
            if hasattr(config, 'algebra'):
                p = config.algebra.get('p', 3)
                q = config.algebra.get('q', 1)
            elif isinstance(config, dict):
                p = config.get('p', 3)
                q = config.get('q', 1)

        self.algebra = CliffordAlgebra(p, q, device=device)

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
            self.embeddings[name] = MotherEmbedding(
                self.algebra, size, channels, U, V
            )

        self.blocks = nn.ModuleList([
            GeometricTransformerBlock(
                self.algebra, channels, num_heads, num_rotors,
                dropout=dropout, use_entropy_gating=True, eta=eta, H_base=H_base,
            )
            for _ in range(num_layers)
        ])

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
