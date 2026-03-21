# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Grid reconstruction head for ARC-AGI.

Per-cell color classification from final CPU state multivectors.
"""

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra


class GridReconstructionHead(nn.Module):
    """Per-cell color classification from Cl(3,0,1) multivectors.

    Maps each cell's 16-component multivector to 10-class color logits.
    Optionally masks out memory channels before decoding.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra, hidden_dim: int = 64,
                 num_memory_channels: int = 0):
        super().__init__()
        self.algebra = algebra_cpu
        self.num_memory_channels = num_memory_channels
        input_dim = algebra_cpu.dim - num_memory_channels
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),  # 10 ARC colors
        )

    def forward(self, cpu_state: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Predict per-cell color logits.

        Args:
            cpu_state: Final CPU state [B, N, D].
            mask: Optional validity mask [B, N] (True=valid). Not used in
                forward (handled by loss function ignore_index), but kept
                for interface compatibility.

        Returns:
            Logits [B, N, 10].
        """
        if self.num_memory_channels > 0:
            cpu_state = cpu_state[..., :-self.num_memory_channels]
        return self.mlp(cpu_state)
