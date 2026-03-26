# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Deterministic ARC grid <-> Cl(3,0,1) PGA multivector codec."""

import torch
from core.algebra import CliffordAlgebra


class GridCodec:
    """Deterministic encoder/decoder for ARC grids in PGA Cl(3,0,1).

    Proper PGA point encoding — each cell is a grade-1 element plus
    scalar color and pseudoscalar occupancy:

        idx 0  (1):     color / 9.0       (grade 0 — motor-invariant)
        idx 1  (e0):    row / (H-1)       (grade 1 — motor-transformable)
        idx 2  (e1):    col / (W-1)       (grade 1 — motor-transformable)
        idx 4  (e2):    reserved (role embed)
        idx 8  (e3):    1.0               (grade 1 — homogeneous coord)
        idx 15 (e0123): occupancy flag    (grade 4)

    Coordinates are normalized to [0, 1] by grid dimensions, then scaled
    by coord_scale.  This balances energy with color (also [0, 1]) and
    makes the motor operate in a grid-size-invariant coordinate system.

    Note: the old encoding placed row*col in idx 3 (e01, grade 2).
    This was removed because (a) it's not a proper PGA point component
    — points are grade-1 in PGA, (b) it dominated 89% of encoding energy,
    drowning out the color signal, (c) it confused the motor which treats
    grade-2 as bivectors (lines/planes).
    """

    def __init__(self, algebra_cpu: CliffordAlgebra, coord_scale: float = 1.0):
        assert algebra_cpu.p == 3 and algebra_cpu.r == 1, \
            f"GridCodec requires Cl(3,0,1), got Cl({algebra_cpu.p},{algebra_cpu.q},{algebra_cpu.r})"
        self.algebra = algebra_cpu
        self.coord_scale = coord_scale

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a single [H, W] grid into [H, W, 16] multivectors."""
        H, W = grid.shape
        device = grid.device
        cs = self.coord_scale

        mv = torch.zeros(H, W, 16, device=device, dtype=torch.float32)
        colors = grid.float()
        rows = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        cols = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)

        # Normalize coords to [0, 1] by grid dims for energy balance
        row_norm = rows / max(H - 1, 1)
        col_norm = cols / max(W - 1, 1)

        mv[:, :, 0] = colors / 9.0
        mv[:, :, 1] = row_norm * cs
        mv[:, :, 2] = col_norm * cs
        mv[:, :, 8] = 1.0
        mv[:, :, 15] = (colors > 0).float()

        return mv

    def encode_batch(self, grids: torch.Tensor,
                     masks: torch.Tensor) -> tuple:
        """Encode padded [B, H_max, W_max] grids into [B, N_max, 16] multivectors."""
        B, H_max, W_max = grids.shape
        N_max = H_max * W_max
        device = grids.device
        cs = self.coord_scale

        colors = grids.float()
        rows = torch.arange(H_max, device=device).float().view(1, H_max, 1).expand(B, H_max, W_max)
        cols = torch.arange(W_max, device=device).float().view(1, 1, W_max).expand(B, H_max, W_max)

        # Normalize coords to [0, 1] by grid dims for energy balance
        row_norm = rows / max(H_max - 1, 1)
        col_norm = cols / max(W_max - 1, 1)

        mv = torch.zeros(B, H_max, W_max, 16, device=device, dtype=torch.float32)
        mv[:, :, :, 0] = colors / 9.0
        mv[:, :, :, 1] = row_norm * cs
        mv[:, :, :, 2] = col_norm * cs
        mv[:, :, :, 8] = 1.0
        mv[:, :, :, 15] = (colors > 0).float()

        mv = mv * masks.unsqueeze(-1).float()
        mv = mv.reshape(B, N_max, 16)
        flat_masks = masks.reshape(B, N_max)

        return mv, flat_masks

    def decode(self, mv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Decode [H*W, 16] multivectors back to [H, W] integer grid."""
        flat = mv.reshape(-1, 16)
        colors = flat[:H * W, 0] * 9.0
        colors = colors.round().long().clamp(0, 9)
        return colors.reshape(H, W)
