# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Deterministic ARC grid <-> Cl(3,0,1) PGA multivector codec.

Grids are kept as 2D tensors [H, W] (or padded [B, H_max, W_max]).
Row and column are directly read from 2D indices — no flattening required.

PGA encoding (each cell -> 1 multivector in Cl(3,0,1), dim=16):
  Grade-0 (scalar, idx 0):       color / 9.0 (invariant under all sandwich products)
  Grade-1 (vectors):
    idx 1 (e0): row (integer)    — spatial position
    idx 2 (e1): col (integer)    — spatial position
    idx 4 (e2): 0.0              — reserved for role embed / auxiliary
    idx 8 (e3): 1.0              — homogeneous coord (enables PGA translation)
  Grade-2 (bivectors):
    idx 3 (e01): row * col       — spatial correlation
  Grade-4 (pseudoscalar):
    idx 15 (e0123): 1.0 if non-background (color!=0), else 0.0

Integer coordinates: no max_grid_size normalization. CliffordLayerNorm
in the VM handles normalization across steps.
"""

import torch
from core.algebra import CliffordAlgebra


class GridCodec:
    """Deterministic encoder/decoder for ARC grids. No learnable parameters.

    Operates on 2D grids [H, W] or batched [B, H_max, W_max] with masks.
    Uses PGA Cl(3,0,1) with dim=16.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra, coord_scale: float = 1.0):
        assert algebra_cpu.p == 3 and algebra_cpu.r == 1, \
            f"GridCodec requires Cl(3,0,1), got Cl({algebra_cpu.p},{algebra_cpu.q},{algebra_cpu.r})"
        self.algebra = algebra_cpu
        self.coord_scale = coord_scale

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a single 2D grid into multivectors.

        Args:
            grid: Integer grid [H, W] with values in [0, 9].

        Returns:
            Multivectors [H, W, 16] in Cl(3,0,1).
        """
        H, W = grid.shape
        device = grid.device
        cs = self.coord_scale

        mv = torch.zeros(H, W, 16, device=device, dtype=torch.float32)
        colors = grid.float()

        # Row and col coordinate grids (integer, no normalization)
        rows = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        cols = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)

        # Grade-0 (idx 0): normalized color
        mv[:, :, 0] = colors / 9.0

        # Grade-1 (idx 1=e0, 2=e1, 4=e2, 8=e3): spatial position + homogeneous
        mv[:, :, 1] = rows * cs
        mv[:, :, 2] = cols * cs
        # idx 4 (e2) left zero — reserved for auxiliary features / role embed
        mv[:, :, 8] = 1.0  # e3 homogeneous coord (enables PGA translations)

        # Grade-2 (idx 3=e01): spatial correlation
        mv[:, :, 3] = rows * cols * (cs * cs)

        # Grade-4 pseudoscalar (idx 15=e0123): occupancy flag
        mv[:, :, 15] = (colors > 0).float()

        return mv

    def encode_batch(self, grids: torch.Tensor,
                     masks: torch.Tensor) -> tuple:
        """Encode a batch of padded 2D grids into flat multivector sequences.

        Args:
            grids: Padded grids [B, H_max, W_max] (long).
            masks: Validity masks [B, H_max, W_max] (bool).

        Returns:
            Tuple of:
                mv: [B, N_max, 16] flattened multivectors (N_max = H_max * W_max)
                flat_masks: [B, N_max] bool
        """
        B, H_max, W_max = grids.shape
        N_max = H_max * W_max
        device = grids.device
        cs = self.coord_scale

        colors = grids.float()
        rows = torch.arange(H_max, device=device).float().view(1, H_max, 1).expand(B, H_max, W_max)
        cols = torch.arange(W_max, device=device).float().view(1, 1, W_max).expand(B, H_max, W_max)

        mv = torch.zeros(B, H_max, W_max, 16, device=device, dtype=torch.float32)
        mv[:, :, :, 0] = colors / 9.0
        mv[:, :, :, 1] = rows * cs
        mv[:, :, :, 2] = cols * cs
        mv[:, :, :, 8] = 1.0  # e3 homogeneous coord
        mv[:, :, :, 3] = rows * cols * (cs * cs)
        mv[:, :, :, 15] = (colors > 0).float()

        # Zero out padding cells
        mv = mv * masks.unsqueeze(-1).float()

        # Flatten spatial dims: [B, H_max, W_max, 16] -> [B, N_max, 16]
        mv = mv.reshape(B, N_max, 16)
        flat_masks = masks.reshape(B, N_max)

        return mv, flat_masks

    def decode(self, mv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Decode multivectors back to a 2D grid.

        Args:
            mv: Multivectors [H*W, 16] or [H, W, 16].
            H: Grid height.
            W: Grid width.

        Returns:
            Integer grid [H, W] with values in [0, 9].
        """
        flat = mv.reshape(-1, 16)
        colors = flat[:H * W, 0] * 9.0
        colors = colors.round().long().clamp(0, 9)
        return colors.reshape(H, W)
