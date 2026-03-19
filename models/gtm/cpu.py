# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""PGA Cl(3,0,1) computation engine: motor sandwich + color remapping."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra


class ColorUnit(nn.Module):
    """Position-conditioned discrete color remapping via K blended [10, 10] tables.

    Table selection conditioned on per-cell spatial features (grade-1 post-motor
    position) plus instruction grade-0/grade-4.  This breaks the fundamental
    GA bottleneck where the motor sandwich product leaves grade-0 (color)
    invariant — by conditioning the remap on post-motor position, each cell
    can receive a different color transformation.

    Spatial indices used:
        idx 1 (e0): row,  idx 2 (e1): col,  idx 8 (e3): homogeneous coord
    """

    # Grade-1 spatial component indices in Cl(3,0,1)
    _SPATIAL_IDX = [1, 2, 8]  # e0(row), e1(col), e3(homo)

    def __init__(self, K_color: int = 4):
        super().__init__()
        self.K_color = K_color
        self.remap_tables = nn.Parameter(
            torch.eye(10).unsqueeze(0).expand(K_color, -1, -1).clone()
            + torch.randn(K_color, 10, 10) * 0.01
        )
        # 2 (instruction g0 + g4) + 3 (cell spatial) = 5 inputs
        self.selector = nn.Linear(5, K_color)

    def forward(self, state: torch.Tensor,
                instruction: torch.Tensor) -> torch.Tensor:
        """Apply position-conditioned color remapping.

        Args:
            state: [L, N, 16] after motor transform.
            instruction: [L, 16].
        """
        L, N, D = state.shape

        # Per-cell spatial features from post-motor grade-1 components
        cell_spatial = state[:, :, self._SPATIAL_IDX]  # [L, N, 3]

        # Instruction features broadcast to every cell
        instr_feat = torch.stack(
            [instruction[:, 0], instruction[:, 15]], dim=-1,
        ).unsqueeze(1).expand(L, N, 2)  # [L, N, 2]

        sel_input = torch.cat([instr_feat, cell_spatial], dim=-1)  # [L, N, 5]
        table_weights = F.softmax(self.selector(sel_input), dim=-1)  # [L, N, K]

        # Per-cell blended remap table
        blended = torch.einsum(
            'lnk,kij->lnij', table_weights, self.remap_tables,
        )  # [L, N, 10, 10]

        raw_color = state[:, :, 0] * 9.0
        centers = torch.arange(10, device=state.device, dtype=state.dtype)
        diffs = raw_color.unsqueeze(-1) - centers
        soft_color = F.softmax(-4.0 * diffs.pow(2), dim=-1)  # [L, N, 10]

        # Per-cell remap: [L, N, 1, 10] @ [L, N, 10, 10] -> [L, N, 1, 10]
        remapped = torch.matmul(
            soft_color.unsqueeze(2), blended,
        ).squeeze(2)  # [L, N, 10]

        new_color = torch.einsum('lni,i->ln', remapped, centers) / 9.0
        new_occupancy = 1.0 - remapped[:, :, 0]

        out = state.clone()
        out[:, :, 0] = new_color
        out[:, :, 15] = new_occupancy
        return out


class GeometricCPU(nn.Module):
    """PGA Cl(3,0,1) computation engine.

    Bivectors e01/e02/e12 produce rotations; null bivectors e03/e13/e23
    produce translations. Both composed into a single motor via exp map.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra, K_color: int = 4):
        super().__init__()
        assert algebra_cpu.p == 3 and algebra_cpu.r == 1, \
            f"GeometricCPU requires Cl(3,0,1), got Cl({algebra_cpu.p},{algebra_cpu.q},{algebra_cpu.r})"
        self.algebra = algebra_cpu
        self.color_unit = ColorUnit(K_color)

    def _transform(self, state: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        """Motor sandwich + color remapping. [L, N, 16] -> [L, N, 16]."""
        L, N, D = state.shape

        bv = self.algebra.grade_projection(instruction, 2)
        M = self.algebra.exp(-0.5 * bv)
        M_rev = self.algebra.reverse(M)

        M_exp = M.unsqueeze(1).expand(L, N, D).reshape(L * N, D)
        M_rev_exp = M_rev.unsqueeze(1).expand(L, N, D).reshape(L * N, D)
        state_flat = state.reshape(L * N, 1, D)

        spatial_out = self.algebra.sandwich_product(
            M_exp, state_flat, M_rev_exp
        ).reshape(L, N, D)

        return self.color_unit(spatial_out, instruction)

    def execute(self, state: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        """Apply transform to [B, N, 16] state with [B, 16] instruction."""
        self.algebra.ensure_device(state.device)
        return self._transform(state, instruction)

    def execute_all(self, state: torch.Tensor,
                    instructions: torch.Tensor) -> torch.Tensor:
        """Execute K instructions batched. [B,N,16] x [B,K,16] -> [B,K,N,16]."""
        B, N, D = state.shape
        K = instructions.shape[1]
        self.algebra.ensure_device(state.device)

        state_exp = state.unsqueeze(1).expand(B, K, N, D).reshape(B * K, N, D)
        instr_flat = instructions.reshape(B * K, D)

        result = self._transform(state_exp, instr_flat)
        return result.reshape(B, K, N, D)
