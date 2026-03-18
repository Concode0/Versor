# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""PGA Motor CPU + ColorUnit — Cl(3,0,1) computation engine.

Core operations (three-part transform):
  Part A — Motor transform: M = exp(-grade_2(instr)/2), X' = MXM~
    The 6 bivectors split into 3 rotation (e01,e02,e12) and
    3 translation (e03,e13,e23) components. The parabolic exp branch
    in core/algebra.py handles null bivectors: exp(t*e03) = 1 + t*e03.
  Part B — ColorUnit: discrete color remapping conditioned on instruction
    K_color learnable tables [K_color, 10, 10], selected by grade-0 + grade-4.
  Part C — Merge: spatial from motor, color from ColorUnit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra


class ColorUnit(nn.Module):
    """Discrete color remapping conditioned on instruction.

    K_color learnable remapping tables [K_color, 10, 10].
    Instruction's grade-0 and grade-4 select and blend tables.
    """

    def __init__(self, K_color: int = 4):
        super().__init__()
        self.K_color = K_color
        # Initialize as near-identity: eye(10) + small noise per table
        self.remap_tables = nn.Parameter(
            torch.eye(10).unsqueeze(0).expand(K_color, -1, -1).clone()
            + torch.randn(K_color, 10, 10) * 0.01
        )
        # Selector: grade-0 (idx 0) + grade-4 (idx 15) → table weights
        self.selector = nn.Linear(2, K_color)

    def forward(self, state: torch.Tensor,
                instruction: torch.Tensor) -> torch.Tensor:
        """Apply color remapping to grade-0 and update occupancy.

        Args:
            state: [L, N, 16] PGA multivectors after motor transform.
            instruction: [L, 16] instruction multivectors.

        Returns:
            [L, N, 16] state with grade-0 (color) and grade-4 (occupancy) updated.
        """
        L, N, D = state.shape

        # Extract selector features from instruction
        sel_input = torch.stack([instruction[:, 0], instruction[:, 15]], dim=-1)  # [L, 2]
        table_weights = F.softmax(self.selector(sel_input), dim=-1)  # [L, K_color]

        # Blend remapping tables: [L, 10, 10]
        # table_weights: [L, K] @ remap_tables: [K, 10, 10] -> [L, 10, 10]
        blended = torch.einsum('lk,kij->lij', table_weights, self.remap_tables)

        # Extract current color: grade-0 → soft 10-class
        raw_color = state[:, :, 0] * 9.0  # [L, N] in [0, 9] range
        # Create soft one-hot via distance to each integer class
        centers = torch.arange(10, device=state.device, dtype=state.dtype)  # [10]
        # Soft assignment: exp(-4 * (color - center)^2)
        diffs = raw_color.unsqueeze(-1) - centers  # [L, N, 10]
        soft_color = F.softmax(-4.0 * diffs.pow(2), dim=-1)  # [L, N, 10]

        # Apply remapping: [L, N, 10] @ [L, 10, 10] -> [L, N, 10]
        remapped = torch.bmm(
            soft_color.reshape(L, N, 10),
            blended
        )  # [L, N, 10]

        # Convert back to scalar: expected value / 9.0
        new_color = torch.einsum('lni,i->ln', remapped, centers) / 9.0  # [L, N]

        # Update occupancy flag (grade-4 pseudoscalar idx 15)
        new_occupancy = 1.0 - remapped[:, :, 0]  # prob of NOT being color 0

        # Construct output: only modify grade-0 and grade-4
        out = state.clone()
        out[:, :, 0] = new_color
        out[:, :, 15] = new_occupancy

        return out


class GeometricCPU(nn.Module):
    """PGA Cl(3,0,1) computation engine with Motor + ColorUnit.

    The motor transform handles both rotation (e01,e02,e12 bivectors)
    and translation (e03,e13,e23 null bivectors) via a single sandwich product.
    The ColorUnit handles discrete color remapping.
    """

    def __init__(self, algebra_cpu: CliffordAlgebra, K_color: int = 4):
        super().__init__()
        assert algebra_cpu.p == 3 and algebra_cpu.r == 1, \
            f"GeometricCPU requires Cl(3,0,1), got Cl({algebra_cpu.p},{algebra_cpu.q},{algebra_cpu.r})"
        self.algebra = algebra_cpu
        self.color_unit = ColorUnit(K_color)

    def _transform(self, state: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        """Core transform: PGA motor sandwich + color remapping.

        Args:
            state: [L, N, 16] — L can be B (single) or B*K (batched).
            instruction: [L, 16].

        Returns:
            [L, N, 16] transformed state.
        """
        L, N, D = state.shape

        # Part A: Motor Transform (rotation + translation via PGA sandwich)
        bv = self.algebra.grade_projection(instruction, 2)  # [L, 16]
        M = self.algebra.exp(-0.5 * bv)  # [L, 16] — motor (rotation + translation)
        M_rev = self.algebra.reverse(M)  # [L, 16]

        M_exp = M.unsqueeze(1).expand(L, N, D).reshape(L * N, D)
        M_rev_exp = M_rev.unsqueeze(1).expand(L, N, D).reshape(L * N, D)
        state_flat = state.reshape(L * N, 1, D)

        spatial_out = self.algebra.sandwich_product(
            M_exp, state_flat, M_rev_exp
        ).reshape(L, N, D)

        # Part B: Color Remapping (grade-0 and grade-4 only)
        color_out = self.color_unit(spatial_out, instruction)

        return color_out

    def execute(self, state: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        """Apply PGA Motor + ColorUnit to state.

        Args:
            state: CPU state [B, N, 16] — per-cell multivectors.
            instruction: Instruction multivector [B, 16].

        Returns:
            New state [B, N, 16].
        """
        self.algebra.ensure_device(state.device)
        return self._transform(state, instruction)

    def execute_all(self, state: torch.Tensor,
                    instructions: torch.Tensor) -> torch.Tensor:
        """Execute K instructions in a single batched call.

        Reshapes [B, N, 16] x [B, K, 16] into [B*K, N, 16] x [B*K, 16],
        runs one _transform call, then reshapes back to [B, K, N, 16].

        Args:
            state: CPU state [B, N, 16].
            instructions: K instruction multivectors [B, K, 16].

        Returns:
            Tensor [B, K, N, 16] — all K outcomes stacked.
        """
        B, N, D = state.shape
        K = instructions.shape[1]
        self.algebra.ensure_device(state.device)

        # Expand state for all K instructions: [B, K, N, D] -> [B*K, N, D]
        state_exp = state.unsqueeze(1).expand(B, K, N, D).reshape(B * K, N, D)
        instr_flat = instructions.reshape(B * K, D)

        result = self._transform(state_exp, instr_flat)  # [B*K, N, D]
        return result.reshape(B, K, N, D)
