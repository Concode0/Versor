# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""GTMNet: Grid-native Geometric Turing Machine for ARC-AGI v4.

Two-phase few-shot pipeline with Rule Memory Bank:
  Phase 1 — Rule Inference:
    1. Encode demo (input,output) pairs -> PGA multivectors
    2. TuringVM processes demo cells -> cpu_state encodes patterns
    3. RuleAggregator compresses demo cpu_state into M rule slots
  Phase 2 — Rule Application:
    4. Encode test input -> PGA multivectors
    5. TuringVM processes test cells, using ctrl_cursor + rule_memory
    6. GridReconstructionHead predicts color logits

Information bridge: M rule slots * 16 dims = 128 floats (vs 4 floats before)
plus the 4D ctrl_cursor for halt control / step navigation.
"""

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from .grid_codec import GridCodec
from .turing_vm import TuringVM
from .heads import GridReconstructionHead
from .rule_memory import RuleAggregator


class GTMNet(nn.Module):
    """Grid-native Geometric Turing Machine Network.

    Two sub-algebras (Mother algebra removed):
        CPU Cl(3,0,1): PGA computation engine (motor + color)
        Control Cl(1,1): learnable search controller
    """

    def __init__(
        self,
        algebra_cpu: CliffordAlgebra,
        algebra_ctrl: CliffordAlgebra,
        channels: int = 16,
        num_steps: int = 8,
        max_steps: int = 20,
        num_hypotheses: int = 4,
        top_k: int = 1,
        head_hidden: int = 64,
        temperature_init: float = 1.0,
        use_act: bool = False,
        lambda_p: float = 0.5,
        coord_scale: float = 1.0,
        K_color: int = 4,
        num_attn_heads: int = 4,
        attn_head_dim: int = 8,
        num_rule_slots: int = 8,
    ):
        super().__init__()
        self.algebra_cpu = algebra_cpu
        self.algebra_ctrl = algebra_ctrl
        self.channels = channels

        D_cpu = algebra_cpu.dim  # 16

        # Grid codec (deterministic, no params)
        self.codec = GridCodec(algebra_cpu, coord_scale)

        # Learnable initial control cursor [4] in Cl(1,1)
        self.init_cursor = nn.Parameter(torch.randn(4) * 0.01)

        # Learnable role markers injected into geometrically reserved slots:
        #   idx 4 (e2): reserved auxiliary vector — never used by GridCodec
        #   idx 15 (pseudoscalar e0123): occupancy/role flag
        # Shape: [3, 2] for (e2_value, pseudoscalar_value) per role
        #   role 0 = demo input, role 1 = demo output, role 2 = test input
        self.role_embed = nn.Parameter(torch.randn(3, 2) * 0.01)

        # Rule Memory Aggregator
        self.rule_aggregator = RuleAggregator(
            d_cpu=D_cpu, num_slots=num_rule_slots, num_heads=num_attn_heads,
        )

        # Turing VM
        self.vm = TuringVM(
            algebra_cpu, algebra_ctrl,
            channels, num_steps, max_steps,
            num_hypotheses, top_k, temperature_init,
            use_act, lambda_p,
            num_attn_heads, attn_head_dim,
            K_color, num_rule_slots,
        )

        # Reconstruction head
        self.head = GridReconstructionHead(algebra_cpu, head_hidden)

    def forward(self, demo_inputs: torch.Tensor, demo_outputs: torch.Tensor,
                demo_masks: torch.Tensor,
                test_inputs: torch.Tensor,
                test_masks: torch.Tensor, num_demos: torch.Tensor,
                demo_output_masks: torch.Tensor = None,
                input_sizes: list = None,
                return_trace: bool = False) -> dict:
        """Two-phase forward pass: Rule Inference -> Rule Application.

        Phase 1 processes demo pairs through the VM to encode transformation
        patterns. RuleAggregator compresses these into rule_memory slots.
        Phase 2 processes test input using ctrl_cursor + rule_memory.

        Args:
            demo_inputs:  [B, K, H_max, W_max] padded demo input grids.
            demo_outputs: [B, K, H_max, W_max] padded demo output grids.
            demo_masks:   [B, K, H_max, W_max] bool (True=valid input cell).
            test_inputs:  [B, H_max, W_max] padded test input.
            test_masks:   [B, H_max, W_max] bool (True=valid).
            num_demos:    [B] int — actual demo count per example.
            demo_output_masks: [B, K, H_max, W_max] bool (True=valid output cell).
                If None, falls back to demo_masks (same dims assumed).
            input_sizes:  Optional list of (H, W) for test inputs.
            return_trace: Collect per-step diagnostics.

        Returns:
            dict with:
                'logits': [B, N_test, 10] color logits for test cells
                'test_flat_masks': [B, N_test] bool
                optionally 'act_info', 'trace'
        """
        B, K, H_max, W_max = demo_inputs.shape
        N_grid = H_max * W_max
        device = demo_inputs.device
        D_cpu = self.algebra_cpu.dim  # 16

        if demo_output_masks is None:
            demo_output_masks = demo_masks

        # --- Encode demo pairs ---
        di_flat = demo_inputs.reshape(B * K, H_max, W_max)
        do_flat = demo_outputs.clamp(min=0).reshape(B * K, H_max, W_max)
        dim_flat = demo_masks.reshape(B * K, H_max, W_max)
        dom_flat = demo_output_masks.reshape(B * K, H_max, W_max)

        di_mv, di_fm = self.codec.encode_batch(di_flat, dim_flat)  # [B*K, N_grid, 16]
        do_mv, do_fm = self.codec.encode_batch(do_flat, dom_flat)  # [B*K, N_grid, 16]

        # Add role markers into reserved slots
        di_mv[:, :, 4] = di_mv[:, :, 4] + self.role_embed[0, 0]
        di_mv[:, :, 15] = di_mv[:, :, 15] + self.role_embed[0, 1]
        do_mv[:, :, 4] = do_mv[:, :, 4] + self.role_embed[1, 0]
        do_mv[:, :, 15] = do_mv[:, :, 15] + self.role_embed[1, 1]

        # Interleave demo input + output: [B*K, 2*N_grid, 16]
        demo_mv = torch.cat([di_mv, do_mv], dim=1)
        demo_fm = torch.cat([di_fm, do_fm], dim=1)

        # Reshape: [B, K * 2 * N_grid, 16]
        N_demo_per_pair = 2 * N_grid
        demo_mv = demo_mv.reshape(B, K * N_demo_per_pair, D_cpu)
        demo_fm = demo_fm.reshape(B, K * N_demo_per_pair)

        # Mask out unused demo pairs — vectorized, no .item() calls
        total_demo_len = K * N_demo_per_pair
        pos_idx = torch.arange(total_demo_len, device=device).unsqueeze(0)  # [1, L]
        limit = (num_demos * N_demo_per_pair).unsqueeze(1)                  # [B, 1]
        valid_demo = pos_idx < limit                                        # [B, L]
        demo_mv = demo_mv * valid_demo.unsqueeze(-1).float()
        demo_fm = demo_fm & valid_demo

        # --- Encode test input ---
        test_mv, test_fm = self.codec.encode_batch(test_inputs, test_masks)
        test_mv[:, :, 4] = test_mv[:, :, 4] + self.role_embed[2, 0]
        test_mv[:, :, 15] = test_mv[:, :, 15] + self.role_embed[2, 1]

        # --- Init control cursor ---
        ctrl_cursor = self.init_cursor.unsqueeze(0).expand(B, -1).clone()

        # === Phase 1: Rule Inference (demo only) ===
        # VM processes demo pairs -> cpu_state encodes patterns, ctrl_cursor updated
        demo_state, ctrl_cursor, act_info_demo, trace_demo = self.vm(
            demo_mv, ctrl_cursor, demo_fm, return_trace,
        )

        # Compress demo state into rule memory slots
        rule_memory = self.rule_aggregator(demo_state, demo_fm)  # [B, M, 16]

        # === Phase 2: Rule Application (test only) ===
        # VM processes test input using ctrl_cursor + rule_memory from Phase 1
        test_state, ctrl_cursor, act_info_test, trace_test = self.vm(
            test_mv, ctrl_cursor, test_fm, return_trace,
            rule_memory=rule_memory,
        )

        # --- Decode ---
        logits = self.head(test_state, test_fm)  # [B, N_grid, 10]

        result = {
            'logits': logits,
            'test_flat_masks': test_fm,
        }

        # ACT info: combine KL loss from both phases
        if act_info_test is not None:
            result['act_info'] = {
                'kl_loss': act_info_test['kl_loss'] + act_info_demo['kl_loss'],
                'expected_steps': act_info_test['expected_steps'],
                'weights': act_info_test['weights'],
            }

        # Merge traces from both phases
        if return_trace:
            trace_keys = ['search_scores', 'search_weights', 'halt_probs',
                          'cursors', 'gate_values']
            trace = {k: [] for k in trace_keys}
            for t in (trace_demo, trace_test):
                if t is not None:
                    for k in trace_keys:
                        trace[k].extend(t.get(k, []))
            result['trace'] = trace

        return result

    def set_temperature(self, tau: float):
        """Set Gumbel-Softmax temperature for all VM steps."""
        self.vm.set_temperature(tau)

    def freeze_vm(self):
        """Freeze all VM parameters (Phase 1: warmup)."""
        for param in self.vm.parameters():
            param.requires_grad = False

    def unfreeze_vm(self):
        """Unfreeze all VM parameters (Phase 2+)."""
        for param in self.vm.parameters():
            param.requires_grad = True

    def enable_act(self):
        """Enable adaptive computation time."""
        if self.vm.adaptive_halt is not None:
            self.vm.use_act = True

    def disable_act(self):
        """Disable adaptive computation time."""
        self.vm.use_act = False

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
