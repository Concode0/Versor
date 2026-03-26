# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""GTM Explainability Analysis — post-training inspection tools.

Usage (from checkpoint):
    analyzer = GTMAnalyzer.from_checkpoint(
        'gtm_arc_best.pt', device='cuda'
    )

    # Static: what did each instruction template learn?
    instr = analyzer.analyze_instructions()

    # Dynamic: full analysis on a batch
    report = analyzer.analyze(batch)

Usage (from existing model):
    analyzer = GTMAnalyzer(model, device='cuda')
    report = analyzer.analyze(batch)
"""

import math
import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra


class GTMAnalyzer:
    """Post-training analysis for pretrained GTM models.

    Provides:
        - Instruction template decomposition (rotation + translation motors)
        - Color remapping table inspection
        - Hypothesis evolution and conviction analysis
        - Write gate analysis
        - FIM traces and information gain
        - Rule memory analysis
        - Per-cell prediction vs target comparison
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.device = device

    @staticmethod
    def from_checkpoint(path: str, device: str = 'cpu') -> 'GTMAnalyzer':
        """Load GTMAnalyzer from a BaseTask checkpoint."""
        from models.gtm import GTMNet

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint['config']
        mcfg = cfg.model
        attn_cfg = mcfg.get('attention', {})
        sp_cfg = mcfg.get('search_plane', {})
        lm_cfg = mcfg.get('log_manifold', {})
        ig_cfg = mcfg.get('info_geometry', {})
        ae_cfg = mcfg.get('action_engine', {})

        algebra_cpu = CliffordAlgebra(3, 0, 1, device=device)
        algebra_ctrl = CliffordAlgebra(1, 1, 0, device=device)

        model = GTMNet(
            algebra_cpu=algebra_cpu,
            algebra_ctrl=algebra_ctrl,
            channels=mcfg.get('channels', 32),
            num_steps=mcfg.get('num_steps', 12),
            max_steps=mcfg.get('max_steps', 24),
            num_hypotheses=mcfg.get('num_hypotheses', 8),
            head_hidden=mcfg.get('head_hidden', 128),
            coord_scale=mcfg.get('coord_scale', 1.0),
            num_attn_heads=attn_cfg.get('num_heads', 4),
            attn_head_dim=attn_cfg.get('head_dim', 8),
            num_rule_slots=mcfg.get('num_rule_slots', 8),
            num_memory_channels=mcfg.get('num_memory_channels', 4),
            weight_share_steps=mcfg.get('weight_share_steps', False),
            log_manifold_gate_init=lm_cfg.get('gate_init', -5.0),
            evolve_hidden=sp_cfg.get('evolve_hidden', 64),
            halt_eps=ig_cfg.get('halt_eps', 0.01),
            use_supervised_fim=ig_cfg.get('use_supervised_fim', True),
            action_gate_init=ae_cfg.get('gate_init', 0.0),
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return GTMAnalyzer(model, device)

    def analyze_instructions(self) -> dict:
        """Decompose instruction templates into geometric components.

        Returns:
            dict with rotation angles/planes, translation vectors, color control.
        """
        templates = self._get_templates()  # [K, 16]
        K = templates.shape[0]

        # Rotation bivectors: e01(idx3), e02(idx5), e12(idx6)
        bv_rot = templates[:, [3, 5, 6]]
        bv_rot_norm = bv_rot.norm(dim=-1)
        rotation_angles = 2.0 * bv_rot_norm

        safe_norm = bv_rot_norm.clamp(min=1e-8).unsqueeze(-1)
        rotation_planes = bv_rot / safe_norm

        # Translation bivectors: e03(idx9), e13(idx10), e23(idx12)
        bv_trans = templates[:, [9, 10, 12]]
        trans_norms = bv_trans.norm(dim=-1)

        # Color control signals
        color_control = templates[:, [0, 15]]

        near_identity = (
            (rotation_angles < 0.1) &
            (trans_norms < 0.05) &
            (color_control.abs().max(dim=-1).values < 0.05)
        )

        return {
            'templates_raw': templates,
            'rotation_angles': rotation_angles,
            'rotation_planes': rotation_planes,
            'rotation_degrees': rotation_angles * (180.0 / math.pi),
            'translation_vectors': bv_trans,
            'translation_norms': trans_norms,
            'color_control': color_control,
            'near_identity': near_identity,
        }

    def analyze_action_gate(self) -> dict:
        """Inspect per-component continuous/discrete blend."""
        # All steps share the same ActionEngine parameters if weight-shared
        step0 = self.model.world_model.steps[0]
        gate = torch.sigmoid(step0.action_engine.action_gate).detach()
        return {
            'gate_values': gate,
            'continuous_dominant': (gate > 0.5).sum().item(),
            'discrete_dominant': (gate <= 0.5).sum().item(),
        }

    def analyze_temperature(self) -> dict:
        """Inspect per-step softmax temperature from SearchPlane buffers."""
        temps = []
        for step in self.model.world_model.steps:
            tau = step.search_plane._temperature.item()
            temps.append(tau)
        return {
            'temperatures': temps,
            'is_sharp': [t < 0.1 for t in temps],
        }

    def analyze_hypothesis_init(self) -> dict:
        """Inspect initial hypothesis positions in Cl(1,1)."""
        h_init = self.model.world_model.hypothesis_init.detach()
        labels = ['scalar', 'e+', 'e-', 'e+e-']
        return {
            'hypothesis_init': h_init,
            'component_labels': labels,
        }

    def analyze(self, batch: dict) -> dict:
        """Full analysis of one batch through both phases."""
        with torch.no_grad():
            result = self._run_forward(batch)

        logits = result['logits']
        preds = logits.argmax(dim=-1)

        test_outputs = batch['test_outputs'].to(self.device)
        test_masks = batch['test_masks'].to(self.device)
        B, H_max, W_max = test_outputs.shape
        targets = test_outputs.reshape(B, H_max * W_max)
        valid = test_masks.reshape(B, H_max * W_max)

        matches = (preds == targets) & valid
        cell_acc = matches.sum().item() / max(valid.sum().item(), 1)

        grid_correct = torch.zeros(B, dtype=torch.bool)
        test_sizes = batch['test_sizes']
        for i in range(B):
            toH, toW = test_sizes[i]
            N = toH * toW
            grid_correct[i] = (preds[i, :N] == targets[i, :N]).all()

        return {
            'instructions': self.analyze_instructions(),
            'action_gate': self.analyze_action_gate(),
            'hypothesis_init': self.analyze_hypothesis_init(),
            'trace': result.get('trace'),
            'world_model_info': result.get('world_model_info'),
            'predictions': preds,
            'targets': targets,
            'cell_accuracy': cell_acc,
            'grid_correct': grid_correct,
            'test_masks': valid,
        }

    def predict(self, batch: dict) -> dict:
        """Lightweight prediction — just logits and accuracy."""
        with torch.no_grad():
            result = self._run_forward(batch)

        logits = result['logits']
        preds = logits.argmax(dim=-1)

        test_outputs = batch['test_outputs'].to(self.device)
        test_masks = batch['test_masks'].to(self.device)
        B, H_max, W_max = test_outputs.shape
        targets = test_outputs.reshape(B, H_max * W_max)
        valid = test_masks.reshape(B, H_max * W_max)

        matches = (preds == targets) & valid
        cell_acc = matches.sum().item() / max(valid.sum().item(), 1)

        grid_correct = torch.zeros(B, dtype=torch.bool)
        test_sizes = batch['test_sizes']
        for i in range(B):
            toH, toW = test_sizes[i]
            N = toH * toW
            grid_correct[i] = (preds[i, :N] == targets[i, :N]).all()

        return {
            'predictions': preds,
            'targets': targets,
            'cell_accuracy': cell_acc,
            'grid_correct': grid_correct,
        }

    def format_instruction_report(self) -> str:
        """Human-readable instruction template summary."""
        info = self.analyze_instructions()
        K = info['templates_raw'].shape[0]
        lines = ['=== Instruction Template Analysis (PGA) ===', '']

        for k in range(K):
            angle_deg = info['rotation_degrees'][k].item()
            plane = info['rotation_planes'][k]
            trans = info['translation_vectors'][k]
            trans_norm = info['translation_norms'][k].item()
            ctrl = info['color_control'][k]
            identity = info['near_identity'][k].item()

            lines.append(f'Template {k}:')
            lines.append(f'  Rotation:    {angle_deg:6.1f}deg  '
                         f'plane=({plane[0]:.2f}*e01 + {plane[1]:.2f}*e02 + {plane[2]:.2f}*e12)')
            lines.append(f'  Translation: |t|={trans_norm:.3f}  '
                         f'({trans[0]:.3f}*e03 + {trans[1]:.3f}*e13 + {trans[2]:.3f}*e23)')
            lines.append(f'  Color ctrl:  grade0={ctrl[0]:.3f}  pseudoscalar={ctrl[1]:.3f}')
            if identity:
                lines.append(f'  ** NEAR IDENTITY (no-op) **')
            lines.append('')

        return '\n'.join(lines)

    def full_report(self, batch: dict) -> str:
        """Generate complete human-readable analysis report."""
        report = self.analyze(batch)

        sections = [
            self.format_instruction_report(),
            '',
            '=== Action Gate ===',
            f'  Continuous-dominant components: {report["action_gate"]["continuous_dominant"]}/16',
            f'  Discrete-dominant components: {report["action_gate"]["discrete_dominant"]}/16',
            '',
            '=== Prediction Summary ===',
            f'  Cell accuracy: {report["cell_accuracy"]:.4f}',
            f'  Grid correct:  {report["grid_correct"].sum().item()}/{report["grid_correct"].shape[0]}',
        ]
        return '\n'.join(sections)

    def _get_templates(self) -> torch.Tensor:
        """Get instruction templates from the first WorldModel step."""
        return self.model.world_model.steps[0].action_engine.instruction_templates.detach()

    def _run_forward(self, batch: dict) -> dict:
        """Run model forward with trace, handling device transfer."""
        demo_inputs = batch['demo_inputs'].to(self.device)
        demo_outputs = batch['demo_outputs'].to(self.device)
        demo_masks = batch['demo_masks'].to(self.device)
        demo_output_masks = batch.get('demo_output_masks', demo_masks).to(self.device)
        test_inputs = batch['test_inputs'].to(self.device)
        test_masks = batch['test_masks'].to(self.device)
        num_demos = batch['num_demos'].to(self.device)

        return self.model(
            demo_inputs, demo_outputs, demo_masks,
            test_inputs, test_masks, num_demos,
            demo_output_masks=demo_output_masks,
            return_trace=True,
        )
