# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""GTM Explainability Analysis — post-training inspection tools (v4 PGA).

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

Standalone script:
    uv run python scripts/analyze_gtm.py --checkpoint gtm_arc_best.pt
"""

import math
import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra


class GTMAnalyzer:
    """Post-training analysis for pretrained GTM v4 models.

    Provides:
        - Instruction template decomposition (rotation + translation motors)
        - Color remapping table inspection
        - Cursor trajectory through both phases
        - Hypothesis selection analysis (scores, weights, temperature)
        - Write gate analysis (per-cell acceptance/rejection)
        - Rule memory analysis
        - Per-cell prediction vs target comparison
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.device = device

    @staticmethod
    def from_checkpoint(path: str, device: str = 'cpu') -> 'GTMAnalyzer':
        """Load GTMAnalyzer from a BaseTask checkpoint.

        Args:
            path: Path to checkpoint saved by BaseTask.save_checkpoint().
            device: Target device.

        Returns:
            GTMAnalyzer instance with loaded model.
        """
        from models.gtm import GTMNet

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint['config']
        mcfg = cfg.model
        act_cfg = mcfg.get('act', {})
        color_cfg = mcfg.get('color_unit', {})
        attn_cfg = mcfg.get('attention', {})

        algebra_cpu = CliffordAlgebra(3, 0, 1, device=device)
        algebra_ctrl = CliffordAlgebra(1, 1, 0, device=device)

        model = GTMNet(
            algebra_cpu=algebra_cpu,
            algebra_ctrl=algebra_ctrl,
            channels=mcfg.get('channels', 16),
            num_steps=mcfg.get('num_steps', 8),
            max_steps=mcfg.get('max_steps', 20),
            num_hypotheses=mcfg.get('num_hypotheses', 4),
            top_k=mcfg.get('top_k', 1),
            head_hidden=mcfg.get('head_hidden', 64),
            temperature_init=mcfg.get('gumbel_temperature', 1.0),
            use_act=act_cfg.get('enabled', True),
            lambda_p=act_cfg.get('lambda_p', 0.5),
            coord_scale=mcfg.get('coord_scale', 1.0),
            K_color=color_cfg.get('K_color', 4),
            num_attn_heads=attn_cfg.get('num_heads', 4),
            attn_head_dim=attn_cfg.get('head_dim', 8),
            num_rule_slots=mcfg.get('num_rule_slots', 8),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return GTMAnalyzer(model, device)

    # ------------------------------------------------------------------
    # Static analysis (no data required)
    # ------------------------------------------------------------------

    def analyze_instructions(self) -> dict:
        """Decompose instruction templates into geometric components.

        For each of the K trainable instruction templates in Cl(3,0,1):
          - Rotation bivectors (e01, e02, e12) -> rotation angle and plane
          - Translation bivectors (e03, e13, e23) -> translation vector
          - Scalar (grade-0) and pseudoscalar (grade-4) -> color control signals

        Returns:
            dict with keys per template index:
                'templates_raw':        [K, 16] raw parameter values
                'rotation_angles':      [K] angle in radians (= 2 * ||B_rot||)
                'rotation_planes':      [K, 3] unit bivector (e01, e02, e12)
                'rotation_degrees':     [K] angle in degrees
                'translation_vectors':  [K, 3] translation (e03, e13, e23) magnitudes
                'translation_norms':    [K] translation magnitude
                'color_control':        [K, 2] (grade-0, grade-4) values
                'near_identity':        [K] bool — True if template ~ no-op
        """
        templates = self._get_templates()  # [K, 16]
        K = templates.shape[0]

        # Rotation bivectors: e01(idx3), e02(idx5), e12(idx6)
        bv_rot = templates[:, [3, 5, 6]]  # [K, 3]
        bv_rot_norm = bv_rot.norm(dim=-1)  # [K]
        rotation_angles = 2.0 * bv_rot_norm

        safe_norm = bv_rot_norm.clamp(min=1e-8).unsqueeze(-1)
        rotation_planes = bv_rot / safe_norm

        # Translation bivectors: e03(idx9), e13(idx10), e23(idx12)
        bv_trans = templates[:, [9, 10, 12]]  # [K, 3]
        trans_norms = bv_trans.norm(dim=-1)    # [K]

        # Color control signals
        color_control = templates[:, [0, 15]]  # [K, 2] (grade-0, grade-4)

        # Near-identity: small rotation + small translation + small color signal
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

    def analyze_color_unit(self) -> dict:
        """Inspect ColorUnit remapping tables.

        Returns:
            dict with:
                'remap_tables': [K_color, 10, 10] learned tables
                'table_diag_dominance': [K_color] how close to identity each table is
        """
        # Get color unit from first step's search module
        color_unit = self.model.vm.steps[0].search.pga_cpu.color_unit
        tables = color_unit.remap_tables.detach()  # [K_color, 10, 10]

        # Diagonal dominance: fraction of mass on diagonal
        diags = torch.diagonal(tables, dim1=-2, dim2=-1)  # [K_color, 10]
        row_sums = tables.abs().sum(dim=-1)  # [K_color, 10]
        diag_dominance = (diags.abs() / row_sums.clamp(min=1e-8)).mean(dim=-1)

        return {
            'remap_tables': tables,
            'table_diag_dominance': diag_dominance,
        }

    def analyze_temperature(self) -> dict:
        """Analyze Gumbel-Softmax temperature across all steps.

        Returns:
            dict with:
                'temperatures': [num_steps] current temperature per step
                'is_sharp': [num_steps] bool — True if tau < 0.5 (near-discrete)
        """
        temps = []
        for step in self.model.vm.steps:
            tau = step.search.log_temperature.exp().clamp(0.1, 5.0)
            temps.append(tau.item())

        temps_t = torch.tensor(temps)
        return {
            'temperatures': temps_t,
            'is_sharp': temps_t < 0.5,
        }

    # ------------------------------------------------------------------
    # Dynamic analysis (requires a batch)
    # ------------------------------------------------------------------

    def analyze(self, batch: dict) -> dict:
        """Full analysis of one batch through both phases.

        Args:
            batch: Collated ARC batch from collate_arc.

        Returns:
            dict with:
                'instructions': instruction decomposition (static)
                'color_unit': color remapping analysis (static)
                'phase1': {cursors, search_scores, search_weights,
                           gate_values, halt_probs}
                'phase2': same structure as phase1
                'cursor_after_phase1': [B, 4]
                'cursor_after_phase2': [B, 4]
                'predictions': [B, N_test] predicted colors
                'targets': [B, N_test] ground truth
                'cell_accuracy': float
                'grid_correct': [B] bool per example
                'test_masks': [B, N_test] validity mask
        """
        num_steps = self.model.vm.num_steps

        # Run full forward with trace
        with torch.no_grad():
            result = self._run_forward(batch)

        logits = result['logits']
        preds = logits.argmax(dim=-1)
        trace = result['trace']

        # Split trace into Phase 1 and Phase 2
        phase1_trace = {k: v[:num_steps] for k, v in trace.items()}
        phase2_trace = {k: v[num_steps:] for k, v in trace.items()}

        # Targets
        test_outputs = batch['test_outputs'].to(self.device)
        test_masks = batch['test_masks'].to(self.device)
        B, H_max, W_max = test_outputs.shape
        targets = test_outputs.reshape(B, H_max * W_max)
        valid = test_masks.reshape(B, H_max * W_max)

        # Metrics
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
            'color_unit': self.analyze_color_unit(),
            'phase1': phase1_trace,
            'phase2': phase2_trace,
            'cursor_after_phase1': phase1_trace['cursors'][-1] if phase1_trace['cursors'] else None,
            'cursor_after_phase2': phase2_trace['cursors'][-1] if phase2_trace['cursors'] else None,
            'predictions': preds,
            'targets': targets,
            'cell_accuracy': cell_acc,
            'grid_correct': grid_correct,
            'test_masks': valid,
        }

    def predict(self, batch: dict) -> dict:
        """Lightweight prediction — just logits and accuracy.

        Args:
            batch: Collated ARC batch.

        Returns:
            dict with 'predictions', 'targets', 'cell_accuracy', 'grid_correct'.
        """
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

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

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

    def format_cursor_report(self, report: dict) -> str:
        """Human-readable cursor trajectory summary."""
        lines = ['=== Cursor Trajectory ===', '']

        # Cl(1,1) components: {1, e3, e4, e34}
        labels = ['scalar(confidence)', 'e3(hypothesis)', 'e4(depth)', 'e34(phase)']

        for phase_name, phase_key in [('Phase 1 (Rule Inference)', 'phase1'),
                                       ('Phase 2 (Rule Application)', 'phase2')]:
            cursors = report[phase_key]['cursors']
            if not cursors:
                continue
            lines.append(f'{phase_name}:')
            for t, cursor in enumerate(cursors):
                vals = cursor[0]  # first batch element
                components = '  '.join(f'{labels[j]}={vals[j]:+.4f}' for j in range(4))
                lines.append(f'  Step {t}: {components}')
            lines.append('')

        return '\n'.join(lines)

    def format_search_report(self, report: dict) -> str:
        """Human-readable hypothesis selection summary."""
        lines = ['=== Hypothesis Selection ===', '']

        for phase_name, phase_key in [('Phase 1', 'phase1'), ('Phase 2', 'phase2')]:
            weights_list = report[phase_key]['search_weights']
            if not weights_list:
                continue
            lines.append(f'{phase_name}:')
            for t, w in enumerate(weights_list):
                w0 = w[0]  # first batch element
                dominant = w0.argmax().item()
                w_str = '  '.join(f'H{k}={w0[k]:.3f}' for k in range(w0.shape[0]))
                lines.append(f'  Step {t}: [{w_str}]  dominant=H{dominant}')
            lines.append('')

        return '\n'.join(lines)

    def format_gate_report(self, report: dict) -> str:
        """Human-readable write gate summary."""
        lines = ['=== Write Gate Analysis ===', '']

        for phase_name, phase_key in [('Phase 1', 'phase1'), ('Phase 2', 'phase2')]:
            gates = report[phase_key]['gate_values']
            if not gates:
                continue
            lines.append(f'{phase_name}:')
            for t, g in enumerate(gates):
                g0 = g[0].squeeze(-1)  # [N] for first batch element
                lines.append(
                    f'  Step {t}: mean={g0.mean():.3f}  '
                    f'min={g0.min():.3f}  max={g0.max():.3f}  '
                    f'accept(>0.5)={(g0 > 0.5).float().mean():.1%}'
                )
            lines.append('')

        return '\n'.join(lines)

    def full_report(self, batch: dict) -> str:
        """Generate complete human-readable analysis report."""
        report = self.analyze(batch)

        sections = [
            self.format_instruction_report(),
            self.format_cursor_report(report),
            self.format_search_report(report),
            self.format_gate_report(report),
            '',
            '=== Prediction Summary ===',
            f'  Cell accuracy: {report["cell_accuracy"]:.4f}',
            f'  Grid correct:  {report["grid_correct"].sum().item()}/{report["grid_correct"].shape[0]}',
        ]
        return '\n'.join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_templates(self) -> torch.Tensor:
        """Get instruction templates from the first step (shared across steps)."""
        return self.model.vm.steps[0].search.instruction_templates.detach()

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
