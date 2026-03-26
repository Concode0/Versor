# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Turing Machine Task.

Few-shot format: each training example = (demo_pairs, test_input, test_output).
The model sees K demo (input,output) pairs to infer the rule, then applies
it to a test input to produce the test output.

Three-phase training: warmup (WorldModel frozen), world model training
(ortho + gate entropy losses), FIM halt + conviction collapse.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from models.gtm import GTMNet
from models.gtm.search_plane import SearchPlane
from log import get_logger

logger = get_logger(__name__)


def _gate_entropy_loss(gate_values: torch.Tensor) -> torch.Tensor:
    """Entropy of write gate values — encourages decisive gating."""
    eps = 1e-8
    p = gate_values.mean(dim=(0, 1))  # [D] average gate per component
    entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    return entropy.mean()


class GTMTask(BaseTask):
    """Geometric Turing Machine task for ARC-AGI."""

    def __init__(self, cfg):
        # Training phase config
        self.warmup_epochs = cfg.training.get('warmup_epochs', 8)
        self.trim_epochs = cfg.training.get('trim_epochs', 72)
        self.act_epochs = cfg.training.get('act_epochs', 70)
        self.grad_clip = cfg.training.get('grad_clip', 1.0)
        self.eval_every = cfg.training.get('eval_every', 5)

        # Temperature schedule
        self.tau_start = cfg.training.get('tau_start', 1.0)
        self.tau_mid = cfg.training.get('tau_mid', 0.5)
        self.tau_end = cfg.training.get('tau_end', 0.05)

        # Loss weights
        self.ortho_weight = cfg.training.get('ortho_weight', 0.005)
        self.gate_entropy_weight = cfg.training.get('gate_entropy_weight', 0.01)
        self.info_gain_weight = cfg.training.get('info_gain_weight', 0.01)

        super().__init__(cfg)

    def setup_algebra(self):
        """Initialize CPU and Control algebras."""
        self.algebra_cpu = CliffordAlgebra(3, 0, 1, device=self.device)
        self.algebra_ctrl = CliffordAlgebra(1, 1, 0, device=self.device)
        return self.algebra_cpu

    def setup_model(self):
        mcfg = self.cfg.model
        attn_cfg = mcfg.get('attention', {})
        sp_cfg = mcfg.get('search_plane', {})
        lm_cfg = mcfg.get('log_manifold', {})
        ig_cfg = mcfg.get('info_geometry', {})
        ae_cfg = mcfg.get('action_engine', {})

        return GTMNet(
            algebra_cpu=self.algebra_cpu,
            algebra_ctrl=self.algebra_ctrl,
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
            gradient_horizon=mcfg.get('gradient_horizon', 2),
        )

    def _setup_optimizer(self):
        """Override to use only trainable parameters."""
        opt_type = self.cfg.training.get('optimizer_type', 'riemannian_adam')
        lr = self.cfg.training.lr
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not trainable_params:
            return torch.optim.SGD([torch.zeros(1)], lr=lr)

        if opt_type == 'riemannian_adam':
            from optimizers.riemannian import RiemannianAdam
            return RiemannianAdam(
                trainable_params, lr=lr,
                betas=self.cfg.training.get('betas', (0.9, 0.999)),
                algebra=self.algebra,
                max_bivector_norm=self.cfg.training.get('max_bivector_norm', 10.0),
            )
        elif opt_type == 'exponential_sgd':
            from optimizers.riemannian import ExponentialSGD
            return ExponentialSGD(
                trainable_params, lr=lr,
                momentum=self.cfg.training.get('momentum', 0.9),
                algebra=self.algebra,
                max_bivector_norm=self.cfg.training.get('max_bivector_norm', 10.0),
            )
        else:
            return torch.optim.AdamW(trainable_params, lr=lr)

    def setup_criterion(self):
        return nn.CrossEntropyLoss(ignore_index=-1)

    def get_data(self):
        from datalib.arc import get_arc_loaders
        dcfg = self.cfg.dataset
        loaders = get_arc_loaders(
            data_dir=dcfg.get('data_dir', 'data/arc'),
            batch_size=self.cfg.training.batch_size,
            include_toy=dcfg.get('include_toy', True),
            toy_n_examples=dcfg.get('toy_n_examples', 5000),
            toy_max_grid_size=dcfg.get('toy_max_grid_size', 10),
            num_workers=self.cfg.training.get('num_workers', 0),
            num_demos=dcfg.get('num_demos', 3),
            pin_memory=self.device_config.pin_memory,
            epoch_samples=dcfg.get('epoch_samples', 0),
        )
        return loaders['train'], loaders['val']

    def _run_model(self, batch, return_trace=False):
        """Run model on a few-shot batch."""
        demo_inputs = batch['demo_inputs'].to(self.device)
        demo_outputs = batch['demo_outputs'].to(self.device)
        demo_masks = batch['demo_masks'].to(self.device)
        demo_output_masks = batch['demo_output_masks'].to(self.device)
        test_inputs = batch['test_inputs'].to(self.device)
        test_masks = batch['test_masks'].to(self.device)
        num_demos = batch['num_demos'].to(self.device)

        # Pass targets for supervised FIM during training
        test_targets = None
        if self.model.training and 'test_outputs' in batch:
            test_targets = batch['test_outputs'].to(self.device)

        return self.model(
            demo_inputs, demo_outputs, demo_masks,
            test_inputs, test_masks, num_demos,
            demo_output_masks=demo_output_masks,
            test_targets=test_targets,
            input_sizes=batch.get('input_sizes'),
            return_trace=return_trace,
        )

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)

        need_trace = (self._phase >= 2 and self.gate_entropy_weight > 0)
        result = self._run_model(batch, return_trace=need_trace)

        logits = result['logits']  # [B, N_grid, 10]

        # Target
        test_outputs = batch['test_outputs'].to(self.device)
        B, H_max, W_max = test_outputs.shape
        targets = test_outputs.reshape(B, H_max * W_max)

        loss = self.criterion(
            logits.reshape(-1, 10),
            targets.reshape(-1),
        )

        wm_info = result.get('world_model_info', {})

        ortho_loss = torch.tensor(0.0, device=self.device)
        if self._phase >= 2 and self.ortho_weight > 0:
            # Get Gram from last step's search info via trace
            if 'trace' in result and result['trace'] is not None:
                test_trace = result['trace'].get('test')
                if test_trace and test_trace['search_info']:
                    last_search = test_trace['search_info'][-1]
                    gram = last_search.get('gram')
                    if gram is not None:
                        ortho_loss = SearchPlane.orthogonality_loss(gram)
                        loss = loss + self.ortho_weight * ortho_loss

        gate_ent = torch.tensor(0.0, device=self.device)
        if need_trace and 'trace' in result and result['trace'] is not None:
            test_trace = result['trace'].get('test')
            if test_trace and test_trace['gate_values']:
                ent_sum = sum(_gate_entropy_loss(g) for g in test_trace['gate_values'])
                gate_ent = ent_sum / len(test_trace['gate_values'])
                loss = loss + self.gate_entropy_weight * gate_ent

        # Penalize negative info gain (monotonic progress)
        info_loss = torch.tensor(0.0, device=self.device)
        if self._phase >= 3 and self.info_gain_weight > 0:
            step_deltas = wm_info.get('step_deltas', [])
            if step_deltas:
                all_deltas = torch.stack(
                    [(d).mean() for d in step_deltas]
                )
                # Penalize negative information gain (want monotonic progress)
                info_loss = torch.relu(-all_deltas).mean()
                loss = loss + self.info_gain_weight * info_loss

        self._backward(loss)

        # Unscale BEFORE grad clip so clipping operates on real gradient magnitudes.
        # Without this, AMP's scale factor (65536) makes the effective clip ~1e-5.
        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)

        if self.grad_clip > 0:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)

        self._optimizer_step()

        logs = {'Loss': loss.item()}
        if ortho_loss.item() > 0:
            logs['Ortho'] = ortho_loss.item()
        if gate_ent.item() != 0:
            logs['GateEnt'] = gate_ent.item()
        if info_loss.item() > 0:
            logs['InfoGain'] = info_loss.item()
        return loss.item(), logs

    def evaluate(self, val_loader):
        self.model.eval()
        cell_correct = 0
        cell_total = 0
        grid_correct = 0
        grid_total = 0

        with torch.no_grad():
            for batch in val_loader:
                result = self._run_model(batch)
                logits = result['logits']
                preds = logits.argmax(dim=-1)

                test_outputs = batch['test_outputs'].to(self.device)
                test_masks = batch['test_masks'].to(self.device)
                B, H_max, W_max = test_outputs.shape
                targets = test_outputs.reshape(B, H_max * W_max)
                valid = test_masks.reshape(B, H_max * W_max)

                matches = (preds == targets) & valid
                cell_correct += matches.sum().item()
                cell_total += valid.sum().item()

                test_sizes = batch['test_sizes']
                for i in range(B):
                    toH, toW = test_sizes[i]
                    N = toH * toW
                    if (preds[i, :N] == targets[i, :N]).all():
                        grid_correct += 1
                    grid_total += 1

        cell_acc = cell_correct / max(cell_total, 1)
        grid_acc = grid_correct / max(grid_total, 1)
        logger.info("Cell accuracy: %.4f | Grid accuracy: %.4f (%d/%d)",
                     cell_acc, grid_acc, grid_correct, grid_total)
        return {'cell_accuracy': cell_acc, 'grid_accuracy': grid_acc}

    def visualize(self, val_loader):
        pass

    def run(self):
        """Three-phase training loop with FIM-based computation budget."""
        logger.info("Starting GTM training")
        train_loader, val_loader = self.get_data()

        total_epochs = self.warmup_epochs + self.trim_epochs + self.act_epochs
        self.epochs = total_epochs

        self._phase = 0
        best_val_metric = 0.0
        metric_key = 'cell_accuracy'

        pbar = tqdm(range(total_epochs))

        for epoch in pbar:
            if epoch < self.warmup_epochs:
                phase = 1
            elif epoch < self.warmup_epochs + self.trim_epochs:
                phase = 2
            else:
                phase = 3

            if phase != self._phase:
                prev_phase = self._phase
                self._phase = phase
                if phase == 1:
                    logger.info("Phase 1: Warmup (WorldModel frozen)")
                    self.model.freeze_world_model()
                    self.model.disable_fim_halt()
                elif phase == 2:
                    logger.info("Phase 2: World Model Training")
                    self.model.unfreeze_world_model()
                    self.model.disable_fim_halt()
                elif phase == 3:
                    logger.info("Phase 3: FIM Halt + Conviction Collapse")
                    self.model.enable_fim_halt()

                min_lr = self.cfg.training.get('min_lr', 1e-5)

                # LR handling per phase transition
                prev_lr = (self.optimizer.param_groups[0]['lr']
                           if prev_phase > 0 else self.cfg.training.lr)
                self.optimizer = self._setup_optimizer()
                if phase == 3 and prev_phase == 2:
                    # Reset to a viable LR floor in case scheduler killed it
                    phase3_lr = max(self.cfg.training.lr * 0.1, min_lr)
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = phase3_lr
                elif prev_phase > 0:
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = prev_lr
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5,
                    min_lr=min_lr)

            # Temperature schedule
            if phase == 1:
                tau = self.tau_start
            elif phase == 2:
                progress = min(1.0, (epoch - self.warmup_epochs) / max(self.trim_epochs, 1))
                tau = self.tau_start + (self.tau_mid - self.tau_start) * progress
            else:  # phase 3
                act_epoch = epoch - (self.warmup_epochs + self.trim_epochs)
                progress = min(1.0, act_epoch / max(self.act_epochs, 1))
                tau = self.tau_mid + (self.tau_end - self.tau_mid) * progress
            self.model.set_temperature(tau)

            # Phase 2 LR warmup: ramp from 1/10 to capped peak over first 10 epochs.
            # Caps at phase2_lr_scale * base_lr to prevent instability in 12-step model.
            if phase == 2:
                phase2_epoch = epoch - self.warmup_epochs
                warmup_len = min(10, max(self.trim_epochs // 4, 1))
                if phase2_epoch < warmup_len:
                    base_lr = self.cfg.training.lr
                    phase2_scale = self.cfg.training.get('phase2_lr_scale', 0.5)
                    peak_lr = base_lr * phase2_scale
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = peak_lr * (phase2_epoch + 1) / warmup_len

            # FIM mixing ramp: gradually blend in FIM-weighted output over Phase 3
            if phase == 3:
                act_epoch = epoch - (self.warmup_epochs + self.trim_epochs)
                self.model.world_model.fim_mix_ramp = min(
                    1.0, act_epoch / max(self.act_epochs * 0.5, 1))
            else:
                self.model.world_model.fim_mix_ramp = 0.0

            # Training
            self.model.train()
            total_loss = 0
            n_batches = 0
            for batch in train_loader:
                loss, _ = self.train_step(batch)
                total_loss += loss
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.scheduler.step(avg_loss)

            # Validation
            do_eval = (
                epoch < self.warmup_epochs or
                phase == 3 or
                (epoch + 1) % self.eval_every == 0 or
                epoch == total_epochs - 1
            )

            if do_eval:
                val_metrics = self.evaluate(val_loader)
                val_metric = val_metrics.get(metric_key, 0.0)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    self.save_checkpoint("gtm_arc_best.pt")
            else:
                val_metric = best_val_metric

            display = {
                'P': phase, 'Loss': avg_loss,
                metric_key: val_metric,
                'LR': self.optimizer.param_groups[0]['lr'],
                'tau': tau,
            }
            desc = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in display.items()
            )
            pbar.set_description(desc)

        logger.info("Training complete. Best %s: %.4f", metric_key, best_val_metric)

        self.load_checkpoint("gtm_arc_best.pt")
        final_metrics = self.evaluate(val_loader)
        logger.info("Final metrics: %s", final_metrics)
        return final_metrics
