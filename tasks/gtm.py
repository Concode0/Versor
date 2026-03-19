# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Turing Machine Task — ARC-AGI v4.

Few-shot format: each training example = (demo_pairs, test_input, test_output).
The model sees K demo (input,output) pairs to infer the rule, then applies
it to a test input to produce the test output.

Three-phase training (anti-lazy-optimization):
1. Warmup: freeze VM, train head + init_cursor + role_embed
2. Circuit Search: unfreeze VM, fixed steps, gate entropy loss
3. ACT: enable adaptive computation, KL ramp-up

Two algebras (Mother algebra removed):
  CPU Cl(3,0,1): PGA computation engine (motor + color)
  Control Cl(1,1): learnable search
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from models.gtm import GTMNet
from log import get_logger

logger = get_logger(__name__)


def _gate_entropy_loss(scores: torch.Tensor) -> torch.Tensor:
    """Entropy of search scores — minimizing this encourages instruction specialization."""
    eps = 1e-8
    probs = torch.softmax(scores, dim=-1)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return entropy.mean()


class GTMTask(BaseTask):
    """Geometric Turing Machine task for ARC-AGI v4."""

    def __init__(self, cfg):
        # Training phase config
        self.warmup_epochs = cfg.training.get('warmup_epochs', 5)
        self.trim_epochs = cfg.training.get('trim_epochs', 50)
        self.act_epochs = cfg.training.get('act_epochs', 45)
        self.act_weight = cfg.training.get('act_weight', 0.01)
        self.act_ramp_epochs = cfg.training.get('act_ramp_epochs', 15)
        self.gate_entropy_weight = cfg.training.get('gate_entropy_weight', 0.01)
        self.grad_clip = cfg.training.get('grad_clip', 1.0)
        self.eval_every = cfg.training.get('eval_every', 5)

        # Gumbel temperature annealing schedule
        self.tau_start = cfg.training.get('tau_start', 1.0)
        self.tau_end = cfg.training.get('tau_end', 0.1)
        # Warm restart at Phase 3: steps[num_steps:max_steps] are untrained,
        # need high tau for exploration before annealing down
        self.tau_act_restart = cfg.training.get('tau_act_restart', 0.7)

        super().__init__(cfg)

    def setup_algebra(self):
        """Initialize CPU and Control algebras. Returns CPU algebra for BaseTask."""
        self.algebra_cpu = CliffordAlgebra(3, 0, 1, device=self.device)
        self.algebra_ctrl = CliffordAlgebra(1, 1, 0, device=self.device)
        return self.algebra_cpu

    def setup_model(self):
        mcfg = self.cfg.model
        act_cfg = mcfg.get('act', {})
        color_cfg = mcfg.get('color_unit', {})
        attn_cfg = mcfg.get('attention', {})

        return GTMNet(
            algebra_cpu=self.algebra_cpu,
            algebra_ctrl=self.algebra_ctrl,
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

        return self.model(
            demo_inputs, demo_outputs, demo_masks,
            test_inputs, test_masks, num_demos,
            demo_output_masks=demo_output_masks,
            input_sizes=batch.get('input_sizes'),
            return_trace=return_trace,
        )

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)

        need_trace = (self._phase >= 2 and self.gate_entropy_weight > 0)
        result = self._run_model(batch, return_trace=need_trace)

        logits = result['logits']  # [B, N_grid, 10]

        # Target: test output grid flattened
        test_outputs = batch['test_outputs'].to(self.device)  # [B, H_max, W_max]
        B, H_max, W_max = test_outputs.shape
        targets = test_outputs.reshape(B, H_max * W_max)  # [B, N_grid]

        loss = self.criterion(
            logits.reshape(-1, 10),
            targets.reshape(-1),
        )

        # ACT KL loss (Phase 3 only)
        act_kl = torch.tensor(0.0, device=self.device)
        if 'act_info' in result and result['act_info'] is not None:
            act_kl = result['act_info']['kl_loss']
            loss = loss + self._current_act_weight * act_kl

        # Gate entropy loss (Phases 2-3)
        gate_ent = torch.tensor(0.0, device=self.device)
        if need_trace and 'trace' in result and result['trace'] is not None:
            trace = result['trace']
            if trace['search_scores']:
                ent_sum = sum(_gate_entropy_loss(s) for s in trace['search_scores'])
                gate_ent = ent_sum / len(trace['search_scores'])
                loss = loss + self.gate_entropy_weight * gate_ent

        self._backward(loss)

        if self.grad_clip > 0:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)

        self._optimizer_step()

        logs = {'Loss': loss.item()}
        if act_kl.item() > 0:
            logs['ACT_KL'] = act_kl.item()
        if gate_ent.item() != 0:
            logs['GateEnt'] = gate_ent.item()
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
                logits = result['logits']  # [B, N_grid, 10]
                preds = logits.argmax(dim=-1)  # [B, N_grid]

                test_outputs = batch['test_outputs'].to(self.device)
                test_masks = batch['test_masks'].to(self.device)
                B, H_max, W_max = test_outputs.shape
                targets = test_outputs.reshape(B, H_max * W_max)
                valid = test_masks.reshape(B, H_max * W_max)

                # Cell accuracy (non-padded cells only)
                matches = (preds == targets) & valid
                cell_correct += matches.sum().item()
                cell_total += valid.sum().item()

                # Grid accuracy (entire grid must match)
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
        """Three-phase training loop with ACT ramp-up."""
        logger.info("Starting GTM ARC-AGI v4 Task")
        train_loader, val_loader = self.get_data()

        total_epochs = self.warmup_epochs + self.trim_epochs + self.act_epochs
        self.epochs = total_epochs

        self._phase = 0
        self._current_act_weight = 0.0
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
                self._phase = phase
                if phase == 1:
                    logger.info("Phase 1: Warmup (VM frozen, train head + init_cursor)")
                    self.model.freeze_vm()
                    self.model.disable_act()
                elif phase == 2:
                    logger.info("Phase 2: Circuit Search (fixed steps)")
                    self.model.unfreeze_vm()
                    self.model.disable_act()
                elif phase == 3:
                    act_cfg = self.cfg.model.get('act', {})
                    if act_cfg.get('enabled', True):
                        logger.info("Phase 3: ACT activation (adaptive computation)")
                        self.model.enable_act()
                    else:
                        logger.info("Phase 3: Extended training (ACT disabled)")
                self.optimizer = self._setup_optimizer()
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=10)

            # ACT weight ramp
            if phase == 3:
                act_epoch = epoch - (self.warmup_epochs + self.trim_epochs)
                ramp = min(1.0, act_epoch / self.act_ramp_epochs) if self.act_ramp_epochs > 0 else 1.0
                self._current_act_weight = self.act_weight * ramp
            else:
                self._current_act_weight = 0.0

            if phase == 1:
                tau = self.tau_start
            elif phase == 2:
                progress = min(1.0, (epoch - self.warmup_epochs) / max(self.trim_epochs, 1))
                tau = self.tau_start + (self.tau_act_restart - self.tau_start) * progress
            else:  # phase 3
                act_epoch = epoch - (self.warmup_epochs + self.trim_epochs)
                progress = min(1.0, act_epoch / max(self.act_epochs, 1))
                tau = self.tau_act_restart + (self.tau_end - self.tau_act_restart) * progress
            self.model.set_temperature(tau)

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
            if self._current_act_weight > 0:
                display['ACT_w'] = self._current_act_weight
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
