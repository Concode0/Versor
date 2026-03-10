# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Geometric Latent Reasoning (GLR) Task -- LQA Redesign.

Three probes testing structural blind spots of flat embeddings:
  1. Chain:      Compositional multi-hop reasoning (CLUTRR-style)
  2. Entailment: Asymmetric entailment (HANS-style)
  3. Negation:   Negation sensitivity (BoolQ-Neg-style)

Each probe demonstrates a specific algebraic advantage of Clifford algebra
over flat inner-product spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from core.algebra import CliffordAlgebra
from tasks.base import BaseTask
from datalib.lqa import get_lqa_loaders
from models.lqa.glr_net import GLRNet
from functional.loss import AsymmetryLoss, InvolutionConsistencyLoss
from log import get_logger

logger = get_logger(__name__)


class LQATask(BaseTask):
    """Geometric Latent Reasoning via three algebraic probes.

    Demonstrates that a small geometric post-processor (~300K params) on
    frozen LLM embeddings outperforms equivalently-sized flat models on
    tasks requiring composition, asymmetry, or negation.

    Config keys:
        probe: "chain" | "entailment" | "negation"
        algebra: {p, q, r, device}
        model: {channels, num_layers, num_heads, num_rotors, ...}
        training: {epochs, lr, batch_size, optimizer_type, ...}
        dataset: {data_root, n_train, n_test}
    """

    def __init__(self, cfg: DictConfig):
        self.probe = cfg.get("probe", "chain")
        self._train_loader = None
        self._test_loader = None
        super().__init__(cfg)

    def setup_algebra(self):
        """Cl(4,1) -- conformal GA, dim=32, 6 grades."""
        p = self.cfg.algebra.get("p", 4)
        q = self.cfg.algebra.get("q", 1)
        r = self.cfg.algebra.get("r", 0)
        return CliffordAlgebra(p, q, r, device=self.device)

    def setup_model(self):
        """GLRNet with probe-specific head."""
        mcfg = self.cfg.model
        return GLRNet(
            algebra=self.algebra,
            encoder_dim=mcfg.get("encoder_dim", 384),
            channels=mcfg.get("channels", 16),
            num_layers=mcfg.get("num_layers", 3),
            num_heads=mcfg.get("num_heads", 4),
            num_rotors=mcfg.get("num_rotors", 8),
            dropout=mcfg.get("dropout", 0.1),
            probe=self.probe,
            max_seq_len=mcfg.get("max_seq_len", 64),
            use_entropy_gating=mcfg.get("use_entropy_gating", True),
            num_relations=mcfg.get("num_relations", 10),
        )

    def setup_criterion(self):
        """Multi-loss per probe: primary + auxiliary."""
        if self.probe == "chain":
            return nn.CrossEntropyLoss()
        elif self.probe == "entailment":
            return nn.BCEWithLogitsLoss()
        elif self.probe == "negation":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown probe: {self.probe}")

    def get_data(self):
        """Load probe-specific datasets."""
        dcfg = self.cfg.dataset
        self._train_loader, self._test_loader = get_lqa_loaders(
            data_root=dcfg.get("data_root", "data"),
            probe=self.probe,
            batch_size=self.cfg.training.get("batch_size", 64),
            encoder_name=self.cfg.model.get("encoder", "sentence-transformers/all-MiniLM-L6-v2"),
            n_train=dcfg.get("n_train", None),
            n_test=dcfg.get("n_test", None),
            num_workers=dcfg.get("num_workers", 0),
            pin_memory=self.device_config.pin_memory,
        )
        return self._train_loader

    def _to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def train_step(self, batch):
        """Forward + multi-loss backward."""
        batch = self._to_device(batch)
        self.optimizer.zero_grad()

        output = self.model(batch)
        logits = output["logits"]

        # Primary loss
        if self.probe == "negation":
            labels = batch["answer"].float().unsqueeze(-1)
            loss = self.criterion(logits, labels)
        elif self.probe == "entailment":
            labels = batch["label"].float().unsqueeze(-1)  # [B, 1]
            loss = self.criterion(logits, labels)
        else:
            labels = batch["label"]
            loss = self.criterion(logits, labels)

        # Auxiliary losses
        aux_losses = {}
        tcfg = self.cfg.training

        if self.probe == "chain":
            iso_w = tcfg.get("isometry_weight", 0.01)
            if iso_w > 0:
                iso_loss = self.model.head.isometry_loss()
                loss = loss + iso_w * iso_loss
                aux_losses["Iso"] = iso_loss.item()

        elif self.probe == "entailment":
            asym_w = tcfg.get("asymmetry_weight", 0.1)
            if asym_w > 0:
                # Compute reverse-order logits for asymmetry penalty
                rev_batch = {
                    "premise_emb": batch["hypothesis_emb"],
                    "hypothesis_emb": batch["premise_emb"],
                }
                with torch.no_grad():
                    rev_output = self.model(rev_batch)
                asym_loss_fn = AsymmetryLoss(margin=tcfg.get("asymmetry_margin", 0.1))
                asym_loss = asym_loss_fn(output["logits"], rev_output["logits"])
                loss = loss + asym_w * asym_loss
                aux_losses["Asym"] = asym_loss.item()

        elif self.probe == "negation":
            inv_w = tcfg.get("involution_weight", 0.1)
            if inv_w > 0 and "passage_mv" in output:
                # For pairs of (original, negated) samples in the batch
                is_neg = batch["is_negated"]
                orig_mask = ~is_neg
                neg_mask = is_neg
                n_pairs = min(orig_mask.sum(), neg_mask.sum())
                if n_pairs > 0:
                    orig_features = self.model.head.get_features(
                        output["passage_mv"][orig_mask][:n_pairs],
                        output["question_mv"][orig_mask][:n_pairs] if "question_mv" in output
                        else output["passage_mv"][orig_mask][:n_pairs],
                    )
                    neg_features = self.model.head.get_features(
                        output["passage_mv"][neg_mask][:n_pairs],
                        output["question_mv"][neg_mask][:n_pairs] if "question_mv" in output
                        else output["passage_mv"][neg_mask][:n_pairs],
                    )
                    inv_loss_fn = InvolutionConsistencyLoss()
                    inv_loss = inv_loss_fn(orig_features, neg_features, self.algebra)
                    loss = loss + inv_w * inv_loss
                    aux_losses["Inv"] = inv_loss.item()

        self._backward(loss)
        self._optimizer_step()

        logs = {"Loss": loss.item()}
        logs.update(aux_losses)
        return loss.item(), logs

    def evaluate(self, data=None):
        """Per-probe evaluation metrics."""
        loader = self._test_loader if self._test_loader is not None else data
        if loader is None:
            logger.warning("No test loader available for evaluation.")
            return {"Accuracy": 0.0}

        self.model.eval()
        all_preds = []
        all_labels = []
        all_chain_lengths = []
        all_confidences = []
        all_g2_norms = []

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                output = self.model(batch)
                logits = output["logits"]

                if self.probe in ("negation", "entailment"):
                    preds = (logits.squeeze(-1) > 0).long()
                    if self.probe == "negation":
                        labels = batch["answer"].long()
                    else:
                        labels = batch["label"].long()
                    # Confidence via sigmoid
                    conf = torch.sigmoid(logits.squeeze(-1))
                    all_confidences.append(conf.cpu())
                    # Grade-2 diagnostics for entailment
                    if self.probe == "entailment" and "premise_mv" in output:
                        _, _, g2_norm, _, _ = self.model.head._compute_product_features(
                            output["premise_mv"], output["hypothesis_mv"]
                        )
                        all_g2_norms.append(g2_norm.cpu())
                else:
                    preds = logits.argmax(dim=-1)
                    labels = batch["label"]

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                if self.probe == "chain" and "chain_length" in batch:
                    all_chain_lengths.append(batch["chain_length"].cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        correct = (all_preds == all_labels).float()
        accuracy = correct.mean().item()

        metrics = {"Accuracy": accuracy}

        # Probe-specific metrics
        if self.probe == "chain" and all_chain_lengths:
            all_chain_lengths = torch.cat(all_chain_lengths)
            # Accuracy by chain length
            for length in sorted(all_chain_lengths.unique().tolist()):
                mask = all_chain_lengths == length
                if mask.sum() > 0:
                    acc = correct[mask].mean().item()
                    metrics[f"Acc@len{int(length)}"] = acc
            logger.info("Chain accuracy by length:")
            for k, v in metrics.items():
                if k.startswith("Acc@"):
                    logger.info("  %s: %.4f", k, v)

        elif self.probe == "entailment":
            # Per-class accuracy (binary: 1=entailment, 0=non-entailment)
            ent_mask = all_labels == 1
            nonent_mask = all_labels == 0
            if ent_mask.sum() > 0:
                metrics["Acc_Entailment"] = correct[ent_mask].mean().item()
            if nonent_mask.sum() > 0:
                metrics["Acc_NonEntailment"] = correct[nonent_mask].mean().item()

            # Prediction distribution
            n_total = len(all_preds)
            metrics["Pred_Entailment_Frac"] = (all_preds == 1).float().mean().item()
            metrics["Pred_NonEntailment_Frac"] = (all_preds == 0).float().mean().item()

            # Confidence statistics
            if all_confidences:
                all_conf = torch.cat(all_confidences)
                metrics["Confidence_Mean"] = all_conf.mean().item()
                metrics["Confidence_Std"] = all_conf.std().item()
                # Confidence for correct vs incorrect predictions
                if correct.sum() > 0:
                    metrics["Confidence_Correct"] = all_conf[correct.bool()].mean().item()
                if (1 - correct).sum() > 0:
                    metrics["Confidence_Incorrect"] = all_conf[~correct.bool()].mean().item()

            # Grade-2 signal diagnostics
            if all_g2_norms:
                g2_all = torch.cat([g.flatten() for g in all_g2_norms])
                metrics["Grade2_Norm_Mean"] = g2_all.mean().item()
                metrics["Grade2_Norm_Std"] = g2_all.std().item()
                metrics["Grade2_Norm_Max"] = g2_all.max().item()

        elif self.probe == "negation":
            # Negation robustness
            all_chain_lengths_or_neg = []
            for batch in loader:
                if "is_negated" in batch:
                    all_chain_lengths_or_neg.append(batch["is_negated"].cpu())
            if all_chain_lengths_or_neg:
                is_neg = torch.cat(all_chain_lengths_or_neg)
                acc_orig = correct[~is_neg].mean().item() if (~is_neg).sum() > 0 else 0.0
                acc_neg = correct[is_neg].mean().item() if is_neg.sum() > 0 else 0.0
                metrics["Acc_Original"] = acc_orig
                metrics["Acc_Negated"] = acc_neg
                metrics["Negation_Robustness"] = min(acc_orig, acc_neg)

        logger.info("Evaluation metrics:")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)

        self.model.train()
        return metrics

    def visualize(self, data=None):
        """Visualization placeholder for GLR probes."""
        logger.info("GLR probe=%s -- visualization not implemented (use evaluate() for metrics)", self.probe)

    def run(self):
        """Full training loop with validation."""
        logger.info("Starting GLR Task: probe=%s, algebra=Cl(%d,%d,%d)",
                     self.probe, self.algebra.p, self.algebra.q, self.algebra.r)

        train_loader = self.get_data()

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Model parameters: %d (%.1fK)", n_params, n_params / 1000)

        # Training loop
        self.model.train()
        best_acc = 0.0

        from tqdm import tqdm
        pbar = tqdm(range(self.epochs))

        for epoch in pbar:
            total_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.scheduler.step(avg_loss)

            # Periodic evaluation
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                metrics = self.evaluate()
                acc = metrics.get("Accuracy", 0.0)
                if acc > best_acc:
                    best_acc = acc
                logs["ValAcc"] = acc
                logs["BestAcc"] = best_acc

            current_lr = self.optimizer.param_groups[0]['lr']
            logs["Loss"] = avg_loss
            logs["LR"] = current_lr
            desc = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            pbar.set_description(desc)

        logger.info("Training complete. Best accuracy: %.4f", best_acc)

        # Final evaluation
        final_metrics = self.evaluate()
        return final_metrics
