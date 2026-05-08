# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Raw-text Small Language Model task.

The SLM task intentionally owns dataset sampling, tokenizer training, and
experiment choices. The model package stays focused on the geometric forward
path and can be reused by later logical reasoning evaluations.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from core.algebra import CliffordAlgebra
from core.analysis import AnalysisConfig, GeometricAnalyzer, SamplingConfig
from datalib.slm import (
    IGNORE_INDEX,
    CausalLoaderConfig,
    TextCorpusConfig,
    build_causal_lm_loaders,
    load_text_samples,
    split_train_eval,
)
from log import get_logger
from models.slm import GeometricSLM, SubwordTokenizer
from tasks.base import BaseTask

logger = get_logger(__name__)


def _section(cfg: DictConfig, name: str) -> dict:
    value = cfg.get(name, {})
    return value if value is not None else {}


class SLMTask(BaseTask):
    """Train a raw-text geometric SLM on a sampled Cosmopedia split."""

    def __init__(self, cfg: DictConfig):
        self.tokenizer = None
        self._train_loader = None
        self._eval_loader = None
        self._last_analysis_summary = None
        super().__init__(cfg)

    def setup_algebra(self):
        algebra_cfg = _section(self.cfg, "algebra")
        p = algebra_cfg.get("p", 4)
        q = algebra_cfg.get("q", 1)
        r = algebra_cfg.get("r", 1)
        return CliffordAlgebra(p, q, r, device=self.device)

    def setup_model(self):
        model_cfg = _section(self.cfg, "model")
        tokenizer_cfg = _section(self.cfg, "tokenizer")
        return GeometricSLM(
            algebra=self.algebra,
            vocab_size=tokenizer_cfg.get("vocab_size", 8192),
            channels=model_cfg.get("channels", 16),
            num_layers=model_cfg.get("num_layers", 4),
            num_heads=model_cfg.get("num_heads", 4),
            num_rotors=model_cfg.get("num_rotors", 8),
            ffn_mult=model_cfg.get("ffn_mult", 4),
            max_seq_len=model_cfg.get("max_seq_len", 128),
            dropout=model_cfg.get("dropout", 0.1),
            bivector_weight=model_cfg.get("bivector_weight", 0.5),
            attn_block_size=model_cfg.get("attn_block_size", 128),
            tie_embeddings=model_cfg.get("tie_embeddings", True),
            use_neutralizer=model_cfg.get("use_neutralizer", True),
            pad_token_id=SubwordTokenizer.pad_id,
        )

    def setup_criterion(self):
        return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def get_data(self):
        dataset_cfg = _section(self.cfg, "dataset")
        tokenizer_cfg = _section(self.cfg, "tokenizer")
        model_cfg = _section(self.cfg, "model")
        training_cfg = _section(self.cfg, "training")

        corpus_cfg = TextCorpusConfig.from_mapping(dataset_cfg)
        texts = load_text_samples(corpus_cfg)
        train_texts, eval_texts = split_train_eval(texts, corpus_cfg.eval_fraction)

        self.tokenizer = SubwordTokenizer(
            vocab_size=tokenizer_cfg.get("vocab_size", 8192),
            min_frequency=tokenizer_cfg.get("min_frequency", 2),
            lowercase=tokenizer_cfg.get("lowercase", True),
            mode=tokenizer_cfg.get("mode", "subword"),
        ).train(train_texts)
        logger.info("Tokenizer mode=%s vocab=%d", tokenizer_cfg.get("mode", "subword"), len(self.tokenizer))

        num_workers = dataset_cfg.get("num_workers", self.device_config.num_workers)
        if num_workers is None:
            num_workers = self.device_config.num_workers
        pin_memory = dataset_cfg.get("pin_memory", self.device_config.pin_memory)
        if pin_memory is None:
            pin_memory = self.device_config.pin_memory

        loader_cfg = CausalLoaderConfig(
            batch_size=training_cfg.get("batch_size", 8),
            max_seq_len=model_cfg.get("max_seq_len", 128),
            chunk_long_texts=dataset_cfg.get("chunk_long_texts", False),
            stride=dataset_cfg.get("stride", None),
            shuffle_train=dataset_cfg.get("shuffle_train", True),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self._train_loader, self._eval_loader = build_causal_lm_loaders(
            train_texts,
            eval_texts,
            self.tokenizer,
            loader_cfg,
        )

        return self._train_loader

    def _to_device(self, batch: dict) -> dict:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train_step(self, batch):
        batch = self._to_device(batch)
        self.optimizer.zero_grad()

        with self.device_config.autocast_context():
            output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = output["logits"]
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))

        self._backward(loss)
        self._optimizer_step()

        with torch.no_grad():
            metrics = self._batch_metrics(logits, batch["labels"], loss)
        return loss.item(), metrics

    def _batch_metrics(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> dict:
        valid = labels != IGNORE_INDEX
        if valid.any():
            preds = logits.argmax(dim=-1)
            acc = (preds[valid] == labels[valid]).float().mean().item()
        else:
            acc = 0.0
        ppl = math.exp(min(loss.item(), 20.0))
        return {"Loss": loss.item(), "PPL": ppl, "TokenAcc": acc}

    def evaluate(self, data=None):
        loader = data if data is not None else self._eval_loader
        if loader is None:
            logger.info("No evaluation split available.")
            return {"Loss": 0.0, "PPL": 1.0, "TokenAcc": 0.0}

        batches = [loader] if isinstance(loader, dict) else loader
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        with torch.no_grad():
            for batch in batches:
                batch = self._to_device(batch)
                with self.device_config.autocast_context():
                    output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                    logits = output["logits"]
                    labels = batch["labels"]
                    loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

                valid = labels != IGNORE_INDEX
                tokens = int(valid.sum().item())
                total_loss += loss.item() * max(tokens, 1)
                total_tokens += tokens
                if tokens > 0:
                    total_correct += int((logits.argmax(dim=-1)[valid] == labels[valid]).sum().item())

        denom = max(total_tokens, 1)
        avg_loss = total_loss / denom
        metrics = {
            "Loss": avg_loss,
            "PPL": math.exp(min(avg_loss, 20.0)),
            "TokenAcc": total_correct / denom,
        }
        logger.info(
            "Evaluation: Loss=%.4f PPL=%.2f TokenAcc=%.4f",
            metrics["Loss"],
            metrics["PPL"],
            metrics["TokenAcc"],
        )
        self.model.train()
        return metrics

    def visualize(self, data=None):
        logger.info("SLM visualization is not implemented; use evaluate() for perplexity and token accuracy.")

    def _model_core(self):
        """Return the original module when torch.compile wraps the model."""
        return getattr(self.model, "_orig_mod", self.model)

    def _checkpoint_dir(self) -> Path:
        checkpoint_cfg = _section(self.cfg, "checkpointing")
        return Path(checkpoint_cfg.get("dir", "checkpoints/slm"))

    def save_checkpoint(self, path: str):
        """Save SLM model state together with tokenizer state."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.cfg,
            "tokenizer_state": self.tokenizer.to_state() if self.tokenizer is not None else None,
            "analysis_summary": self._last_analysis_summary,
        }
        torch.save(checkpoint, path_obj)
        logger.info("Checkpoint saved to %s", path_obj)

    def load_checkpoint(self, path: str):
        """Restore model, optimizer, scheduler, and tokenizer state when available."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        tokenizer_state = checkpoint.get("tokenizer_state")
        if tokenizer_state is not None:
            self.tokenizer = SubwordTokenizer.from_state(tokenizer_state)
        self._last_analysis_summary = checkpoint.get("analysis_summary")
        logger.info("Checkpoint loaded from %s", path)

    def _metric_is_better(self, value: float, best: float, mode: str) -> bool:
        return value < best if mode == "min" else value > best

    def _maybe_save_best(self, logs: dict, best_value):
        checkpoint_cfg = _section(self.cfg, "checkpointing")
        if not checkpoint_cfg.get("enabled", True) or not checkpoint_cfg.get("save_best", True):
            return best_value

        monitor = checkpoint_cfg.get("monitor", "EvalPPL")
        metric = logs.get(monitor)
        if metric is None:
            return best_value

        mode = checkpoint_cfg.get("mode", "min")
        if best_value is None or self._metric_is_better(metric, best_value, mode):
            best_value = metric
            self.save_checkpoint(self._checkpoint_dir() / checkpoint_cfg.get("best_filename", "slm_best.pt"))
        return best_value

    def _maybe_save_final(self):
        checkpoint_cfg = _section(self.cfg, "checkpointing")
        if checkpoint_cfg.get("enabled", True) and checkpoint_cfg.get("save_final", True):
            self.save_checkpoint(self._checkpoint_dir() / checkpoint_cfg.get("filename", "slm_final.pt"))

    def _analysis_should_run(self, stage: str) -> bool:
        analysis_cfg = _section(self.cfg, "analysis")
        if not analysis_cfg.get("enabled", False):
            return False
        run_on = analysis_cfg.get("run_on", "final")
        return run_on == stage or run_on == "both"

    def _collect_analysis_states(self, loader, max_batches: int, max_samples: int):
        if loader is None:
            return None

        states = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                batch = self._to_device(batch)
                output = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                hidden = output["hidden_states"]  # [B, L, C, D]
                valid = batch["attention_mask"].bool()
                states.append(hidden[valid].detach())

                total = sum(part.shape[0] for part in states)
                if total >= max_samples:
                    break

        if not states:
            return None
        return torch.cat(states, dim=0)[:max_samples]

    def _run_model_analysis(self, stage: str):
        analysis_cfg = _section(self.cfg, "analysis")
        if not self._analysis_should_run(stage):
            return None

        loader = self._eval_loader if self._eval_loader is not None else self._train_loader
        max_batches = analysis_cfg.get("max_batches", 1)
        max_samples = analysis_cfg.get("max_samples", 256)
        mv_states = self._collect_analysis_states(loader, max_batches=max_batches, max_samples=max_samples)
        if mv_states is None:
            logger.info("Skipping SLM analysis: no hidden states collected.")
            return None

        config = AnalysisConfig(
            device=self.device,
            sampling=SamplingConfig(
                strategy=analysis_cfg.get("sampling_strategy", "passthrough"),
                max_samples=max_samples,
                seed=_section(self.cfg, "dataset").get("seed", 0),
            ),
            run_dimension=analysis_cfg.get("run_dimension", False),
            run_signature=analysis_cfg.get("run_signature", False),
            run_spectral=analysis_cfg.get("run_spectral", True),
            run_symmetry=analysis_cfg.get("run_symmetry", True),
            run_commutator=analysis_cfg.get("run_commutator", True),
            energy_threshold=analysis_cfg.get("energy_threshold", 0.05),
            k_neighbors=analysis_cfg.get("k_neighbors", 8),
        )
        report = GeometricAnalyzer(config).analyze(mv_states, algebra=self.algebra)
        summary = report.summary()
        self._last_analysis_summary = summary
        logger.info("SLM %s analysis:\n%s", stage, summary)

        checkpoint_cfg = _section(self.cfg, "checkpointing")
        if analysis_cfg.get("save_summary", True) and checkpoint_cfg.get("enabled", True):
            out = self._checkpoint_dir() / f"analysis_{stage}_summary.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(summary + "\n", encoding="utf-8")
            logger.info("Analysis summary saved to %s", out)

        self.model.train()
        return report

    def _run_inference_preview(self):
        inference_cfg = _section(self.cfg, "inference")
        if not inference_cfg.get("enabled", False):
            return None
        if self.tokenizer is None:
            logger.info("Skipping SLM inference preview: tokenizer is not available.")
            return None

        prompt = inference_cfg.get("prompt", "")
        prompt_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated = self._model_core().generate(
            input_ids=input_ids,
            max_new_tokens=inference_cfg.get("max_new_tokens", 32),
            temperature=inference_cfg.get("temperature", 1.0),
            top_k=inference_cfg.get("top_k", 50),
            sample=inference_cfg.get("sample", True),
            eos_token_id=self.tokenizer.eos_id,
        )
        text = self.tokenizer.decode(generated[0].tolist())
        logger.info("SLM inference preview: %s", text)
        return text

    def run(self):
        logger.info("Starting SLM task.")
        train_loader = self.get_data()

        total_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        reasoner_params = self._model_core().reasoner_parameter_count()
        logger.info("Model parameters: total=%d reasoner=%d", total_params, reasoner_params)

        eval_interval = _section(self.cfg, "training").get("eval_interval", 1)
        best_value = None
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            total_loss = 0.0
            logs = {"Loss": 0.0, "PPL": 1.0, "TokenAcc": 0.0}
            for batch in train_loader:
                loss, logs = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / max(len(train_loader), 1)
            self.scheduler.step(avg_loss)

            if self._eval_loader is not None and (epoch + 1) % eval_interval == 0:
                eval_metrics = self.evaluate()
                logs["EvalPPL"] = eval_metrics["PPL"]
                logs["EvalAcc"] = eval_metrics["TokenAcc"]
                if self._analysis_should_run("eval"):
                    self._run_model_analysis("eval")

            logs["Loss"] = avg_loss
            logs["LR"] = self.optimizer.param_groups[0]["lr"]
            best_value = self._maybe_save_best(logs, best_value)
            pbar.set_description(" | ".join(f"{key}: {value:.4f}" for key, value in logs.items()))

        logger.info("SLM training complete.")
        final_metrics = self.evaluate() if self._eval_loader is not None else {}
        self._run_model_analysis("final")
        self._run_inference_preview()
        self._maybe_save_final()
        return final_metrics
