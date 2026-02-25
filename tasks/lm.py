# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

import math
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig

from core.algebra import CliffordAlgebra
from models.ga_transformer import GALanguageModel
from datasets.text import get_text_loaders
from tasks.base import BaseTask
from log import get_logger

logger = get_logger(__name__)


class LanguageModelingTask(BaseTask):
    """Character-level language modeling with a GA-native Transformer.

    Uses Cl(3,1) - Minkowski spacetime algebra - so token embeddings
    are Lorentz-covariant multivectors. The FFN is the Embedded Geometric
    Toolbox (MultiRotorFFN), replacing the standard MLP nonlinearity with
    K parallel rotor superpositions.

    Metrics: cross-entropy loss, perplexity.
    Visualization: greedy-decoded text from seed "The ".
    """

    def __init__(self, cfg: DictConfig):
        # Pre-scan vocab before super().__init__() which calls setup_model()
        _, _, vocab_size = get_text_loaders(
            data_path=cfg.dataset.data_path,
            seq_len=cfg.dataset.seq_len,
            batch_size=cfg.training.batch_size,
            tokenizer=cfg.dataset.get('tokenizer', 'char'),
        )
        self.vocab_size = vocab_size
        self._train_dataset = None
        self._val_dataset = None
        super().__init__(cfg)

    def setup_algebra(self) -> CliffordAlgebra:
        return CliffordAlgebra(
            p=self.cfg.algebra.p,
            q=self.cfg.algebra.q,
            device=self.device,
        )

    def setup_model(self) -> GALanguageModel:
        m = self.cfg.model
        return GALanguageModel(
            algebra=self.algebra,
            vocab_size=self.vocab_size,
            channels=m.channels,
            num_layers=m.num_layers,
            num_heads=m.num_heads,
            max_seq_len=m.max_seq_len,
            ffn_mult=m.get('ffn_mult', 4),
            num_rotors=m.get('num_rotors', 8),
            causal=True,
            dropout=self.cfg.training.get('dropout', 0.0),
            use_rotor_backend=m.get('use_rotor_backend', False),
            use_decomposition=m.get('use_decomposition', False),
        )

    def setup_criterion(self) -> nn.CrossEntropyLoss:
        return nn.CrossEntropyLoss()

    def get_data(self):
        """Returns (train_loader, val_loader)."""
        train_loader, val_loader, _ = get_text_loaders(
            data_path=self.cfg.dataset.data_path,
            seq_len=self.cfg.dataset.seq_len,
            batch_size=self.cfg.training.batch_size,
            tokenizer=self.cfg.dataset.get('tokenizer', 'char'),
        )
        self._train_dataset = train_loader.dataset
        self._val_dataset = val_loader.dataset
        return train_loader, val_loader

    def train_step(self, batch):
        """One optimisation step.

        Args:
            batch: (x, y) tensors [B, L].

        Returns:
            (loss_scalar, logs_dict)
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        # Forward with hidden for optional grade regularisation
        grade_reg_weight = self.cfg.training.loss_weights.get('grade_reg', 0.0)
        if grade_reg_weight > 0.0:
            logits, hidden = self.model(x, return_hidden=True)
        else:
            logits = self.model(x)
            hidden = None

        # Primary language-modelling loss
        loss = self.criterion(logits, y.flatten())

        # Optional grade regularisation: penalise non-scalar energy in hidden
        if grade_reg_weight > 0.0 and hidden is not None:
            non_scalar_energy = (hidden[..., 1:] ** 2).mean()
            loss = loss + grade_reg_weight * non_scalar_energy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), {'LM': loss.item()}

    def evaluate(self, val_loader) -> float:
        """Evaluates perplexity on the validation set.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Perplexity (float).
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)           # [B*L, vocab]
                loss = self.criterion(logits, y.flatten())
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20.0))  # cap to avoid overflow
        logger.info(f"Val Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return perplexity

    def generate(self, seed: str = "The ", length: int = 200,
                 temperature: float = 1.0, top_k: int = 0) -> str:
        """Generate text from a seed string with temperature sampling.

        Args:
            seed: Starting text.
            length: Number of characters to generate.
            temperature: Sampling temperature (0 = greedy, <1 = sharper, >1 = more random).
            top_k: If > 0, restrict sampling to top-k tokens.

        Returns:
            Generated text string.
        """
        dataset = self._train_dataset
        if dataset is None:
            return ""

        ids = [dataset.char_to_idx.get(c, 0) for c in seed]
        generated = list(ids)
        seq_len = self.cfg.dataset.seq_len

        self.model.eval()
        with torch.no_grad():
            for _ in range(length):
                context = generated[-seq_len:]
                x = torch.tensor(
                    [context], dtype=torch.long, device=self.device
                )
                logits = self.model(x)  # [T, vocab]
                next_logits = logits[-1]  # [vocab]

                if temperature <= 0:
                    # Greedy
                    next_id = next_logits.argmax().item()
                else:
                    next_logits = next_logits / temperature

                    if top_k > 0:
                        # Zero out tokens outside top-k
                        values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        next_logits[next_logits < values[-1]] = float('-inf')

                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_id)

        return dataset.decode(generated)

    def visualize(self, val_loader=None):
        """Generate text samples at multiple temperatures.

        Args:
            val_loader: Unused; kept for API compatibility with BaseTask.
        """
        if self._train_dataset is None:
            logger.warning("Dataset not available for text generation.")
            return

        for temp in [0.0, 0.5, 0.8, 1.0]:
            label = "greedy" if temp == 0 else f"T={temp}"
            text = self.generate(temperature=temp, top_k=50)
            logger.info(f"[{label}]: {text[:200]}")

    def run(self):
        """Train/val loop with checkpoint saving and text generation."""
        logger.info(f"Starting Task: {self.cfg.name}")
        train_loader, val_loader = self.get_data()

        pbar = tqdm(range(self.epochs))
        best_ppl = float('inf')

        for epoch in pbar:
            self.model.train()
            total_loss = 0.0

            inner_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in inner_pbar:
                loss_val, _ = self.train_step(batch)
                total_loss += loss_val
                inner_pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg_loss = total_loss / max(len(train_loader), 1)
            ppl = self.evaluate(val_loader)
            self.scheduler.step(ppl)

            if ppl < best_ppl:
                best_ppl = ppl
                self.save_checkpoint(f"{self.cfg.name}_best.pt")

            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.6f}"
            )

        logger.info(f"Training Complete. Best Val Perplexity: {best_ppl:.2f}")

        self.model.eval()
        with torch.no_grad():
            gen_cfg = self.cfg.get('generation', {})
            temp = gen_cfg.get('temperature', 0.8)
            top_k = gen_cfg.get('top_k', 50)
            length = gen_cfg.get('length', 200)
            text = self.generate(temperature=temp, top_k=top_k, length=length)
            logger.info(f"Generated (T={temp}, top_k={top_k}):\n{text}")
