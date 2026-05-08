# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Raw-text geometric SLM."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from core.module import CliffordModule
from layers.adapters.embedding import MultivectorEmbedding, RotaryBivectorPE
from layers.primitives.normalization import CliffordLayerNorm
from layers.primitives.projection import GeometricNeutralizer

from .reasoning import GeometricSLMBlock


class GeometricSLM(CliffordModule):
    """Small causal language model with multivector token states."""

    def __init__(
        self,
        algebra: CliffordAlgebra,
        vocab_size: int,
        channels: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        num_rotors: int = 8,
        ffn_mult: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        bivector_weight: float = 0.5,
        attn_block_size: int = 128,
        tie_embeddings: bool = True,
        use_neutralizer: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__(algebra)
        self.vocab_size = vocab_size
        self.channels = channels
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.use_neutralizer = use_neutralizer
        self.pad_token_id = pad_token_id

        self.token_embedding = MultivectorEmbedding(algebra, vocab_size, channels)
        with torch.no_grad():
            self.token_embedding.embedding.weight[pad_token_id].zero_()

        self.position_embedding = RotaryBivectorPE(algebra, channels, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                GeometricSLMBlock(
                    algebra=algebra,
                    channels=channels,
                    num_heads=num_heads,
                    num_rotors=num_rotors,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                    bivector_weight=bivector_weight,
                    attn_block_size=attn_block_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = CliffordLayerNorm(algebra, channels)
        self.neutralizer = GeometricNeutralizer(algebra, channels) if use_neutralizer else None

        if tie_embeddings:
            self.lm_head = None
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.lm_head = nn.Linear(channels * algebra.dim, vocab_size)
            self.output_bias = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> dict:
        """Return next-token logits for ``input_ids``.

        Args:
            input_ids: Token ids ``[B, L]``.
            attention_mask: Optional mask ``[B, L]`` where 1/True means valid.
        """
        B, L = input_ids.shape
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len={self.max_seq_len}")

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        h = self.token_embedding(input_ids)
        h = self.position_embedding(h)

        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)

        h = self.final_norm(h.reshape(B * L, self.channels, self.algebra.dim))
        if self.neutralizer is not None:
            h = self.neutralizer(h)

        flat = h.reshape(B * L, self.channels * self.algebra.dim)
        if self.tie_embeddings:
            logits = flat @ self.token_embedding.embedding.weight.t()
            logits = logits / math.sqrt(self.channels * self.algebra.dim)
            logits = logits + self.output_bias
        else:
            logits = self.lm_head(flat)

        return {
            "logits": logits.reshape(B, L, self.vocab_size),
            "hidden_states": h.reshape(B, L, self.channels, self.algebra.dim),
        }

    def reasoner_parameter_count(self) -> int:
        """Count trainable parameters outside token embedding and decoder head."""
        total = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("token_embedding."):
                continue
            if name.startswith("lm_head.") or name == "output_bias":
                continue
            total += param.numel()
        return total

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Naive autoregressive generation for small local inference checks."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        was_training = self.training
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            context = generated[:, -self.max_seq_len :]
            attention_mask = torch.ones_like(context)
            logits = self(context, attention_mask=attention_mask)["logits"][:, -1, :]
            logits = logits / temperature

            if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                values, indices = logits.topk(top_k, dim=-1)
                filtered = torch.full_like(logits, float("-inf"))
                logits = filtered.scatter(-1, indices, values)

            if sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        if was_training:
            self.train()
        return generated
