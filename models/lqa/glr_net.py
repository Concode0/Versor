# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Latent Reasoning Network (GLRNet).

A small geometric post-processor (~300K params) on frozen LLM embeddings.
Uses Cl(4,1) conformal GA to provide algebraic structure for:
- Non-commutative products (asymmetry)
- Rotor composition (exact multi-hop)
- Grade involution (negation as automorphism)
- Grade-0 invariance (truth preserved under transformations)

Data flow ensures L >= 2 for every probe so the transformer has
real sequence structure to attend over:
  Chain:      L = chain_length  (each sentence is a token)
  Entailment: L = 2             (premise, hypothesis stacked)
  Negation:   L = 2             (passage, question stacked)
"""

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from core.module import CliffordModule
from layers.adapters.embedding import RotaryBivectorPE
from layers.adapters.mother import MotherEmbedding
from layers.blocks.transformer import GeometricTransformerBlock
from layers.primitives.normalization import CliffordLayerNorm
from layers.primitives.projection import GeometricNeutralizer

from .heads import ChainReasoningHead, EntailmentHead, NegationHead


class GLRNet(CliffordModule):
    """Geometric Latent Reasoning Network.

    Architecture:
        frozen_embeddings [B, L, encoder_dim]
            -> MotherEmbedding(encoder_dim -> C x 2^n)  per token
            -> RotaryBivectorPE                          position-dependent rotors
            -> GeometricTransformerBlock x N              cross-token attention
            -> GeometricNeutralizer                      grade-0/grade-2 separation
            -> TaskHead (probe-specific)
            -> prediction

    Every probe constructs a multi-token sequence (L >= 2) so the
    transformer always has cross-token attention to work with.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        encoder_dim: int = 384,
        channels: int = 16,
        num_layers: int = 3,
        num_heads: int = 4,
        num_rotors: int = 8,
        dropout: float = 0.1,
        probe: str = "chain",
        max_seq_len: int = 64,
        use_entropy_gating: bool = True,
        num_relations: int = 10,
    ):
        super().__init__(algebra)
        self.channels = channels
        self.probe = probe
        self.encoder_dim = encoder_dim

        # 1. Lift frozen embeddings to multivector space
        self.mother = MotherEmbedding(algebra, encoder_dim, channels)

        # 2. Positional encoding via bivector rotors
        self.pe = RotaryBivectorPE(algebra, channels, max_seq_len)

        # 3. Geometric transformer backbone
        self.blocks = nn.ModuleList(
            [
                GeometricTransformerBlock(
                    algebra,
                    channels,
                    num_heads,
                    num_rotors,
                    dropout=dropout,
                    use_entropy_gating=use_entropy_gating,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Final normalization + neutralization
        self.final_norm = CliffordLayerNorm(algebra, channels)
        self.neutralizer = GeometricNeutralizer(algebra, channels)

        # 5. Probe-specific head
        if probe == "chain":
            self.head = ChainReasoningHead(algebra, channels, num_relations=num_relations)
        elif probe == "entailment":
            self.head = EntailmentHead(algebra, channels)
        elif probe == "negation":
            self.head = NegationHead(algebra, channels)
        else:
            raise ValueError(f"Unknown probe: {probe}")

    def _lift_and_attend(self, embeddings: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Lift a sequence of embeddings through the full backbone.

        Args:
            embeddings: [B, L, encoder_dim] with L >= 2.
            key_padding_mask: Optional [B, L] bool mask where True = padded.

        Returns:
            Attended multivectors [B, L, C, D] (sequence preserved).
        """
        B, L, _ = embeddings.shape
        D = self.algebra.dim

        # Lift per-token: [B*L, encoder_dim] -> [B*L, C, D]
        mv = self.mother(embeddings.reshape(B * L, -1))
        mv = mv.reshape(B, L, self.channels, D)

        # Positional encoding (rotor per position)
        mv = self.pe(mv)

        # Transformer blocks -- cross-token attention happens here
        for block in self.blocks:
            mv = block(mv, key_padding_mask=key_padding_mask)

        return mv  # [B, L, C, D]

    def forward(self, batch: dict) -> dict:
        """Forward pass dispatching to probe-specific logic."""
        if self.probe == "chain":
            return self._forward_chain(batch)
        elif self.probe == "entailment":
            return self._forward_entailment(batch)
        elif self.probe == "negation":
            return self._forward_negation(batch)

    def _forward_chain(self, batch: dict) -> dict:
        """Chain reasoning: per-sentence embeddings -> rotor composition -> classify.

        Each sentence in the chain is a separate token. The transformer
        attends across all chain steps, then the head applies soft-gated
        rotor bank and classifies from the composed result.

        Args:
            batch: {
                "sentence_embeddings": [B, L, encoder_dim],  padded
                "chain_length":        [B],
                "label":               [B],
            }
        """
        embs = batch["sentence_embeddings"]  # [B, L_max, encoder_dim]
        chain_lengths = batch["chain_length"]  # [B]

        B, L_max, _ = embs.shape

        # Build padding mask: True = padded position
        positions = torch.arange(L_max, device=embs.device).unsqueeze(0)  # [1, L_max]
        key_padding_mask = positions >= chain_lengths.unsqueeze(1)  # [B, L_max]

        # Full backbone with padding mask
        mv = self._lift_and_attend(embs, key_padding_mask=key_padding_mask)  # [B, L, C, D]

        B, L, C, D = mv.shape

        # Masked mean pooling: exclude padded positions
        valid_mask = ~key_padding_mask  # [B, L], True = valid
        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        mv_pooled = (mv * valid_mask_expanded).sum(dim=1) / valid_count.unsqueeze(-1)  # [B, C, D]

        mv_pooled = self.final_norm(mv_pooled)
        mv_pooled = self.neutralizer(mv_pooled)

        logits = self.head(mv_pooled)
        return {"logits": logits}

    def _forward_entailment(self, batch: dict) -> dict:
        """Entailment: stack premise + hypothesis as 2-token sequence.

        The transformer sees both as a length-2 sequence, enabling
        cross-attention between premise and hypothesis. After attention,
        we split back and feed into the asymmetric head.

        Args:
            batch: {
                "premise_emb":    [B, encoder_dim],
                "hypothesis_emb": [B, encoder_dim],
            }
        """
        # Stack as 2-token sequence: [B, 2, encoder_dim]
        seq = torch.stack([batch["premise_emb"], batch["hypothesis_emb"]], dim=1)

        # Full backbone
        mv = self._lift_and_attend(seq)  # [B, 2, C, D]

        # Split back into premise/hypothesis multivectors
        premise_mv = mv[:, 0, :, :]  # [B, C, D]
        hypothesis_mv = mv[:, 1, :, :]  # [B, C, D]

        # Normalize + neutralize each
        premise_mv = self.neutralizer(self.final_norm(premise_mv))
        hypothesis_mv = self.neutralizer(self.final_norm(hypothesis_mv))

        logits = self.head(premise_mv, hypothesis_mv)
        return {
            "logits": logits,
            "premise_mv": premise_mv,
            "hypothesis_mv": hypothesis_mv,
        }

    def _forward_negation(self, batch: dict) -> dict:
        """Negation: stack passage + question as 2-token sequence.

        Same principle as entailment -- the transformer cross-attends
        between passage and question, then the head uses grade involution.

        Args:
            batch: {
                "passage_emb":  [B, encoder_dim],
                "question_emb": [B, encoder_dim],
            }
        """
        seq = torch.stack([batch["passage_emb"], batch["question_emb"]], dim=1)

        mv = self._lift_and_attend(seq)  # [B, 2, C, D]

        passage_mv = mv[:, 0, :, :]
        question_mv = mv[:, 1, :, :]

        passage_mv = self.neutralizer(self.final_norm(passage_mv))
        question_mv = self.neutralizer(self.final_norm(question_mv))

        logits = self.head(passage_mv, question_mv)
        return {
            "logits": logits,
            "passage_mv": passage_mv,
            "question_mv": question_mv,
        }
