# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Probe-specific task heads for Geometric Latent Reasoning.

Three CliffordModule heads:
- ChainReasoningHead: soft-gated rotor bank, classify from weighted rotor composition
- EntailmentHead: geometric product P*rev(H) -> asymmetric features -> 3-way
- NegationHead: grade involution + GeometricNeutralizer -> binary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import CliffordAlgebra
from layers.primitives.base import CliffordModule
from layers.primitives.rotor import RotorLayer
from layers.primitives.projection import GeometricNeutralizer


class ChainReasoningHead(CliffordModule):
    """Soft-gated rotor bank for compositional chain reasoning.

    Learns K relation rotors as a geometric basis. A gating MLP maps
    the pooled grade-0 features to softmax weights over K rotors.
    All K rotors are applied, and the weighted-sum of transformed
    grade-0 features is classified.

    The key insight: the rotor bank provides a learned geometric basis
    for relation composition, with soft gating selecting the composition.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int,
                 num_relations: int = 18, hidden_dim: int = 64):
        super().__init__(algebra)
        self.channels = channels
        self.num_relations = num_relations

        # Learned relation rotors -- each captures a geometric transformation
        self.relation_rotors = nn.ModuleList([
            RotorLayer(algebra, channels) for _ in range(num_relations)
        ])

        # Gating MLP: grade-0 features -> softmax weights over K rotors
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations),
        )

        # Grade-0 (scalar) features -> classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft-gated rotor bank and classify.

        Args:
            x: Pooled multivectors [B, C, D] from backbone.

        Returns:
            Logits [B, num_relations].
        """
        B, C, D = x.shape

        # Compute gating weights from grade-0 features
        g0 = x[..., 0]  # [B, C]
        gate_weights = F.softmax(self.gate(g0), dim=-1)  # [B, K]

        # Apply all K rotors and weighted-sum their grade-0 outputs
        transformed_g0 = torch.zeros(B, self.num_relations, C, device=x.device, dtype=x.dtype)
        for k, rotor_layer in enumerate(self.relation_rotors):
            R, R_rev = rotor_layer._compute_rotors(x.device, x.dtype)
            Rx = self.algebra.geometric_product(R.unsqueeze(0), x)
            RxRr = self.algebra.geometric_product(Rx, R_rev.unsqueeze(0))
            transformed_g0[:, k] = RxRr[..., 0]  # [B, C]

        # Weighted sum: [B, K, C] * [B, K, 1] -> [B, C]
        weighted = (transformed_g0 * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, C]

        return self.classifier(weighted)

    def isometry_loss(self) -> torch.Tensor:
        """Regularization: relation rotors should preserve norms."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for rotor in self.relation_rotors:
            bw = rotor.bivector_weights  # [C, num_bv]
            loss = loss + (bw.pow(2).sum(dim=-1) - 1.0).pow(2).mean()
        return loss / max(len(self.relation_rotors), 1)


class EntailmentHead(CliffordModule):
    """Asymmetric entailment via geometric product structure (binary).

    Computes P * rev(H) where P=premise, H=hypothesis multivectors.
    The grade-0 part (symmetric alignment) and grade-2 part (antisymmetric
    orientation) provide naturally asymmetric features.

    Binary output: entailment (1) vs non-entailment (0), matching HANS protocol.

    Key: <P * rev(H)>_2 flips sign when P <-> H are swapped,
    giving the model asymmetry for free from the algebra.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, hidden_dim: int = 64):
        super().__init__(algebra)
        self.channels = channels

        g2_mask = algebra.grade_masks[2]
        self.register_buffer('g2_idx', g2_mask.nonzero(as_tuple=False).squeeze(-1))
        self.d2 = len(self.g2_idx)

        # Features: grade-0 (1) + grade-2 norm (1) + grade-2 direction (min(d2, 4))
        feature_dim = channels * (1 + 1 + min(self.d2, 4))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _compute_product_features(self, premise: torch.Tensor, hypothesis: torch.Tensor):
        """Compute geometric product and extract grade features.

        Returns:
            (g0, g2, g2_norm, g2_dir, features) tuple.
        """
        H_rev = self.algebra.reverse(hypothesis)
        product = self.algebra.geometric_product(premise, H_rev)  # [B, C, D]

        g0 = product[..., 0]                    # [B, C] -- symmetric
        g2 = product[..., self.g2_idx]           # [B, C, d2] -- antisymmetric
        g2_norm = g2.norm(dim=-1)                # [B, C]

        k = min(self.d2, 4)
        g2_dir = g2[..., :k]                    # [B, C, k]

        features = torch.cat([
            g0,
            g2_norm,
            g2_dir.reshape(g2_dir.shape[0], -1),
        ], dim=-1)

        return g0, g2, g2_norm, g2_dir, features

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        """Compute binary entailment logits from geometric product features.

        Args:
            premise: Premise multivectors [B, C, D].
            hypothesis: Hypothesis multivectors [B, C, D].

        Returns:
            Logits [B, 1].
        """
        _, _, _, _, features = self._compute_product_features(premise, hypothesis)
        return self.classifier(features)

    def get_grade2_stats(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> dict:
        """Diagnostic: grade-2 signal statistics.

        Returns dict with g2_norm mean/std/max across the batch.
        """
        _, _, g2_norm, _, _ = self._compute_product_features(premise, hypothesis)
        g2_flat = g2_norm.flatten()
        return {
            "g2_norm_mean": g2_flat.mean().item(),
            "g2_norm_std": g2_flat.std().item(),
            "g2_norm_max": g2_flat.max().item(),
        }


class NegationHead(CliffordModule):
    """Negation sensitivity via grade involution.

    Grade involution x^ = sum (-1)^k <x>_k flips odd grades and
    preserves even grades. This is an algebraic automorphism.

    The head measures the involution-distance between features and
    uses GeometricNeutralizer to separate truth (grade-0) from
    relational noise (grade-2).
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, hidden_dim: int = 64):
        super().__init__(algebra)
        self.channels = channels
        self.neutralizer = GeometricNeutralizer(algebra, channels)

        # Features: grade-0 (neutralized) + involution distance + original grade-0
        feature_dim = channels * 3
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, passage_mv: torch.Tensor, question_mv: torch.Tensor) -> torch.Tensor:
        """Predict answer using grade involution structure.

        Args:
            passage_mv: Passage multivectors [B, C, D].
            question_mv: Question multivectors [B, C, D].

        Returns:
            Logits [B, 1].
        """
        combined = self.algebra.geometric_product(passage_mv, question_mv)
        involuted = self.algebra.grade_involution(combined)

        inv_dist = (combined - involuted).norm(dim=-1)   # [B, C]
        neutralized = self.neutralizer(combined)
        g0_neutralized = neutralized[..., 0]             # [B, C]
        g0_original = combined[..., 0]                   # [B, C]

        features = torch.cat([g0_neutralized, g0_original, inv_dist], dim=-1)
        return self.classifier(features)

    def get_features(self, passage_mv: torch.Tensor, question_mv: torch.Tensor) -> torch.Tensor:
        """Get intermediate multivector features (for InvolutionConsistencyLoss)."""
        return self.algebra.geometric_product(passage_mv, question_mv)
