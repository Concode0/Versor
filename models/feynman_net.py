# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Feynman Geometric Blade Network.

Multi-Rotor system for symbolic regression of physics equations.
Sparse rotor superposition mirrors the parsimony of symbolic expressions.
"""

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from layers.base import CliffordModule
from layers.linear import CliffordLinear
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from layers.multi_rotor import MultiRotorLayer
from functional.activation import GeometricGELU


# ---------------------------------------------------------------------------
# Blade-name helpers
# ---------------------------------------------------------------------------

def _blade_name(idx: int, n: int) -> str:
    """Return a human-readable blade name from a basis-blade index.

    Examples (n=4):
        idx=0  -> '1'      (scalar)
        idx=1  -> 'e1'     (grade-1)
        idx=3  -> 'e12'    (grade-2, binary 0011 -> bits 0 and 1)
        idx=15 -> 'e1234'  (pseudoscalar)
    """
    if idx == 0:
        return "1"
    bits = [i + 1 for i in range(n) if idx & (1 << i)]
    return "e" + "".join(str(b) for b in bits)


def blade_names_for_algebra(algebra) -> list:
    """Return a list of blade names for every basis element of *algebra*."""
    return [_blade_name(i, algebra.n) for i in range(algebra.dim)]


def bivector_plane_names(algebra) -> list:
    """Return blade names for the grade-2 (bivector / rotation-plane) blades."""
    return [
        _blade_name(i, algebra.n)
        for i in range(algebra.dim)
        if bin(i).count("1") == 2
    ]


class FeynmanMultiGradeEmbedding(CliffordModule):
    """Embeds scalar inputs into multiple Clifford algebra grades.

    Populates:
      - Grade 0: learnable scalar bias per channel.
      - Grade 1: linear projection of raw inputs.
      - Grade 2: linear projection of pairwise products (opt-in).

    Attributes:
        in_features (int): Number of scalar inputs k.
        channels (int): Number of channels C.
        embed_grade2 (bool): Whether to populate grade-2 blades.
        grade0_bias (Parameter): [C] bias added to scalar component.
        grade1_proj (Linear): k -> C*n_g1.
        grade2_proj (Linear | None): k*(k-1)/2 -> C*n_g2  (opt-in).
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_features: int,
        channels: int,
        embed_grade2: bool = False,
    ):
        super().__init__(algebra)
        self.in_features = in_features
        self.channels = channels
        self.embed_grade2 = embed_grade2

        # Precompute grade indices
        dim = algebra.dim
        g1_idx = [i for i in range(dim) if bin(i).count('1') == 1]
        g2_idx = [i for i in range(dim) if bin(i).count('1') == 2]
        self.n_g1 = len(g1_idx)
        self.n_g2 = len(g2_idx)

        self.register_buffer('g1_idx', torch.tensor(g1_idx, dtype=torch.long))
        self.register_buffer('g2_idx', torch.tensor(g2_idx, dtype=torch.long))

        # Grade-0: scalar bias per channel
        self.grade0_bias = nn.Parameter(torch.zeros(channels))

        # Grade-1: project k inputs -> C * n_g1
        self.grade1_proj = nn.Linear(in_features, channels * self.n_g1, bias=False)

        # Grade-2: project pairwise products -> C * n_g2  (opt-in)
        n_pairs = in_features * (in_features - 1) // 2
        if embed_grade2 and n_pairs > 0:
            self.grade2_proj = nn.Linear(n_pairs, channels * self.n_g2, bias=False)
        else:
            self.grade2_proj = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.grade1_proj.weight, std=0.01)
        if self.grade2_proj is not None:
            nn.init.normal_(self.grade2_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scalar inputs into the multivector space.

        Args:
            x (Tensor): [B, k] scalar inputs.

        Returns:
            Tensor: [B, C, 2^p] multivectors.
        """
        B = x.size(0)
        out = torch.zeros(B, self.channels, self.algebra.dim,
                          device=x.device, dtype=x.dtype)

        # Grade-0 bias
        out[:, :, 0] = self.grade0_bias.unsqueeze(0).expand(B, -1)

        # Grade-1
        g1_feats = self.grade1_proj(x).reshape(B, self.channels, self.n_g1)
        g1_idx = self.g1_idx.view(1, 1, -1).expand(B, self.channels, -1)
        out.scatter_(2, g1_idx, g1_feats)

        # Grade-2 (opt-in)
        if self.grade2_proj is not None and self.in_features >= 2:
            pairs = [
                x[:, i] * x[:, j]
                for i in range(self.in_features)
                for j in range(i + 1, self.in_features)
            ]
            pairs_t = torch.stack(pairs, dim=1)  # [B, n_pairs]
            g2_feats = self.grade2_proj(pairs_t).reshape(B, self.channels, self.n_g2)
            g2_idx = self.g2_idx.view(1, 1, -1).expand(B, self.channels, -1)
            out.scatter_(2, g2_idx, g2_feats)

        return out


class _ResidualBlock(nn.Module):
    """One residual block: Norm -> Linear -> GELU -> MultiRotor -> BladeSelector -> skip."""

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_rotors: int,
        use_decomposition: bool,
    ):
        super().__init__()
        self.norm       = CliffordLayerNorm(algebra, channels)
        self.linear     = CliffordLinear(algebra, channels, channels)
        self.activation = GeometricGELU(algebra, channels)
        self.multi_rotor = MultiRotorLayer(
            algebra, channels,
            num_rotors=num_rotors,
            use_decomposition=use_decomposition,
        )
        self.blade = BladeSelector(algebra, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.linear(out)
        out = self.activation(out)
        out = self.multi_rotor(out)
        out = self.blade(out)
        return out + x   # residual skip


class FeynmanGBN(CliffordModule):
    """Geometric Blade Network for Feynman symbolic regression.

    Architecture:
        FeynmanMultiGradeEmbedding -> N residual blocks -> output head -> scalar

    Each residual block: CliffordLayerNorm -> CliffordLinear -> GeometricGELU
                         -> MultiRotorLayer -> BladeSelector -> +skip

    Output head: final norm -> BladeSelector -> CliffordLinear(C->1)
                 -> nn.Linear(dim->1)

    The sparsity loss on rotor bivectors + weights encourages parsimonious
    decompositions (few active rotation planes = simple symbolic structure).

    Attributes:
        in_features (int): Number of scalar inputs.
        channels (int): Hidden channels C.
        embedding (FeynmanMultiGradeEmbedding): Input embedding.
        blocks (ModuleList): Residual processing blocks.
        output_* : Output projection head.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_features: int,
        channels: int = 16,
        num_layers: int = 3,
        num_rotors: int = 8,
        embed_grade2: bool = False,
        use_decomposition: bool = False,
    ):
        super().__init__(algebra)
        self.in_features = in_features
        self.channels = channels

        self.embedding = FeynmanMultiGradeEmbedding(
            algebra, in_features, channels, embed_grade2
        )

        self.blocks = nn.ModuleList([
            _ResidualBlock(algebra, channels, num_rotors, use_decomposition)
            for _ in range(num_layers)
        ])

        self.output_norm   = CliffordLayerNorm(algebra, channels)
        self.output_blade  = BladeSelector(algebra, channels)
        self.output_linear = CliffordLinear(algebra, channels, 1)
        self.output_scalar = nn.Linear(algebra.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): [B, k] normalised scalar inputs.

        Returns:
            Tensor: [B, 1] predicted (normalised) scalar output.
        """
        out = self.embedding(x)      # [B, C, dim]

        for block in self.blocks:
            out = block(out)         # [B, C, dim]

        # Cache last hidden state (detached) for analysis / ortho monitoring.
        self._last_hidden = out.detach()   # [B, C, dim]

        out = self.output_norm(out)
        out = self.output_blade(out)
        out = self.output_linear(out)          # [B, 1, dim]
        out = self.output_scalar(out.squeeze(1))  # [B, 1]

        return out

    def get_rotor_analysis(self) -> list:
        """Per-block rotor activity and dominant rotation planes.

        Returns:
            List of dicts (one per residual block), each containing:
                layer           - block index
                weights         - Tensor [C, K] raw mixing weights
                bivectors       - Tensor [K, n_bv] raw bivector parameters
                plane_names     - list[str] names for each of the n_bv bivectors
                rotor_activity  - list[float] mean |weight| per rotor [K]
                dominant_planes - list[str] dominant plane name per rotor [K]
        """
        plane_names = bivector_plane_names(self.algebra)
        results = []
        for i, block in enumerate(self.blocks):
            mr = block.multi_rotor
            w_mag = mr.weights.detach().abs().mean(0)   # [K]
            bv    = mr.rotor_bivectors.detach().abs()    # [K, n_bv]
            dom   = bv.argmax(dim=1)                    # [K]
            results.append({
                "layer":          i,
                "weights":        mr.weights.detach().cpu(),           # [C, K]
                "bivectors":      mr.rotor_bivectors.detach().cpu(),   # [K, n_bv]
                "plane_names":    plane_names,
                "rotor_activity": w_mag.tolist(),                      # [K]
                "dominant_planes": [plane_names[j] for j in dom.tolist()],
            })
        return results

    def get_output_blade_weights(self, algebra) -> dict:
        """Map each basis-blade name -> its weight in the final scalar projection.

        Returns:
            dict {blade_name: weight_value}
        """
        w     = self.output_scalar.weight.detach()[0]  # [dim]
        names = blade_names_for_algebra(algebra)
        return {name: w[i].item() for i, name in enumerate(names)}

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 sparsity losses over all MultiRotorLayer instances.

        Encourages fewer active rotation planes, mirroring symbolic parsimony.
        """
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, MultiRotorLayer):
                total = total + module.sparsity_loss()
        return total
