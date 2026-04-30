# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from core.module import CliffordModule

from ..blocks.attention import GeometricProductAttention
from ..primitives.normalization import CliffordLayerNorm


class MotherEmbedding(CliffordModule):
    """Embeds local feature groups into a canonical Mother Algebra with Procrustes Alignment.

    Uses fixed rotors (R_fixed) to rotate individual channel vectors into a shared
    reference frame, effectively aligning disparate geometric manifolds.
    """

    def __init__(self, algebra: CliffordAlgebra, input_dim: int, channels: int, U: float = 0.0, V: torch.Tensor = None):
        """Initializes the Mother Embedding.

        Args:
            algebra: Clifford algebra instance.
            input_dim: Dimension of the input features.
            channels: Number of multivector channels.
            U: Geometric uncertainty index for manifold suppression.
            V: Fixed rotor proxy for Procrustes alignment (input_dim x input_dim).
        """
        super().__init__(algebra)
        self.channels = channels

        # Procrustes Alignment Matrix (Fixed Rotor Proxy)
        if V is None:
            V = torch.eye(input_dim)
        self.register_buffer("R_fixed", V)

        # Up-cast to Mother Algebra multivector channels
        self.linear = nn.Linear(input_dim, channels * algebra.dim)
        self.norm = CliffordLayerNorm(algebra, channels)

        # Pre-condition LayerNorm scale with Uncertainty Index
        with torch.no_grad():
            if hasattr(self.norm, "weight"):
                # Suppress highly uncertain (twisted) manifolds initially
                scale = 1.0 / (1.0 + U)
                self.norm.weight.data.fill_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input into the aligned mother manifold.

        Args:
            x: Input features [B, input_dim].

        Returns:
            Aligned multivectors [B, channels, dim].
        """
        # 1. Apply Geometric Procrustes Alignment
        if self.R_fixed is not None:
            x = x @ self.R_fixed.T

        # 2. Mother Projection
        c = self.linear(x).view(-1, self.channels, self.algebra.dim)
        return self.norm(c)


class EntropyGatedAttention(CliffordModule):
    """Dynamic geometric attention governed by bivector information entropy.

    Segments with high bivector entropy (disordered phase states) are "stiffened"
    or suppressed, allowing only coherent, synchronized states to propagate.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int, num_heads: int, eta: float = 1.0, H_base: float = 0.5):
        """Initializes Entropy-Gated Attention.

        Args:
            algebra: Clifford algebra instance.
            channels: Total multivector channels.
            num_heads: Number of attention heads.
            eta: Gating multiplier.
            H_base: Base entropy threshold.
        """
        super().__init__(algebra)
        self.channels = channels
        self.eta = eta
        self.H_base = H_base
        self.base_attention = GeometricProductAttention(algebra, channels, num_heads, causal=False)

        # Cache bivector indices and float mask for compile-friendly gating
        mask = self.algebra.grade_masks[2]
        self.register_buffer("g2_idx", mask.nonzero(as_tuple=True)[0])
        self.register_buffer("_g2_float_mask", mask.float())

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, return_gating: bool = False
    ) -> torch.Tensor:
        """Applies entropy-gated geometric attention.

        Args:
            x: Input multivectors [B, L, C, D].
            key_padding_mask: Optional [B, L] bool mask where True = padded.
            return_gating: If True, returns entropy and gating values.

        Returns:
            Attended multivectors [B, L, C, D].
        """
        # 1. Calculate Information Entropy of Bivector Energy
        # Summing across multivector components (g2_idx) and across channels (dim 2)
        # x: [B, L, C, D]
        g2_energy = (x[..., self.g2_idx] ** 2).sum(dim=(-1, -2))  # [B, L]

        # Mask padded positions before entropy calc
        if key_padding_mask is not None:
            g2_energy = g2_energy.masked_fill(key_padding_mask, 0.0)

        # Normalize to probability distribution over sequence
        p = g2_energy / (g2_energy.sum(dim=1, keepdim=True) + 1e-8)

        # Shannon Entropy H per batch [B]
        H = -(p * torch.log(p + 1e-8)).sum(dim=1)

        # 2. Base-Adjusted Gating Function
        lambda_dyn = self.eta * torch.sigmoid(H - self.H_base)  # [B]

        # 3. Apply dynamic geometric stiffness
        # Scale the rotational components (bivectors)
        lambda_view = lambda_dyn.view(-1, 1, 1, 1)

        g2_mask = self._g2_float_mask.to(dtype=x.dtype)
        scale = 1.0 + (lambda_view - 1.0) * g2_mask  # [B, 1, 1, D]
        x_gated = x * scale

        out = self.base_attention(x_gated, key_padding_mask=key_padding_mask)

        if return_gating:
            return out, H, lambda_dyn
        return out


class PhaseShiftHead(CliffordModule):
    """Multi-Grade Mixer using Pseudoscalar Phase Delay.

    Resolves the final state by mixing the scalar component (G0) and
    high-grade component (G4) via a learned phase angle theta.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        """Initializes the Phase-Shift Head.

        Args:
            algebra: Clifford algebra instance.
            channels: Number of channels to mix.
        """
        super().__init__(algebra)
        self.channels = channels
        # Learned phase angle theta
        self.theta = nn.Parameter(torch.randn(1, channels, 1) * 0.1)

        # Identify grade-4 pseudoscalar in Cl(3,1)
        mask_g4 = self.algebra.grade_masks[4]
        if mask_g4.sum() > 0:
            self.register_buffer("g4_idx", mask_g4.nonzero(as_tuple=True)[0])
        else:
            # Fallback if algebra doesn't have grade 4
            self.g4_idx = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixes grades using pseudoscalar rotation.

        Args:
            x: Aligned multivectors [B, L, C, D].

        Returns:
            Final prediction [B, 1].
        """
        # Global mean pool over groups/sequence
        x_pool = x.mean(dim=1)  # [B, C, D]

        # Grade-0 (Scalar)
        G0 = x_pool[..., 0:1]

        # Grade-4 (High-grade/Pseudoscalar)
        if self.g4_idx is not None and len(self.g4_idx) > 0:
            # For Cl(3,1), index 15 is typical
            G4 = x_pool[..., self.g4_idx]
        else:
            G4 = torch.zeros_like(G0)

        # Phase Equation: Re( G0 * exp(G4 * theta) )
        # G4 acts as an imaginary unit if G4^2 = -1
        # Result = G0 * cos(theta) - G4 * sin(theta)
        result = G0 * torch.cos(self.theta) - G4 * torch.sin(self.theta)

        # Mean across channels for final scalar output
        return result.mean(dim=1)  # [B, 1]
