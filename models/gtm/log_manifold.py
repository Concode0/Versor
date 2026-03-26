# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""LogManifoldProjector: mantissa/exponent split with cross-modulation.

Splits x = mantissa * exp(exponent) for high-depth stability.
The mantissa lives on the unit manifold; the exponent captures log-scale.
Cross-modulation gates (initialized near zero) gradually couple the two
as training progresses.
"""

import torch
import torch.nn as nn
from core.algebra import CliffordAlgebra
from core.metric import hermitian_norm
from layers.primitives.base import CliffordModule


class LogManifoldProjector(CliffordModule):
    """Split/merge multivectors into mantissa (unit) and exponent (log-scale).

    Split: x -> (mantissa, exponent) where mantissa = x / ||x||_H, exponent = log(||x||_H)
    Merge: (mantissa, exponent) -> mantissa * exp(exponent) with cross-modulation

    Cross-modulation gates start near zero (logit=-5 -> sigmoid~0.007) so the
    projector initially behaves as a pure split/merge. As training progresses,
    the gates open to enable information flow between scale and direction.
    """

    def __init__(self, algebra: CliffordAlgebra, gate_init: float = -5.0):
        super().__init__(algebra)
        # Cross-modulation gates (initialized near zero)
        self.gate_e = nn.Parameter(torch.tensor(gate_init))  # exponent <- mantissa feedback
        self.gate_m = nn.Parameter(torch.tensor(gate_init))  # mantissa <- exponent feedback

    def split(self, x: torch.Tensor) -> tuple:
        """Split multivector into unit-norm mantissa and log-scale exponent.

        Args:
            x: Multivector [B, N, D].

        Returns:
            (mantissa [B, N, D], exponent [B, N, 1]).
        """
        norm = hermitian_norm(self.algebra, x).clamp(min=1e-8)  # [B, N, 1]
        mantissa = x / norm
        exponent = torch.log(norm)
        return mantissa, exponent

    def merge(self, mantissa: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        """Merge mantissa and exponent back into full multivector.

        Cross-modulation allows information flow between scale and direction:
        - gate_e: exponent adjusted by mantissa's scalar component
        - gate_m: mantissa scaled by original exponent

        Args:
            mantissa: Unit-norm multivector [B, N, D].
            exponent: Log-scale [B, N, 1].

        Returns:
            Reconstructed multivector [B, N, D].
        """
        # Cross-modulation
        e_mod = exponent + torch.sigmoid(self.gate_e) * mantissa[..., 0:1]
        m_mod = mantissa * (1.0 + torch.sigmoid(self.gate_m) * torch.tanh(exponent))
        return m_mod * torch.exp(e_mod.clamp(-10.0, 10.0))
