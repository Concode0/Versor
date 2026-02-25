# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

from core.algebra import CliffordAlgebra
from layers.base import CliffordModule
from layers.linear import CliffordLinear
from layers.normalization import CliffordLayerNorm
from layers.multi_rotor import MultiRotorLayer
from layers.projection import BladeSelector
from functional.activation import GeometricGELU


class MultiRotorFFN(CliffordModule):
    """Embedded Geometric Toolbox - Feed-Forward Network via rotor superposition.

    Standard transformers use: Linear -> GELU -> Linear.
    This replaces that with:

        CliffordLinear(expand) -> CliffordLayerNorm
            -> MultiRotorLayer(K rotors) -> GeometricGELU
            -> CliffordLinear(contract) -> BladeSelector

    The expand step lifts x into a ``ffn_mult x channels`` toolbox subspace.
    ``MultiRotorLayer`` applies K parallel rotors, each exploring a different
    rotation plane - this IS the nonlinearity, not just a scalar gate.
    The contract step projects back to the original channel count.

    Designed as a standalone module so it can be reused in other tasks
    (md17, pdbbind, etc.) beyond the language model.

    Args:
        algebra (CliffordAlgebra): The algebra instance.
        channels (int): Input/output channel count.
        ffn_mult (int): Expansion factor (ffn_channels = channels * ffn_mult).
        num_rotors (int): Number of parallel rotors K in the toolbox.
        use_decomposition (bool): Use power-iteration bivector decomposition.
        decomp_k (int, optional): Number of simple components for decomposition.
        use_rotor_backend (bool): Use RotorGadget backend for CliffordLinear.

    Input/Output shape: ``[B, C, D]`` where D = algebra.dim.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        ffn_mult: int = 4,
        num_rotors: int = 8,
        use_decomposition: bool = False,
        decomp_k: int = None,
        use_rotor_backend: bool = False,
    ):
        super().__init__(algebra)
        self.channels = channels
        ffn_channels = channels * ffn_mult
        backend = 'rotor' if use_rotor_backend else 'traditional'

        self.expand = CliffordLinear(algebra, channels, ffn_channels, backend=backend)
        self.norm = CliffordLayerNorm(algebra, ffn_channels)
        self.toolbox = MultiRotorLayer(
            algebra, ffn_channels, num_rotors,
            use_decomposition=use_decomposition,
            decomp_k=decomp_k,
        )
        self.act = GeometricGELU(algebra, channels=ffn_channels)
        self.contract = CliffordLinear(algebra, ffn_channels, channels, backend=backend)
        self.gate = BladeSelector(algebra, channels)

    def forward(self, x):
        """Applies the geometric toolbox FFN.

        Args:
            x (torch.Tensor): Input ``[B, C, D]``.

        Returns:
            torch.Tensor: Output ``[B, C, D]``.
        """
        h = self.expand(x)    # [B, ffn_channels, D]
        h = self.norm(h)      # [B, ffn_channels, D]
        h = self.toolbox(h)   # [B, ffn_channels, D]  - K-rotor superposition
        h = self.act(h)       # [B, ffn_channels, D]
        h = self.contract(h)  # [B, channels, D]
        h = self.gate(h)      # [B, channels, D]      - per-blade gating
        return h
