# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

import torch
import torch.nn as nn
from typing import Literal, Optional
from core.algebra import CliffordAlgebra
from layers.base import CliffordModule

class CliffordLinear(CliffordModule):
    """Fully connected layer with optional rotor-based backend.

    Can use either:
    - Traditional scalar weight matrix (default, backward compatible)
    - Rotor-based transformation (new, parameter efficient via RotorGadget)

    The traditional backend uses O(in_channels × out_channels) parameters,
    while the rotor backend uses O(num_rotor_pairs × n(n-1)/2) parameters
    where n is the number of basis vectors.

    Attributes:
        in_channels (int): Input features.
        out_channels (int): Output features.
        backend (str): 'traditional' or 'rotor'
        weight (torch.Tensor): Weights [Out, In] (traditional backend only)
        gadget (RotorGadget): Rotor transformation (rotor backend only)
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_channels: int,
        out_channels: int,
        backend: Literal['traditional', 'rotor'] = 'traditional',
        num_rotor_pairs: int = 4,
        use_decomposition: bool = False,
        decomp_k: int = 10,
        aggregation: Literal['mean', 'sum', 'learned'] = 'mean',
        shuffle: Literal['none', 'fixed', 'random'] = 'none',
    ):
        """Sets up the linear layer.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input size.
            out_channels (int): Output size.
            backend (str): 'traditional' for standard linear layer,
                          'rotor' for rotor-based transformation
            num_rotor_pairs (int): Number of rotor pairs (rotor backend only)
            use_decomposition (bool): Use bivector decomposition (rotor backend only)
            decomp_k (int): Decomposition iterations (rotor backend only)
            aggregation (str): Aggregation method (rotor backend only)
            shuffle (str): Input channel shuffle strategy (rotor backend only):
                - 'none': No shuffle (default)
                - 'fixed': Fixed random permutation
                - 'random': Random permutation each forward pass
        """
        super().__init__(algebra)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backend = backend

        if backend == 'traditional':
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = nn.Parameter(torch.Tensor(out_channels, algebra.dim))
            self.reset_parameters()
            self.gadget = None

        elif backend == 'rotor':
            from layers.rotor_gadget import RotorGadget
            self.gadget = RotorGadget(
                algebra=algebra,
                in_channels=in_channels,
                out_channels=out_channels,
                num_rotor_pairs=num_rotor_pairs,
                use_decomposition=use_decomposition,
                decomp_k=decomp_k,
                aggregation=aggregation,
                shuffle=shuffle,
                bias=True,  # Include bias in rotor gadget
            )
            self.weight = None
            self.bias = None

        else:
            raise ValueError(
                f"Unknown backend: {backend}. Must be 'traditional' or 'rotor'."
            )

    def reset_parameters(self):
        """Random initialization. Standard stuff."""
        if self.backend == 'traditional':
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mixes the channels.

        Args:
            x (torch.Tensor): Input [Batch, In, Dim].

        Returns:
            torch.Tensor: Output [Batch, Out, Dim].
        """
        if self.backend == 'traditional':
            # Traditional linear transformation
            # x: [Batch, In, Dim]
            # weight: [Out, In]
            # out: [Batch, Out, Dim]
            out = torch.einsum('oi,bid->bod', self.weight, x)
            out = out + self.bias.unsqueeze(0)
            return out
        else:
            # Rotor-based transformation
            return self.gadget(x)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        if self.backend == 'traditional':
            return f"in_channels={self.in_channels}, out_channels={self.out_channels}, backend=traditional"
        else:
            return f"in_channels={self.in_channels}, out_channels={self.out_channels}, backend=rotor"