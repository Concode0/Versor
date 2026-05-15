# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geometric Algebra Linear Layers.

Supports traditional matrix-based mixing and parameter-efficient rotor-based backends.
"""

from typing import Literal

import torch
import torch.nn as nn

from core.foundation.layout import GradeLayout
from core.foundation.module import CliffordModule
from core.runtime.algebra import CliffordAlgebra

from ._utils import check_multivector_storage, lane_dim, require_choice, require_positive_int, resolve_layer_layout


class CliffordLinear(CliffordModule):
    """Fully connected layer with optional rotor-based backend.

    Can use either:
    - Traditional scalar weight matrix (default, backward compatible)
    - Rotor-based transformation (new, parameter efficient via RotorGadget)

    The traditional backend uses O(in_channels x out_channels) parameters,
    while the rotor backend uses O(num_rotor_pairs x n(n-1)/2) parameters
    where n is the number of basis vectors.

    Attributes:
        in_channels (int): Input features.
        out_channels (int): Output features.
        backend (str): 'traditional' or 'rotor'
        weight (torch.nn.Parameter | None): Weights [Out, In] (traditional backend only).
        bias (torch.nn.Parameter | None): Bias multivector [Out, Dim] (traditional backend only).
        gadget (nn.Module | None): Rotor transformation (rotor backend only).
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_channels: int,
        out_channels: int,
        backend: Literal["traditional", "rotor"] = "traditional",
        num_rotor_pairs: int = 4,
        aggregation: Literal["mean", "sum", "learned"] = "mean",
        shuffle: Literal["none", "fixed", "random"] = "none",
        grades=None,
        layout: GradeLayout = None,
    ):
        """Initialize Clifford Linear.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            in_channels (int): Input size.
            out_channels (int): Output size.
            backend (str): 'traditional' for standard linear layer,
                          'rotor' for rotor-based transformation
            num_rotor_pairs (int): Number of rotor pairs (rotor backend only)
            aggregation (str): Aggregation method (rotor backend only)
            shuffle (str): Input channel shuffle strategy (rotor backend only):
                - 'none': No shuffle (default)
                - 'fixed': Fixed random permutation
                - 'random': Random permutation each forward pass
        """
        super().__init__(algebra)
        self.in_channels = require_positive_int(in_channels, "in_channels")
        self.out_channels = require_positive_int(out_channels, "out_channels")
        self.backend = require_choice(backend, "backend", ("traditional", "rotor"))
        self.layout = resolve_layer_layout(algebra, layout=layout, grades=grades)
        self.lane_dim = lane_dim(algebra, self.layout)

        if self.backend == "traditional":
            self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels))
            self.bias = nn.Parameter(torch.Tensor(self.out_channels, self.lane_dim))
            self.reset_parameters()
            self.gadget = None

        elif self.backend == "rotor":
            if self.layout is not None:
                raise ValueError(
                    "CliffordLinear rotor backend is dense-only; use traditional backend for compact lanes."
                )
            from .rotor_gadget import RotorGadget

            self.gadget = RotorGadget(
                algebra=algebra,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_rotor_pairs=num_rotor_pairs,
                aggregation=aggregation,
                shuffle=shuffle,
                bias=True,  # Include bias in rotor gadget
            )
            self.weight = None
            self.bias = None

    def reset_parameters(self):
        """Initialize weights with Xavier uniform and zero bias."""
        if self.backend == "traditional":
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-mixing linear transformation.

        Args:
            x (torch.Tensor): Input [Batch, In, Dim].

        Returns:
            torch.Tensor: Output [Batch, Out, Dim].
        """
        check_multivector_storage(
            x,
            self.algebra,
            channels=self.in_channels,
            name="CliffordLinear input",
            layout=self.layout,
            allow_dense=self.layout is None or self.layout.dim == self.algebra.dim,
        )

        if self.backend == "traditional":
            out = torch.einsum("oi,...id->...od", self.weight, x)
            bias_shape = (1,) * (x.ndim - 2) + (self.out_channels, self.lane_dim)
            out = out + self.bias.view(bias_shape)
            return out
        return self.gadget(x)

    def extra_repr(self) -> str:
        """String representation for debugging.

        Returns:
            str: Layer parameters description
        """
        parts = [f"in_channels={self.in_channels}", f"out_channels={self.out_channels}", f"backend={self.backend}"]
        if self.layout is not None:
            parts.append(f"grades={self.layout.grades}")
        return ", ".join(parts)
