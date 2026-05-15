"""Rotor-based linear transformation layer (Generalized Rotor Gadget).

Implements Section 4.2 from Pence et al. (2025), "Composing Linear Layers
from Irreducibles." Replaces standard linear layers with parameter-efficient
rotor-sandwich transformations.

Reference:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1, Section 4.2, Equation 6
"""

from typing import Literal

import torch
import torch.nn as nn

from core.foundation.manifold import MANIFOLD_SPIN, tag_manifold
from core.foundation.module import CliffordModule
from core.foundation.validation import check_channels, check_multivector
from core.runtime.algebra import CliffordAlgebra

from ._utils import (
    cache_matches,
    channel_mix,
    dense_from_indices,
    grade_indices,
    pair_mean,
    require_choice,
    require_positive_int,
)


class RotorGadget(CliffordModule):
    """Rotor-based linear transformation (Generalized Rotor Gadget).

    Replaces standard linear layers with parameter-efficient rotor-sandwich
    transformations. Instead of using O(in_channels x out_channels) parameters,
    this uses O(num_rotor_pairs x n(n-1)/2) parameters where n is the number
    of basis vectors in the Clifford algebra.

    Architecture:
        1. Partition input channels into blocks
        2. For each rotor pair (i, j):
           - Apply rotor sandwich: r_ij . x_i . s_ij.H
        3. Pool/aggregate results to output channels

    The transformation is: psi(x) = r.x.s.H where r, s are rotors (bivector exponentials).

    Attributes:
        algebra: CliffordAlgebra instance
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_rotor_pairs: Number of rotor pairs to use
        aggregation: Aggregation method ('mean', 'sum', or 'learned')
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_channels: int,
        out_channels: int,
        num_rotor_pairs: int = 4,
        aggregation: Literal["mean", "sum", "learned"] = "mean",
        shuffle: Literal["none", "fixed", "random"] = "none",
        bias: bool = False,
    ):
        """Initialize rotor gadget layer.

        Args:
            algebra: CliffordAlgebra instance
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_rotor_pairs: Number of rotor pairs (higher = more expressive)
            aggregation: How to pool rotor outputs ('mean', 'sum', 'learned')
            shuffle: Input channel shuffle strategy:
                - 'none': No shuffle, sequential block assignment (default)
                - 'fixed': Random permutation at initialization (fixed during training)
                - 'random': Random permutation each forward pass (regularization)
            bias: Whether to include bias term (applied after transformation)
        """
        super().__init__(algebra)
        if not hasattr(algebra, "per_channel_sandwich"):
            raise ValueError("RotorGadget is dense-only and requires CliffordAlgebra.")

        self.in_channels = require_positive_int(in_channels, "in_channels")
        self.out_channels = require_positive_int(out_channels, "out_channels")
        self.num_rotor_pairs = require_positive_int(num_rotor_pairs, "num_rotor_pairs")
        self.aggregation = require_choice(aggregation, "aggregation", ("mean", "sum", "learned"))
        self.shuffle = require_choice(shuffle, "shuffle", ("none", "fixed", "random"))

        if algebra.num_grades <= 2:
            raise ValueError(f"Algebra has no bivectors. RotorGadget requires at least one bivector for rotation.")
        self.register_buffer("bivector_indices", grade_indices(algebra, 2, name="bivector grade"))
        self.num_bivectors = self.bivector_indices.numel()

        # Rotor parameters: bivector coefficients for exponential map
        # Left rotors: [num_rotor_pairs, num_bivectors]
        self.bivector_left = nn.Parameter(torch.randn(self.num_rotor_pairs, self.num_bivectors) * 0.1)
        tag_manifold(self.bivector_left, MANIFOLD_SPIN)
        # Right rotors: [num_rotor_pairs, num_bivectors]
        self.bivector_right = nn.Parameter(torch.randn(self.num_rotor_pairs, self.num_bivectors) * 0.1)
        tag_manifold(self.bivector_right, MANIFOLD_SPIN)

        # Channel routing: block diagonal partitioning (paper style)
        # Each rotor pair processes a subset of input channels
        self._setup_channel_routing()

        # Aggregation weights (if learned)
        if self.aggregation == "learned":
            self.agg_weights = nn.Parameter(torch.ones(self.num_rotor_pairs, self.out_channels) / self.num_rotor_pairs)
        else:
            self.register_buffer("agg_weights", None)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, algebra.dim))
        else:
            self.register_buffer("bias", None)

        # Rotor cache for eval mode
        self._cached_rotors = None

    def _setup_channel_routing(self):
        """Set up block diagonal channel routing with optional shuffle.

        Partitions input and output channels into blocks, where each rotor
        pair operates on a specific block. Optionally shuffles input channels
        before routing for regularization.
        """
        in_assignment = torch.div(
            torch.arange(self.in_channels) * self.num_rotor_pairs,
            self.in_channels,
            rounding_mode="floor",
        ).clamp_max(self.num_rotor_pairs - 1)
        out_assignment = torch.div(
            torch.arange(self.out_channels) * self.num_rotor_pairs,
            self.out_channels,
            rounding_mode="floor",
        ).clamp_max(self.num_rotor_pairs - 1)

        in_indices = []
        out_indices = []
        for i in range(self.num_rotor_pairs):
            in_members = (in_assignment == i).nonzero(as_tuple=False).flatten()
            out_members = (out_assignment == i).nonzero(as_tuple=False).flatten()
            if in_members.numel() == 0:
                in_indices.append((self.in_channels, self.in_channels))
            else:
                in_indices.append((int(in_members[0]), int(in_members[-1]) + 1))
            if out_members.numel() == 0:
                out_indices.append((self.out_channels, self.out_channels))
            else:
                out_indices.append((int(out_members[0]), int(out_members[-1]) + 1))

        self.in_indices = in_indices
        self.out_indices = out_indices

        ch2pair = in_assignment.to(dtype=torch.long)
        self.register_buffer("_ch2pair", ch2pair)
        self.register_buffer("_channel_mix_mean", channel_mix(self.in_channels, self.out_channels, normalize=True))
        self.register_buffer("_channel_mix_sum", channel_mix(self.in_channels, self.out_channels, normalize=False))
        self.register_buffer("_pair_mean", pair_mean(ch2pair, self.num_rotor_pairs))

        # Set up channel shuffle permutation
        if self.shuffle == "fixed":
            # Create fixed random permutation at initialization
            perm = torch.randperm(self.in_channels)
            self.register_buffer("channel_permutation", perm)
        elif self.shuffle == "random":
            # Random shuffle each forward pass - no fixed permutation
            self.register_buffer("channel_permutation", None)
        else:  # 'none'
            # No shuffle - identity permutation
            self.register_buffer("channel_permutation", None)

    def _bivector_to_multivector(self, bivector_coeffs: torch.Tensor) -> torch.Tensor:
        """Convert bivector coefficients to full multivector via vectorized scatter.

        Args:
            bivector_coeffs: Tensor of shape [..., num_bivectors]

        Returns:
            Multivector tensor of shape [..., algebra.dim]
        """
        return dense_from_indices(bivector_coeffs, self.bivector_indices, self.algebra.dim)

    def _compute_rotors(self, device=None, dtype=None):
        """Compute rotor multivectors from bivector parameters.

        Returns:
            Tuple of (left_rotors, right_rotors_reversed) where each is
            a tensor of shape [num_rotor_pairs, algebra.dim]
        """
        left = self.bivector_left
        right = self.bivector_right
        if device is not None or dtype is not None:
            left = left.to(device=device, dtype=dtype)
            right = right.to(device=device, dtype=dtype)

        # Convert bivector parameters to multivectors
        B_left = self._bivector_to_multivector(left)  # [pairs, dim]
        B_right = self._bivector_to_multivector(right)  # [pairs, dim]

        # Compute rotors via exponential map: R = exp(-0.5 * B)
        R_left = self.algebra.exp(-0.5 * B_left)  # [pairs, dim]
        R_right = self.algebra.exp(-0.5 * B_right)  # [pairs, dim]

        # Compute reverse of right rotors for sandwich product
        R_right_rev = self.algebra.reverse(R_right)  # [pairs, dim]

        return R_left, R_right_rev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotor-based transformation.

        Uses batched geometric products - all rotor pairs are applied in
        parallel via a single pair of GP calls.

        Args:
            x: Input tensor of shape [Batch, In_Channels, Dim]

        Returns:
            Output tensor of shape [Batch, Out_Channels, Dim]
        """
        check_multivector(x, self.algebra, "RotorGadget input")
        check_channels(x, self.in_channels, "RotorGadget input")

        # Apply input channel shuffle if enabled
        if self.shuffle == "fixed":
            x = x.index_select(-2, self.channel_permutation)
        elif self.shuffle == "random":
            perm = torch.randperm(self.in_channels, device=x.device)
            x = x.index_select(-2, perm)

        # Compute rotors (cached in eval mode)
        if not self.training and cache_matches(self._cached_rotors, x):
            R_left, R_right_rev = self._cached_rotors
        else:
            R_left, R_right_rev = self._compute_rotors(x.device, x.dtype)
            if not self.training:
                self._cached_rotors = (R_left, R_right_rev)

        ch2pair = self._ch2pair.to(device=R_left.device)
        R_left_by_channel = R_left[ch2pair]
        R_right_by_channel = R_right_rev[ch2pair]
        concat_out = self.algebra.per_channel_sandwich(R_left_by_channel, x, R_right_by_channel)

        # Map to output channels
        out = self._aggregate_to_output_channels(concat_out)

        if self.bias is not None:
            bias_shape = (1,) * (out.ndim - 2) + (self.out_channels, self.algebra.dim)
            out = out + self.bias.view(bias_shape)

        return out

    def _aggregate_to_output_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate rotor pair outputs to match output channel count.

        Args:
            x: Concatenated outputs from rotor pairs [B, total_channels, dim]

        Returns:
            Aggregated output [B, out_channels, dim]
        """
        if self.aggregation == "learned":
            pair_values = torch.einsum("ki,...id->...kd", self._pair_mean.to(device=x.device, dtype=x.dtype), x)
            return torch.einsum("ko,...kd->...od", self.agg_weights.to(device=x.device, dtype=x.dtype), pair_values)

        mix = self._channel_mix_sum if self.aggregation == "sum" else self._channel_mix_mean
        return torch.einsum("oi,...id->...od", mix.to(device=x.device, dtype=x.dtype), x)

    def train(self, mode: bool = True):
        """Override to invalidate rotor cache when switching to train mode."""
        if mode:
            self._cached_rotors = None
        return super().train(mode)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_rotor_pairs={self.num_rotor_pairs}, "
            f"aggregation={self.aggregation}, "
            f"shuffle={self.shuffle}, "
            f"bias={self.bias is not None}"
        )
