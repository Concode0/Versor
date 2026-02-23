"""Rotor-based linear transformation layer (Generalized Rotor Gadget).

Implements Section 4.2 from Pence et al. (2025), "Composing Linear Layers
from Irreducibles." Replaces standard linear layers with parameter-efficient
rotor-sandwich transformations.

Reference:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1, Section 4.2, Equation 6
"""

import torch
import torch.nn as nn
from typing import Literal

from core.algebra import CliffordAlgebra
from layers.base import CliffordModule


class RotorGadget(CliffordModule):
    """Rotor-based linear transformation (Generalized Rotor Gadget).

    Replaces standard linear layers with parameter-efficient rotor-sandwich
    transformations. Instead of using O(in_channels × out_channels) parameters,
    this uses O(num_rotor_pairs × n(n-1)/2) parameters where n is the number
    of basis vectors in the Clifford algebra.

    Architecture:
        1. Partition input channels into blocks
        2. For each rotor pair (i, j):
           - Apply rotor sandwich: r_ij · x_i · s_ij†
        3. Pool/aggregate results to output channels

    The transformation is: ψ(x) = r·x·s† where r, s are rotors (bivector exponentials).

    Attributes:
        algebra: CliffordAlgebra instance
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_rotor_pairs: Number of rotor pairs to use
        use_decomposition: Whether to use bivector decomposition
        aggregation: Aggregation method ('mean', 'sum', or 'learned')
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_channels: int,
        out_channels: int,
        num_rotor_pairs: int = 4,
        use_decomposition: bool = False,
        decomp_k: int = 10,
        aggregation: Literal['mean', 'sum', 'learned'] = 'mean',
        shuffle: Literal['none', 'fixed', 'random'] = 'none',
        bias: bool = False,
    ):
        """Initialize rotor gadget layer.

        Args:
            algebra: CliffordAlgebra instance
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_rotor_pairs: Number of rotor pairs (higher = more expressive)
            use_decomposition: Use bivector decomposition for efficiency
            decomp_k: Number of iterations for decomposition (if enabled)
            aggregation: How to pool rotor outputs ('mean', 'sum', 'learned')
            shuffle: Input channel shuffle strategy:
                - 'none': No shuffle, sequential block assignment (default)
                - 'fixed': Random permutation at initialization (fixed during training)
                - 'random': Random permutation each forward pass (regularization)
            bias: Whether to include bias term (applied after transformation)
        """
        super().__init__(algebra)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rotor_pairs = num_rotor_pairs
        self.use_decomposition = use_decomposition
        self.decomp_k = decomp_k
        self.aggregation = aggregation
        self.shuffle = shuffle

        # Use algebra's precomputed grade masks for bivector indices
        if algebra.num_grades > 2:
            bv_mask = algebra.grade_masks[2]
            self.register_buffer('bivector_indices', bv_mask.nonzero(as_tuple=False).squeeze(-1))
        else:
            self.register_buffer('bivector_indices', torch.tensor([], dtype=torch.long, device=algebra.device))
        self.num_bivectors = len(self.bivector_indices)

        if self.num_bivectors == 0:
            raise ValueError(
                f"Algebra has no bivectors. RotorGadget requires "
                "at least one bivector for rotation."
            )

        # Rotor parameters: bivector coefficients for exponential map
        # Left rotors: [num_rotor_pairs, num_bivectors]
        self.bivector_left = nn.Parameter(
            torch.randn(num_rotor_pairs, self.num_bivectors) * 0.1
        )
        # Right rotors: [num_rotor_pairs, num_bivectors]
        self.bivector_right = nn.Parameter(
            torch.randn(num_rotor_pairs, self.num_bivectors) * 0.1
        )

        # Channel routing: block diagonal partitioning (paper style)
        # Each rotor pair processes a subset of input channels
        self._setup_channel_routing()

        # Aggregation weights (if learned)
        if aggregation == 'learned':
            # Learned weights for combining rotor outputs
            self.agg_weights = nn.Parameter(
                torch.ones(num_rotor_pairs, out_channels) / num_rotor_pairs
            )
        else:
            self.register_buffer('agg_weights', None)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, algebra.dim))
        else:
            self.register_buffer('bias', None)

        # Rotor cache for eval mode
        self._cached_rotors = None

    def _setup_channel_routing(self):
        """Set up block diagonal channel routing with optional shuffle.

        Partitions input and output channels into blocks, where each rotor
        pair operates on a specific block. Optionally shuffles input channels
        before routing for regularization.
        """
        # Compute block sizes
        in_block_size = max(1, self.in_channels // self.num_rotor_pairs)
        out_block_size = max(1, self.out_channels // self.num_rotor_pairs)

        # Create routing indices
        in_indices = []
        out_indices = []

        for i in range(self.num_rotor_pairs):
            # Input block for this rotor pair
            in_start = i * in_block_size
            in_end = min((i + 1) * in_block_size, self.in_channels)
            in_indices.append((in_start, in_end))

            # Output block for this rotor pair
            out_start = i * out_block_size
            out_end = min((i + 1) * out_block_size, self.out_channels)
            out_indices.append((out_start, out_end))

        self.in_indices = in_indices
        self.out_indices = out_indices

        # Set up channel shuffle permutation
        if self.shuffle == 'fixed':
            # Create fixed random permutation at initialization
            perm = torch.randperm(self.in_channels)
            self.register_buffer('channel_permutation', perm)
        elif self.shuffle == 'random':
            # Random shuffle each forward pass - no fixed permutation
            self.register_buffer('channel_permutation', None)
        else:  # 'none'
            # No shuffle - identity permutation
            self.register_buffer('channel_permutation', None)

    def _bivector_to_multivector(self, bivector_coeffs: torch.Tensor) -> torch.Tensor:
        """Convert bivector coefficients to full multivector via vectorized scatter.

        Args:
            bivector_coeffs: Tensor of shape [..., num_bivectors]

        Returns:
            Multivector tensor of shape [..., algebra.dim]
        """
        batch_shape = bivector_coeffs.shape[:-1]
        mv = torch.zeros(*batch_shape, self.algebra.dim,
                          device=bivector_coeffs.device, dtype=bivector_coeffs.dtype)
        # Expand indices to match batch shape for scatter_
        idx = self.bivector_indices.expand(*batch_shape, -1)
        mv.scatter_(-1, idx, bivector_coeffs)
        return mv

    def _compute_rotors(self):
        """Compute rotor multivectors from bivector parameters.

        Returns:
            Tuple of (left_rotors, right_rotors_reversed) where each is
            a tensor of shape [num_rotor_pairs, algebra.dim]
        """
        # Convert bivector parameters to multivectors
        B_left = self._bivector_to_multivector(self.bivector_left)  # [pairs, dim]
        B_right = self._bivector_to_multivector(self.bivector_right)  # [pairs, dim]

        # Compute rotors via exponential map: R = exp(-0.5 * B)
        if self.use_decomposition:
            # Use decomposed exponential (more efficient)
            R_left = self.algebra.exp_decomposed(
                -0.5 * B_left,
                use_decomposition=True,
                k=self.decomp_k
            )  # [pairs, dim]
            R_right = self.algebra.exp_decomposed(
                -0.5 * B_right,
                use_decomposition=True,
                k=self.decomp_k
            )  # [pairs, dim]
        else:
            # Standard exponential
            R_left = self.algebra.exp(-0.5 * B_left)  # [pairs, dim]
            R_right = self.algebra.exp(-0.5 * B_right)  # [pairs, dim]

        # Compute reverse of right rotors for sandwich product
        R_right_rev = self.algebra.reverse(R_right)  # [pairs, dim]

        return R_left, R_right_rev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotor-based transformation.

        Uses batched geometric products — all rotor pairs are applied in
        parallel via a single pair of GP calls.

        Args:
            x: Input tensor of shape [Batch, In_Channels, Dim]

        Returns:
            Output tensor of shape [Batch, Out_Channels, Dim]
        """
        from core.validation import check_multivector, check_channels
        check_multivector(x, self.algebra, "RotorGadget input")
        check_channels(x, self.in_channels, "RotorGadget input")

        self.algebra.ensure_device(x.device)

        # Apply input channel shuffle if enabled
        if self.shuffle == 'fixed':
            x = x[:, self.channel_permutation, :]
        elif self.shuffle == 'random':
            perm = torch.randperm(self.in_channels, device=x.device)
            x = x[:, perm, :]

        # Compute rotors (cached in eval mode)
        if not self.training and self._cached_rotors is not None:
            R_left, R_right_rev = self._cached_rotors
        else:
            R_left, R_right_rev = self._compute_rotors()
            if not self.training:
                self._cached_rotors = (R_left, R_right_rev)

        # Vectorized sandwich: apply each rotor pair to its channel block
        # Build expanded rotor tensors [1, in_channels, dim] where each channel
        # gets the rotor for its assigned pair
        D = self.algebra.dim
        R_left_expanded = torch.zeros(1, self.in_channels, D,
                                       device=x.device, dtype=x.dtype)
        R_right_expanded = torch.zeros(1, self.in_channels, D,
                                        device=x.device, dtype=x.dtype)

        for i in range(self.num_rotor_pairs):
            in_start, in_end = self.in_indices[i]
            if in_end > in_start:
                R_left_expanded[0, in_start:in_end] = R_left[i]
                R_right_expanded[0, in_start:in_end] = R_right_rev[i]

        # Two batched GPs instead of 2*K sequential GPs
        temp = self.algebra.geometric_product(R_left_expanded, x)
        concat_out = self.algebra.geometric_product(temp, R_right_expanded)

        # Map to output channels
        out = self._aggregate_to_output_channels(concat_out)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)

        return out

    def _aggregate_to_output_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate rotor pair outputs to match output channel count.

        Args:
            x: Concatenated outputs from rotor pairs [B, total_channels, dim]

        Returns:
            Aggregated output [B, out_channels, dim]
        """
        batch_size = x.shape[0]

        if self.aggregation == 'learned':
            # Weighted aggregation with learned weights
            # agg_weights: [num_pairs, out_channels]
            # Need to apply per-block
            outputs = []
            for i in range(self.num_rotor_pairs):
                in_start, in_end = self.in_indices[i]
                block_size = in_end - in_start
                if block_size == 0:
                    continue

                x_i = x[:, in_start:in_end, :]  # [B, block, dim]
                # Average over block channels and weight
                x_i_mean = x_i.mean(dim=1, keepdim=True)  # [B, 1, dim]
                # Expand to output channels with weights
                weighted = x_i_mean * self.agg_weights[i:i+1, :, None]  # [B, out_ch, dim]
                outputs.append(weighted)

            out = torch.stack(outputs, dim=0).sum(dim=0)  # [B, out_ch, dim]

        elif self.aggregation == 'sum':
            # Simple channel-wise sum with reshaping
            if x.shape[1] == self.out_channels:
                out = x
            elif x.shape[1] > self.out_channels:
                # Pool down by summing
                fold = x.shape[1] // self.out_channels
                out = x[:, :fold*self.out_channels, :].reshape(
                    batch_size, self.out_channels, fold, self.algebra.dim
                ).sum(dim=2)
            else:
                # Expand by tiling
                repeats = (self.out_channels + x.shape[1] - 1) // x.shape[1]
                out = x.repeat(1, repeats, 1)[:, :self.out_channels, :]

        else:  # 'mean'
            # Mean pooling
            if x.shape[1] == self.out_channels:
                out = x
            elif x.shape[1] > self.out_channels:
                # Pool down by averaging
                fold = x.shape[1] // self.out_channels
                out = x[:, :fold*self.out_channels, :].reshape(
                    batch_size, self.out_channels, fold, self.algebra.dim
                ).mean(dim=2)
            else:
                # Expand by tiling
                repeats = (self.out_channels + x.shape[1] - 1) // x.shape[1]
                out = x.repeat(1, repeats, 1)[:, :self.out_channels, :]

        return out

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
            f"use_decomposition={self.use_decomposition}, "
            f"bias={self.bias is not None}"
        )
