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
from core.algebra import CliffordAlgebra
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from layers.attention import GeometricProductAttention
from layers.multi_rotor_ffn import MultiRotorFFN
from layers.embedding import MultivectorEmbedding, RotaryBivectorPE


class GATransformerBlock(nn.Module):
    """One layer of the GA-native Transformer.

    Pre-norm architecture:

        x -> Norm -> Attention -> residual ->
          -> Norm -> MultiRotorFFN -> residual -> x'

    The Norm layers are CliffordLayerNorm.
    The FFN is a MultiRotorFFN
    (the "Embedded Geometric Toolbox") which applies K parallel rotors
    in an expanded subspace rather than a plain MLP.

    Args:
        algebra (CliffordAlgebra): Algebra instance.
        channels (int): Number of multivector channels.
        num_heads (int): Number of attention heads.
        ffn_mult (int): FFN expansion factor.
        num_rotors (int): Number of rotors in the geometric toolbox.
        causal (bool): Autoregressive causal mask.
        dropout (float): Dropout rate on attention weights.
        use_rotor_backend (bool): RotorGadget backend for CliffordLinear.
        use_decomposition (bool): Bivector decomposition in rotors.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        num_heads: int,
        ffn_mult: int = 4,
        num_rotors: int = 8,
        causal: bool = True,
        dropout: float = 0.0,
        use_rotor_backend: bool = False,
        use_decomposition: bool = False,
    ):
        super().__init__()
        self.algebra = algebra

        self.norm1 = CliffordLayerNorm(algebra, channels)
        self.attn = GeometricProductAttention(
            algebra, channels, num_heads,
            causal=causal, dropout=dropout,
        )
        self.norm2 = CliffordLayerNorm(algebra, channels)
        self.ffn = MultiRotorFFN(
            algebra, channels,
            ffn_mult=ffn_mult,
            num_rotors=num_rotors,
            use_decomposition=use_decomposition,
            use_rotor_backend=use_rotor_backend,
        )

    def _seq_apply(self, layer, x: torch.Tensor) -> torch.Tensor:
        """Applies a [B, C, D] layer to a [B, L, C, D] sequence tensor.

        Flattens B*L, calls the layer, then unflattens back.

        Args:
            layer: Any module expecting ``[B, C, D]``.
            x (torch.Tensor): Sequence tensor ``[B, L, C, D]``.

        Returns:
            torch.Tensor: Output ``[B, L, C, D]``.
        """
        B, L, C, D = x.shape
        return layer(x.reshape(B * L, C, D)).reshape(B, L, C, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs one transformer block.

        Args:
            x (torch.Tensor): ``[B, L, C, D]`` - batch of token sequences.

        Returns:
            torch.Tensor: ``[B, L, C, D]``.
        """
        # Pre-norm attention + residual
        x = x + self.attn(self._seq_apply(self.norm1, x))
        # Pre-norm FFN + residual
        x = x + self._seq_apply(self.ffn, self._seq_apply(self.norm2, x))
        return x


class GALanguageModel(nn.Module):
    """Geometric Algebra native language model.

    Architecture:
        token_ids -> MultivectorEmbedding -> RotaryBivectorPE
            -> N x GATransformerBlock
            -> CliffordLayerNorm -> BladeSelector
            -> grade-0 extraction -> nn.Linear -> logits

    The grade-0 (scalar) part of each token position is extracted before
    the final linear projection.  Scalars are invariants of rotor action
    so they accumulate the "meaning" distilled by each block.

    Args:
        algebra (CliffordAlgebra): Algebra instance (use Cl(3,1)).
        vocab_size (int): Vocabulary size.
        channels (int): Multivector channels per token.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Attention heads per block.
        max_seq_len (int): Maximum sequence length for positional encoding.
        ffn_mult (int): FFN expansion factor.
        num_rotors (int): Rotors in each geometric toolbox FFN.
        causal (bool): Autoregressive causal masking.
        dropout (float): Dropout on attention weights.
        use_rotor_backend (bool): RotorGadget for CliffordLinear layers.
        use_decomposition (bool): Bivector decomposition in rotors.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        vocab_size: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        ffn_mult: int = 4,
        num_rotors: int = 8,
        causal: bool = True,
        dropout: float = 0.0,
        use_rotor_backend: bool = False,
        use_decomposition: bool = False,
    ):
        super().__init__()
        self.algebra = algebra
        self.channels = channels

        self.embed = MultivectorEmbedding(algebra, vocab_size, channels)
        self.pe = RotaryBivectorPE(algebra, channels, max_seq_len)
        self.blocks = nn.ModuleList([
            GATransformerBlock(
                algebra, channels, num_heads,
                ffn_mult=ffn_mult,
                num_rotors=num_rotors,
                causal=causal,
                dropout=dropout,
                use_rotor_backend=use_rotor_backend,
                use_decomposition=use_decomposition,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = CliffordLayerNorm(algebra, channels)
        self.out_selector = BladeSelector(algebra, channels)
        # Grade-0 scalars [channels] -> logits [vocab_size]
        self.head = nn.Linear(channels, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_hidden: bool = False,
    ):
        """Forward pass: token ids -> logits.

        Args:
            token_ids (torch.Tensor): ``[B, L]`` integer token indices.
            return_hidden (bool): If True, also return the post-norm hidden
                multivectors ``[B, L, C, D]``.

        Returns:
            logits (torch.Tensor): ``[B*L, vocab_size]``.
            hidden (torch.Tensor, optional): ``[B, L, C, D]`` if return_hidden.
        """
        # Embed and add positional encoding
        x = self.embed(token_ids)  # [B, L, C, D]
        x = self.pe(x)             # [B, L, C, D]

        # Stack of transformer blocks
        for block in self.blocks:
            x = block(x)           # [B, L, C, D]

        # Final norm + blade gating (on flattened sequence)
        B, L, C, D = x.shape
        x_flat = x.reshape(B * L, C, D)
        x_flat = self.out_norm(x_flat)      # [B*L, C, D]
        x_flat = self.out_selector(x_flat)  # [B*L, C, D]

        # Extract grade-0 (scalar) components: index 0 in blade basis
        grade0 = x_flat[:, :, 0]   # [B*L, C]
        logits = self.head(grade0)  # [B*L, vocab_size]

        if return_hidden:
            return logits, x_flat.reshape(B, L, C, D)
        return logits
