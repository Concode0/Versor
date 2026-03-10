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
from ..primitives.base import CliffordModule
from ..primitives.normalization import CliffordLayerNorm
from .attention import GeometricProductAttention
from ..adapters.mother import EntropyGatedAttention
from .multi_rotor_ffn import MultiRotorFFN

class GeometricTransformerBlock(CliffordModule):
    """Modular Geometric Transformer block.
    
    Architecture:
    1. Pre-norm
    2. Geometric Attention (Standard or Entropy-Gated)
    3. Residual connection
    4. Pre-norm
    5. Multi-Rotor FFN
    6. Residual connection
    """
    def __init__(
        self, 
        algebra: CliffordAlgebra, 
        channels: int, 
        num_heads: int = 4, 
        num_rotors: int = 8, 
        dropout: float = 0.1, 
        use_entropy_gating: bool = False, 
        eta: float = 1.5, 
        H_base: float = 0.5
    ):
        """Initializes the Geometric Transformer Block.
        
        Args:
            algebra: Clifford algebra instance.
            channels: Total multivector channels.
            num_heads: Number of attention heads.
            num_rotors: Number of rotors in the FFN.
            dropout: Dropout rate.
            use_entropy_gating: If True, uses EntropyGatedAttention.
            eta: Gating multiplier for entropy attention.
            H_base: Base entropy threshold.
        """
        super().__init__(algebra)
        self.use_entropy_gating = use_entropy_gating
        self.norm1 = CliffordLayerNorm(algebra, channels)
        
        if use_entropy_gating:
            self.attn = EntropyGatedAttention(algebra, channels, num_heads, eta=eta, H_base=H_base)
        else:
            self.attn = GeometricProductAttention(algebra, channels, num_heads, causal=False, dropout=dropout)
            
        self.norm2 = CliffordLayerNorm(algebra, channels)
        
        # Check MultiRotorFFN class name in multi_rotor_ffn.py
        from .multi_rotor_ffn import MultiRotorFFN
        self.ffn = MultiRotorFFN(algebra, channels, num_rotors=num_rotors)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None,
                return_state: bool = False) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input multivectors [B, L, C, D].
            key_padding_mask: Optional [B, L] bool mask where True = padded.
            return_state: If True, returns intermediate entropy/gating states.

        Returns:
            Processed multivectors [B, L, C, D] (and optionally intermediate states).
        """
        B, L, C, D = x.shape

        # 1. Attention path
        res = x
        x_n = self.norm1(x.reshape(B*L, C, D)).reshape(B, L, C, D)

        if self.use_entropy_gating and return_state:
            attn_out, H, lambda_dyn = self.attn(x_n, key_padding_mask=key_padding_mask, return_gating=True)
        else:
            attn_out = self.attn(x_n, key_padding_mask=key_padding_mask)
            H, lambda_dyn = None, None
            
        x = res + attn_out
        
        # 2. FFN path
        res = x
        x_n = self.norm2(x.reshape(B*L, C, D)).reshape(B, L, C, D)
        f_out = self.ffn(x_n.reshape(B*L, C, D)).reshape(B, L, C, D)
        x = res + f_out
        
        if return_state:
            return x, H, lambda_dyn
        return x
