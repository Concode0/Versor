# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Lensing Geometric Blade Network for gravitational lensing analysis.

U-Net-like encoder-decoder with geometric bottleneck in Spacetime Algebra Cl(1,3).
Features: Rotary2DBivectorPE, GeometricPatchMerging/Expanding, BladeSelector at output only.
Multi-head output: source reconstruction, convergence map, shear, classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from layers.base import CliffordModule
from layers.linear import CliffordLinear
from layers.normalization import CliffordLayerNorm
from layers.projection import BladeSelector
from layers.rotor import RotorLayer
from layers.multi_rotor import MultiRotorLayer
from layers.attention import GeometricProductAttention
from functional.activation import GeometricGELU


class Rotary2DBivectorPE(CliffordModule):
    """2D rotary positional encoding via bivector rotors on a spatial grid.

    Extends RotaryBivectorPE to 2D: each (row, col) position gets a rotor
    R(h,w) = exp(-B_row(h)/2) * exp(-B_col(w)/2). Row and col use disjoint
    bivector subsets so the rotations compose cleanly.

    For a grid of nH x nW patches, the position-dependent rotor rotates
    each multivector token, injecting spatial structure without breaking
    geometric equivariance.
    """

    def __init__(self, algebra, channels, max_h, max_w, learnable=True):
        """Initialize 2D rotary bivector PE.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Unused (API consistency).
            max_h (int): Maximum grid height (number of patch rows).
            max_w (int): Maximum grid width (number of patch columns).
            learnable (bool): If True, bivector weights are nn.Parameters.
        """
        super().__init__(algebra)
        self.max_h = max_h
        self.max_w = max_w

        bv_indices = [i for i in range(algebra.dim) if bin(i).count('1') == 2]
        self.register_buffer('bivector_indices',
                             torch.tensor(bv_indices, dtype=torch.long))
        num_bv = len(bv_indices)

        # Split bivectors: first half for rows, second half for cols
        self.n_bv_row = num_bv // 2
        self.n_bv_col = num_bv - self.n_bv_row

        init_row = self._sinusoidal_init(max_h, self.n_bv_row)
        init_col = self._sinusoidal_init(max_w, self.n_bv_col)

        if learnable:
            self.row_weights = nn.Parameter(init_row)
            self.col_weights = nn.Parameter(init_col)
        else:
            self.register_buffer('row_weights', init_row)
            self.register_buffer('col_weights', init_col)

    def _sinusoidal_init(self, L, num_bv):
        """Initialize sinusoidal weights.

        Args:
            L (int): Number of positions.
            num_bv (int): Number of bivectors.

        Returns:
            torch.Tensor: Sinusoidal weights B[p, k] = p * 10000^(-2k/num_bv) * 0.01.
        """
        positions = torch.arange(L, dtype=torch.float32).unsqueeze(1)
        freqs = torch.pow(
            10000.0,
            -2.0 * torch.arange(num_bv, dtype=torch.float32) / max(num_bv, 1)
        ).unsqueeze(0)
        return positions * freqs * 0.01

    def _build_rotor(self, weights, bv_start, bv_end, L):
        """Build rotors from bivector weights for L positions.

        Args:
            weights (torch.Tensor): [L, n_bv] bivector coefficients.
            bv_start (int): Start index into self.bivector_indices.
            bv_end (int): End index.
            L (int): Number of positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - R: [L, D] rotor multivectors.
                - R_rev: [L, D] reversed rotors.
        """
        D = self.algebra.dim
        device = weights.device
        dtype = weights.dtype

        B_pos = torch.zeros(L, D, device=device, dtype=dtype)
        bv_idx = self.bivector_indices[bv_start:bv_end]  # [n_bv]
        B_pos.scatter_(1, bv_idx.unsqueeze(0).expand(L, -1), weights[:L])

        R = self.algebra.exp(-0.5 * B_pos)
        R_rev = self.algebra.reverse(R)
        return R, R_rev

    def forward(self, x, nH, nW):
        """Apply 2D rotary PE to a grid of multivector tokens.

        Args:
            x (torch.Tensor): [B, nH*nW, C, D] multivector tokens on a spatial grid.
            nH (int): Grid height.
            nW (int): Grid width.

        Returns:
            torch.Tensor: [B, nH*nW, C, D] with position-dependent rotor applied.
        """
        B, N, C, D = x.shape
        device = x.device
        dtype = x.dtype

        # Row rotors: [nH, D]
        R_row, R_row_rev = self._build_rotor(
            self.row_weights[:nH], 0, self.n_bv_row, nH
        )
        # Col rotors: [nW, D]
        R_col, R_col_rev = self._build_rotor(
            self.col_weights[:nW], self.n_bv_row, self.n_bv_row + self.n_bv_col, nW
        )

        # Compose: R(h,w) = R_row(h) * R_col(w) for each grid position
        # Expand to [nH, nW, D] via geometric product
        R_row_exp = R_row.unsqueeze(1).expand(nH, nW, D).reshape(nH * nW, D)
        R_col_exp = R_col.unsqueeze(0).expand(nH, nW, D).reshape(nH * nW, D)
        R = self.algebra.geometric_product(R_row_exp, R_col_exp)  # [N, D]

        R_col_rev_exp = R_col_rev.unsqueeze(0).expand(nH, nW, D).reshape(nH * nW, D)
        R_row_rev_exp = R_row_rev.unsqueeze(1).expand(nH, nW, D).reshape(nH * nW, D)
        # reverse(R_row * R_col) = reverse(R_col) * reverse(R_row)
        R_rev = self.algebra.geometric_product(R_col_rev_exp, R_row_rev_exp)  # [N, D]

        # Apply sandwich: x' = R x R~  (broadcast over B and C)
        # x: [B, N, C, D] -> [B*N*C, D]
        x_flat = x.reshape(B * N * C, D)
        R_flat = R.unsqueeze(0).expand(B, N, D).unsqueeze(2).expand(B, N, C, D)
        R_flat = R_flat.reshape(B * N * C, D)
        R_rev_flat = R_rev.unsqueeze(0).expand(B, N, D).unsqueeze(2).expand(B, N, C, D)
        R_rev_flat = R_rev_flat.reshape(B * N * C, D)

        Rx = self.algebra.geometric_product(R_flat, x_flat)
        RxRr = self.algebra.geometric_product(Rx, R_rev_flat)

        return RxRr.reshape(B, N, C, D)


class GeometricPatchMerging(CliffordModule):
    """Downsample by merging 2x2 spatial patches, doubling channels.

    Like Swin Transformer's PatchMerging but operating on multivectors.
    Merges 2x2 adjacent patches via concatenation along channel dim,
    then projects back with CliffordLinear.

    [B, nH*nW, C, D] -> [B, (nH/2)*(nW/2), 2*C, D]
    """

    def __init__(self, algebra, channels, use_rotor_backend=False):
        """Initialize Geometric Patch Merging.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Input channel count.
            use_rotor_backend (bool): If True, use Rotor-based CliffordLinear.
        """
        super().__init__(algebra)
        self.channels = channels
        backend = 'rotor' if use_rotor_backend else 'traditional'
        # 4*C -> 2*C via CliffordLinear
        self.reduction = CliffordLinear(
            algebra, 4 * channels, 2 * channels, backend=backend
        )
        self.norm = CliffordLayerNorm(algebra, 4 * channels)

    def forward(self, x, nH, nW):
        """Merge 2x2 patches.

        Args:
            x (torch.Tensor): [B, nH*nW, C, D] multivector tokens.
            nH (int): Spatial grid height (must be even).
            nW (int): Spatial grid width (must be even).

        Returns:
            torch.Tensor: [B, (nH//2)*(nW//2), 2*C, D] merged tokens.
        """
        B, N, C, D = x.shape
        x = x.reshape(B, nH, nW, C, D)

        # Extract 2x2 quadrants
        x0 = x[:, 0::2, 0::2, :, :]  # [B, nH/2, nW/2, C, D] top-left
        x1 = x[:, 1::2, 0::2, :, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :, :]  # top-right
        x3 = x[:, 1::2, 1::2, :, :]  # bottom-right

        # Concatenate along channel dim: [B, nH/2, nW/2, 4*C, D]
        merged = torch.cat([x0, x1, x2, x3], dim=3)
        nH2, nW2 = nH // 2, nW // 2
        merged = merged.reshape(B * nH2 * nW2, 4 * C, D)

        # Norm + reduce: 4*C -> 2*C
        merged = self.norm(merged)
        merged = self.reduction(merged)
        return merged.reshape(B, nH2 * nW2, 2 * C, D)


class GeometricPatchExpanding(CliffordModule):
    """Upsample by expanding patches, halving channels.

    Inverse of GeometricPatchMerging. Projects C -> 2*C channels via
    CliffordLinear, then reshapes to distribute over 2x2 spatial positions.

    [B, nH*nW, C, D] -> [B, (2*nH)*(2*nW), C//2, D]
    """

    def __init__(self, algebra, channels, use_rotor_backend=False):
        """Initialize Geometric Patch Expanding.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Input channel count.
            use_rotor_backend (bool): If True, use Rotor-based CliffordLinear.
        """
        super().__init__(algebra)
        self.channels = channels
        backend = 'rotor' if use_rotor_backend else 'traditional'
        # C -> 2*C (will be split into 4 groups of C//2)
        self.expansion = CliffordLinear(
            algebra, channels, 2 * channels, backend=backend
        )
        self.norm = CliffordLayerNorm(algebra, channels // 2)

    def forward(self, x, nH, nW):
        """Expand patches to 2x spatial resolution.

        Args:
            x (torch.Tensor): [B, nH*nW, C, D] multivector tokens.
            nH (int): Current grid height.
            nW (int): Current grid width.

        Returns:
            torch.Tensor: [B, (2*nH)*(2*nW), C//2, D] expanded tokens.
        """
        B, N, C, D = x.shape

        # Expand channels: C -> 2*C
        x_flat = x.reshape(B * N, C, D)
        x_exp = self.expansion(x_flat)  # [B*N, 2*C, D]
        x_exp = x_exp.reshape(B, nH, nW, 2 * C, D)

        # Redistribute: split 2*C into 4 groups of C//2, place in 2x2 grid
        c_out = C // 2
        x_exp = x_exp.reshape(B, nH, nW, 4, c_out, D)

        nH2, nW2 = 2 * nH, 2 * nW
        out = torch.zeros(B, nH2, nW2, c_out, D, device=x.device, dtype=x.dtype)
        out[:, 0::2, 0::2, :, :] = x_exp[:, :, :, 0, :, :]
        out[:, 1::2, 0::2, :, :] = x_exp[:, :, :, 1, :, :]
        out[:, 0::2, 1::2, :, :] = x_exp[:, :, :, 2, :, :]
        out[:, 1::2, 1::2, :, :] = x_exp[:, :, :, 3, :, :]

        out = out.reshape(B * nH2 * nW2, c_out, D)
        out = self.norm(out)
        return out.reshape(B, nH2 * nW2, c_out, D)


class ImageToMultivector(nn.Module):
    """Lifts 2D image patches into multivector space (grade-0 and grade-1 only).

    Splits the image into non-overlapping patches and projects each patch
    into grade-0 (scalar) and grade-1 (vector) components of multivector
    channels. Higher grades (bivector, trivector, pseudoscalar) are left as
    zero - the network learns to populate them via rotor sandwich products.
    """

    def __init__(self, algebra, in_channels, channels, patch_size, image_size):
        """Initialize Image-to-Multivector projection.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            in_channels (int): Input image channels.
            channels (int): Output multivector channels.
            patch_size (int): Spatial patch size.
            image_size (int): Total image resolution.
        """
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels
        self.algebra_dim = algebra.dim
        self.n_patches_h = image_size // patch_size
        self.n_patches_w = image_size // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        # Grade indices for Cl(1,3): grade-0 = [0], grade-1 = [1,2,4,8]
        g0_idx = [i for i in range(algebra.dim) if bin(i).count('1') == 0]
        g1_idx = [i for i in range(algebra.dim) if bin(i).count('1') == 1]
        self.register_buffer('g0_indices', torch.tensor(g0_idx, dtype=torch.long))
        self.register_buffer('g1_indices', torch.tensor(g1_idx, dtype=torch.long))
        n_low = len(g0_idx) + len(g1_idx)  # 1 + 4 = 5 for Cl(1,3)

        patch_dim = in_channels * patch_size * patch_size
        # Project to grade-0 + grade-1 components only
        self.proj = nn.Linear(patch_dim, channels * n_low)
        self.n_g0 = len(g0_idx)
        self.n_g1 = len(g1_idx)
        self.n_low = n_low

    def forward(self, x):
        """Lifts image to multivectors.

        Args:
            x (torch.Tensor): [B, C_in, H, W] image.

        Returns:
            torch.Tensor: [B, nH*nW, channels, algebra_dim] multivector tokens.
        """
        B, C, H, W = x.shape
        ps = self.patch_size

        x = x.unfold(2, ps, ps).unfold(3, ps, ps)  # [B, C, nH, nW, ps, ps]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.reshape(B, self.n_patches, -1)  # [B, N, C*ps*ps]

        low = self.proj(x)  # [B, N, channels * n_low]
        low = low.reshape(B, self.n_patches, self.channels, self.n_low)

        # Place into full multivector: grade-0 and grade-1 slots only
        out = torch.zeros(
            B, self.n_patches, self.channels, self.algebra_dim,
            device=x.device, dtype=x.dtype,
        )
        out[:, :, :, self.g0_indices] = low[:, :, :, :self.n_g0]
        out[:, :, :, self.g1_indices] = low[:, :, :, self.n_g0:]
        return out


class MultivectorToImage(nn.Module):
    """Projects multivectors back to image pixel space using all grades.

    Generic head for source reconstruction (uses full multivector).
    """

    def __init__(self, algebra, channels, out_channels, patch_size, image_size):
        """Initialize Multivector-to-Image projection.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Input multivector channels.
            out_channels (int): Output image channels.
            patch_size (int): Spatial patch size.
            image_size (int): Total image resolution.
        """
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches_h = image_size // patch_size
        self.n_patches_w = image_size // patch_size

        patch_dim = out_channels * patch_size * patch_size
        self.proj = nn.Linear(channels * algebra.dim, patch_dim)
        self.out_channels = out_channels

    def forward(self, x):
        """Projects multivectors to pixels.

        Args:
            x (torch.Tensor): [B, N, C, D] multivector tokens.

        Returns:
            torch.Tensor: [B, C_out, H, W] reconstructed image.
        """
        B, N, C, D = x.shape
        ps = self.patch_size
        nH = self.n_patches_h
        nW = self.n_patches_w

        x = x.reshape(B, N, C * D)
        x = self.proj(x)  # [B, N, out_c * ps * ps]
        x = x.reshape(B, nH, nW, self.out_channels, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, self.out_channels, nH * ps, nW * ps)
        return x


class GradeSliceToImage(nn.Module):
    """Projects specific grade components of multivectors to image pixel space.

    Slices the requested grade from each channel before projecting to pixels.
    Used for physically meaningful outputs: grade-0 for convergence kappa,
    grade-2 for shear gamma.
    """

    def __init__(self, algebra, channels, out_channels, patch_size, image_size, grade):
        """Initialize Grade-Slice-to-Image projection.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Input multivector channels.
            out_channels (int): Output image channels.
            patch_size (int): Spatial patch size.
            image_size (int): Total image resolution.
            grade (int): Grade to extract (e.g., 0 for scalar, 2 for bivector).
        """
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches_h = image_size // patch_size
        self.n_patches_w = image_size // patch_size
        self.out_channels = out_channels

        # Get indices for the requested grade
        grade_idx = [i for i in range(algebra.dim) if bin(i).count('1') == grade]
        self.register_buffer('grade_indices', torch.tensor(grade_idx, dtype=torch.long))
        n_grade = len(grade_idx)  # grade-0: 1, grade-2: 6 for Cl(1,3)

        patch_dim = out_channels * patch_size * patch_size
        self.proj = nn.Linear(channels * n_grade, patch_dim)

    def forward(self, x):
        """Projects grade-sliced tokens to pixels.

        Args:
            x (torch.Tensor): [B, N, C, D] multivector tokens.

        Returns:
            torch.Tensor: [B, C_out, H, W] projected image.
        """
        B, N, C, D = x.shape
        ps = self.patch_size
        nH = self.n_patches_h
        nW = self.n_patches_w

        # Slice only the requested grade components
        x = x[:, :, :, self.grade_indices]  # [B, N, C, n_grade]
        x = x.reshape(B, N, -1)  # [B, N, C * n_grade]
        x = self.proj(x)  # [B, N, out_c * ps * ps]
        x = x.reshape(B, nH, nW, self.out_channels, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, self.out_channels, nH * ps, nW * ps)
        return x


class _EncoderBlock(nn.Module):
    """Encoder block: Norm -> GELU -> MultiRotor -> Linear + residual."""

    def __init__(self, algebra, channels, num_rotors, use_decomposition, decomp_k,
                 use_rotor_backend):
        """Initialize Encoder Block.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Channel count.
            num_rotors (int): Number of rotors in MultiRotorLayer.
            use_decomposition (bool): If True, use bivector decomposition.
            decomp_k (int): Rank for bivector decomposition.
            use_rotor_backend (bool): If True, use Rotor-based CliffordLinear.
        """
        super().__init__()
        self.norm = CliffordLayerNorm(algebra, channels)
        self.activation = GeometricGELU(algebra, channels)
        self.multi_rotor = MultiRotorLayer(
            algebra, channels, num_rotors=num_rotors,
            use_decomposition=use_decomposition, decomp_k=decomp_k,
        )
        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.linear = CliffordLinear(algebra, channels, channels, backend=backend)

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x (torch.Tensor): [*, C, D] multivector tokens.

        Returns:
            torch.Tensor: [*, C, D] processed tokens.
        """
        out = self.norm(x)
        out = self.activation(out)
        out = self.multi_rotor(out)
        out = self.linear(out)
        return out + x  # residual


class _DecoderBlock(nn.Module):
    """Decoder block: skip concat -> project -> Norm -> GELU -> Rotor -> Linear."""

    def __init__(self, algebra, channels, skip_channels, use_decomposition, decomp_k,
                 use_rotor_backend):
        """Initialize Decoder Block.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            channels (int): Current channel count.
            skip_channels (int): Channels from skip connection (0 if none).
            use_decomposition (bool): If True, use bivector decomposition.
            decomp_k (int): Rank for bivector decomposition.
            use_rotor_backend (bool): If True, use Rotor-based CliffordLinear.
        """
        super().__init__()
        backend = 'rotor' if use_rotor_backend else 'traditional'
        self.skip_proj = CliffordLinear(
            algebra, channels + skip_channels, channels, backend=backend
        )
        self.norm = CliffordLayerNorm(algebra, channels)
        self.activation = GeometricGELU(algebra, channels)
        self.rotor = RotorLayer(
            algebra, channels, use_decomposition=use_decomposition, decomp_k=decomp_k,
        )
        self.linear = CliffordLinear(algebra, channels, channels, backend=backend)

    def forward(self, x, skip):
        """Forward pass with skip connection.

        Args:
            x (torch.Tensor): [B, N, C, D] multivector tokens.
            skip (torch.Tensor): [B, N, C_skip, D] tokens from encoder.

        Returns:
            torch.Tensor: [B, N, C, D] processed tokens.
        """
        B, N, C, D = x.shape
        combined = torch.cat([x, skip], dim=2)  # [B, N, C+C_skip, D]
        out = self.skip_proj(combined.reshape(B * N, -1, D))
        out = self.norm(out)
        out = self.activation(out)
        out = self.rotor(out)
        out = self.linear(out)
        return out.reshape(B, N, C, D)


class LensingGBN(CliffordModule):
    """Gravitational Lensing Geometric Blade Network.

    U-Net-like architecture with:
    - Rotary2DBivectorPE for spatial position encoding
    - GeometricPatchMerging for downsampling (2x per stage)
    - GeometricPatchExpanding for upsampling (2x per stage)
    - GeometricProductAttention bottleneck
    - BladeSelector at final output only

    Multi-head output:
        source: [B, 1, H, W]  - reconstructed source galaxy
        kappa:  [B, 1, H, W]  - convergence map
        shear:  [B, 2, H, W]  - shear gamma1, gamma2
        logits: [B, 3]        - dark matter substructure class
    """

    def __init__(
        self,
        algebra,
        image_size=64,
        patch_size=8,
        in_channels=1,
        hidden_channels=32,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_rotors=8,
        num_attention_heads=4,
        use_decomposition=True,
        decomp_k=10,
        use_rotor_backend=True,
        mode="full",
    ):
        """Initialize Lensing GBN.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            image_size (int): Input image resolution.
            patch_size (int): Initial patch size.
            in_channels (int): Input image channels.
            hidden_channels (int): Base channel count.
            num_encoder_layers (int): Total encoder layers.
            num_decoder_layers (int): Total decoder layers.
            num_rotors (int): Rotors per MultiRotorLayer.
            num_attention_heads (int): Heads for bottleneck attention.
            use_decomposition (bool): If True, use bivector decomposition.
            decomp_k (int): Rank for bivector decomposition.
            use_rotor_backend (bool): If True, use Rotor-based CliffordLinear.
            mode (str): Output mode ("full", "reconstruct", "convergence", "classify").
        """
        super().__init__(algebra)
        self.image_size = image_size
        self.patch_size = patch_size
        self.mode = mode
        self.hidden_channels = hidden_channels

        nH0 = image_size // patch_size
        nW0 = nH0  # assume square

        # --- Patch embedding ---
        self.img_to_mv = ImageToMultivector(
            algebra, in_channels, hidden_channels, patch_size, image_size
        )

        # --- Rotary 2D PE ---
        self.pos_enc = Rotary2DBivectorPE(
            algebra, hidden_channels, max_h=nH0, max_w=nW0, learnable=True
        )

        # --- Time embedding (temporal lensing inversion) ---
        # Inject time into grade-1 temporal direction e_0 (index 1 in Cl(1,3))
        # instead of spreading across all 16 components.
        self.time_embed = nn.Linear(1, hidden_channels)
        # e_0 is the first basis vector: binary index 1 (0b0001)
        self.e0_index = 1  # temporal direction in STA Cl(1,3)

        # --- Encoder: stages with downsampling ---
        # Each stage has `blocks_per_stage` encoder blocks + patch merging
        # Channels double at each downsampling: C -> 2C -> 4C ...
        # We group encoder layers into stages:
        #   num_encoder_layers=4 => 2 stages of 2 blocks each (with 1 downsample)
        self.num_stages = max(1, num_encoder_layers // 2)
        blocks_per_stage = max(1, num_encoder_layers // self.num_stages)

        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.stage_channels = []  # track channels at each stage for decoder

        ch = hidden_channels
        cur_h, cur_w = nH0, nW0

        for s in range(self.num_stages):
            blocks = nn.ModuleList([
                _EncoderBlock(algebra, ch, num_rotors, use_decomposition,
                              decomp_k, use_rotor_backend)
                for _ in range(blocks_per_stage)
            ])
            self.encoder_stages.append(blocks)
            self.stage_channels.append(ch)

            # Downsample if not last stage and spatial dim allows
            if s < self.num_stages - 1 and cur_h >= 2 and cur_w >= 2:
                self.downsamplers.append(
                    GeometricPatchMerging(algebra, ch, use_rotor_backend)
                )
                ch = ch * 2
                cur_h, cur_w = cur_h // 2, cur_w // 2
            else:
                self.downsamplers.append(None)

        self.bottleneck_channels = ch
        self.bottleneck_h = cur_h
        self.bottleneck_w = cur_w

        # --- Bottleneck attention ---
        self.bottleneck_attn = GeometricProductAttention(
            algebra, ch, num_heads=min(num_attention_heads, ch),
            causal=False, bivector_weight=0.5, dropout=0.1,
        )
        self.bottleneck_norm = CliffordLayerNorm(algebra, ch)

        # --- Decoder: stages with upsampling + skip connections ---
        self.decoder_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for s in range(self.num_stages - 1, -1, -1):
            skip_ch = self.stage_channels[s]

            # Upsample if not at full resolution
            if s < self.num_stages - 1:
                self.upsamplers.append(
                    GeometricPatchExpanding(algebra, ch, use_rotor_backend)
                )
                ch = ch // 2
            else:
                self.upsamplers.append(None)

            blocks = nn.ModuleList([
                _DecoderBlock(algebra, ch, skip_ch if i == 0 else 0,
                              use_decomposition, decomp_k, use_rotor_backend)
                if i == 0 else
                _EncoderBlock(algebra, ch, num_rotors, use_decomposition,
                              decomp_k, use_rotor_backend)
                for i in range(blocks_per_stage)
            ])
            self.decoder_stages.append(blocks)

        self.final_ch = ch

        # --- Final BladeSelector (only place it appears) ---
        self.output_blade = BladeSelector(algebra, self.final_ch)
        self.output_norm = CliffordLayerNorm(algebra, self.final_ch)

        # --- Output heads ---
        self.source_head = MultivectorToImage(
            algebra, self.final_ch, 1, patch_size, image_size
        )
        self.kappa_head = GradeSliceToImage(
            algebra, self.final_ch, 1, patch_size, image_size, grade=0
        )
        self.shear_head = GradeSliceToImage(
            algebra, self.final_ch, 2, patch_size, image_size, grade=2
        )
        self.cls_norm = CliffordLayerNorm(algebra, self.final_ch)
        self.cls_proj = nn.Linear(self.final_ch * algebra.dim, 3)

    def _apply_blocks(self, blocks, x, nH, nW):
        """Apply a list of encoder/decoder blocks to x.

        Flattens [B, N, C, D] -> [B*N, C, D] for Clifford layers.
        """
        B, N, C, D = x.shape
        for block in blocks:
            if isinstance(block, _DecoderBlock):
                # DecoderBlocks handled separately (need skip)
                raise ValueError("Use _apply_decoder_blocks for DecoderBlocks")
            x_flat = x.reshape(B * N, C, D)
            x_flat = block(x_flat)
            x = x_flat.reshape(B, N, C, D)
        return x

    def forward(self, lensed, t=None):
        """Forward pass.

        Args:
            lensed (torch.Tensor): [B, 1, H, W] lensed image.
            t (torch.Tensor, optional): [B, 1] time parameter for temporal inversion.

        Returns:
            Dict[str, torch.Tensor]: Outputs depending on mode (source, kappa, shear, logits).
        """
        B = lensed.shape[0]
        nH0 = self.image_size // self.patch_size
        nW0 = nH0

        # --- Embed ---
        x = self.img_to_mv(lensed)  # [B, N, C, D]

        # --- Rotary 2D PE ---
        x = self.pos_enc(x, nH0, nW0)

        # --- Time conditioning: inject into e_0 (grade-1 temporal) ---
        if t is not None:
            t_coeff = self.time_embed(t)  # [B, C] per-channel scalar
            t_mv = torch.zeros(
                B, 1, self.hidden_channels, self.algebra.dim,
                device=x.device, dtype=x.dtype,
            )
            t_mv[:, 0, :, self.e0_index] = t_coeff  # inject into e_0 only
            x = x + t_mv

        # --- Encoder ---
        skips = []
        nH, nW = nH0, nW0

        for s, (enc_blocks, downsampler) in enumerate(
            zip(self.encoder_stages, self.downsamplers)
        ):
            x = self._apply_blocks(enc_blocks, x, nH, nW)
            skips.append((x, nH, nW))

            if downsampler is not None:
                x = downsampler(x, nH, nW)
                nH, nW = nH // 2, nW // 2

        # --- Bottleneck ---
        BN_C, BN_D = x.shape[2], x.shape[3]
        x_normed = self.bottleneck_norm(x.reshape(-1, BN_C, BN_D))
        x_normed = x_normed.reshape(B, -1, BN_C, BN_D)

        x_bot = x + self.bottleneck_attn(x_normed) 
        x = x_bot

        # --- Decoder ---
        for s, (dec_blocks, upsampler) in enumerate(
            zip(self.decoder_stages, self.upsamplers)
        ):
            if upsampler is not None:
                x = upsampler(x, nH, nW)
                nH, nW = nH * 2, nW * 2

            # Pop matching skip
            skip_x, skip_nH, skip_nW = skips[-(s + 1)]

            # First block is DecoderBlock (handles skip connection)
            first_block = dec_blocks[0]
            x = first_block(x, skip_x)

            # Remaining blocks are regular encoder-style blocks
            if len(dec_blocks) > 1:
                x = self._apply_blocks(dec_blocks[1:], x, nH, nW)

        # --- Final BladeSelector + Norm (only grade filtering) ---
        x_flat = x.reshape(B * nH * nW, self.final_ch, self.algebra.dim)
        x_flat = self.output_norm(x_flat)
        x_flat = self.output_blade(x_flat)
        x = x_flat.reshape(B, nH * nW, self.final_ch, self.algebra.dim)

        # --- Output heads ---
        outputs = {}

        if self.mode in ("full", "reconstruct"):
            outputs['source'] = self.source_head(x)

        if self.mode in ("full", "convergence"):
            outputs['kappa'] = self.kappa_head(x)

        if self.mode == "full":
            outputs['shear'] = self.shear_head(x)

        if self.mode in ("full", "classify"):
            x_pooled = x.amax(dim=1)  # [B, C, D]
            x_cls = self.cls_norm(x_pooled)
            x_cls = x_cls.reshape(B, -1)
            outputs['logits'] = self.cls_proj(x_cls)

        return outputs

    def total_sparsity_loss(self):
        """Sum of L1 sparsity losses over all MultiRotorLayer instances.

        Returns:
            torch.Tensor: Total sparsity loss scalar.
        """
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, MultiRotorLayer):
                total = total + module.sparsity_loss()
        return total