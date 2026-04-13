# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Symbolic Regression Geometric Blade Network.

Multi-Rotor system for symbolic regression.
Sparse rotor superposition mirrors the parsimony of symbolic expressions.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from layers import CliffordModule
from layers import CliffordLinear
from layers import CliffordLayerNorm
from layers import BladeSelector
from layers import RotorLayer
from functional.activation import GeometricGELU, GeometricSquare


def _blade_name(idx: int, n: int) -> str:
    """Return a human-readable blade name from a basis-blade index.

    Examples:
        (n=4):
        idx=0  -> '1'      (scalar)
        idx=1  -> 'e1'     (grade-1)
        idx=3  -> 'e12'    (grade-2, binary 0011 -> bits 0 and 1)
        idx=15 -> 'e1234'  (pseudoscalar)

    Args:
        idx (int): Basis-blade index.
        n (int): Number of basis vectors.

    Returns:
        str: Human-readable blade name.
    """
    if idx == 0:
        return "1"
    bits = [i + 1 for i in range(n) if idx & (1 << i)]
    return "e" + "".join(str(b) for b in bits)


def blade_names_for_algebra(algebra) -> list:
    """Return a list of blade names for every basis element of *algebra*.

    Args:
        algebra (CliffordAlgebra): CliffordAlgebra instance.

    Returns:
        list: List of blade name strings.
    """
    return [_blade_name(i, algebra.n) for i in range(algebra.dim)]


def bivector_plane_names(algebra) -> list:
    """Return blade names for the grade-2 (bivector / rotation-plane) blades.

    Args:
        algebra (CliffordAlgebra): CliffordAlgebra instance.

    Returns:
        list: List of bivector blade name strings.
    """
    return [
        _blade_name(i, algebra.n)
        for i in range(algebra.dim)
        if bin(i).count("1") == 2
    ]


@dataclass
class FormulaResult:
    """Result of GA-native formula extraction from a trained SRGBN.

    Attributes:
        formula: Human-readable formula string, e.g. "y = 3.14*x1*x2 + 2*x1^2 - 1/2".
        coefficients: Mapping from monomial name to coefficient value.
        r2_vs_model: R-squared of the polynomial probe vs GBN predictions.
        max_degree: Maximum polynomial degree used in the probe.
        n_terms: Number of non-zero terms in the extracted formula.
        active_variables: Indices of variables with significant importance.
        grade_energy: Per-grade energy fractions from the hidden representation.
        var_names: Physical variable names when available.
    """
    formula: str
    coefficients: dict
    r2_vs_model: float
    max_degree: int
    n_terms: int
    active_variables: list
    grade_energy: list
    var_names: list = field(default_factory=list)


class SRMultiGradeEmbedding(CliffordModule):
    """Embeds scalar inputs into multiple Clifford algebra grades.

    Populates:
      - Grade 0: learnable scalar bias per channel.
      - Grade 1: linear projection of raw inputs.

    Attributes:
        in_features (int): Number of scalar inputs k.
        channels (int): Number of channels C.
        grade0_bias (nn.Parameter): [C] bias added to scalar component.
        grade1_proj (nn.Linear): k -> C*n_g1 projection.
    """

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_features: int,
        channels: int,
    ):
        """Initialize SR Multi-Grade Embedding.

        Args:
            algebra (CliffordAlgebra): CliffordAlgebra instance.
            in_features (int): Number of scalar inputs.
            channels (int): Number of hidden channels.
        """
        super().__init__(algebra)
        self.in_features = in_features
        self.channels = channels

        # Precompute grade-1 indices
        dim = algebra.dim
        g1_idx = [i for i in range(dim) if bin(i).count('1') == 1]
        self.n_g1 = len(g1_idx)

        self.register_buffer('g1_idx', torch.tensor(g1_idx, dtype=torch.long))

        # Grade-0: scalar bias per channel
        self.grade0_bias = nn.Parameter(torch.zeros(channels))

        # Grade-1: project k inputs -> C * n_g1
        self.grade1_proj = nn.Linear(in_features, channels * self.n_g1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.grade1_proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scalar inputs into the multivector space.

        Args:
            x (torch.Tensor): [B, k] scalar inputs.

        Returns:
            torch.Tensor: [B, C, 2^p] multivectors.
        """
        B = x.size(0)
        out = torch.zeros(B, self.channels, self.algebra.dim,
                          device=x.device, dtype=x.dtype)

        # Grade-0 bias
        out[:, :, 0] = self.grade0_bias.unsqueeze(0).expand(B, -1)

        # Grade-1
        g1_feats = self.grade1_proj(x).reshape(B, self.channels, self.n_g1)
        g1_idx = self.g1_idx.view(1, 1, -1).expand(B, self.channels, -1)
        out.scatter_(2, g1_idx, g1_feats)

        return out


class _ResidualBlock(nn.Module):
    """One residual block: Norm -> Linear -> Activation -> Rotor -> BladeSelector -> skip."""

    def __init__(
        self,
        algebra: CliffordAlgebra,
        channels: int,
        use_skip: bool = True,
        activation_type: str = "gelu",
    ):
        super().__init__()
        # SR uses recover=False: pure direction normalization is symbolically
        # interpretable (constant scaling absorbed by downstream linear weights).
        # recover=True would inject log1p(||x||) into grade-0, a transcendental
        # function of the hidden state norm that cannot be traced symbolically.
        self.norm       = CliffordLayerNorm(algebra, channels, recover=False)
        self.linear     = CliffordLinear(algebra, channels, channels)
        if activation_type == "square":
            self.activation = GeometricSquare(algebra, channels)
        else:
            self.activation = GeometricGELU(algebra, channels)
        self.rotor      = RotorLayer(algebra, channels)
        self.blade = BladeSelector(algebra, channels)
        self.use_skip = use_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.linear(out)
        out = self.activation(out)
        out = self.rotor(out)
        out = self.blade(out)
        if self.use_skip:
            return out + x
        return out


class SRGBN(CliffordModule):
    """Geometric Blade Network for symbolic regression.

    Architecture:
        SRMultiGradeEmbedding -> N residual blocks -> output head -> scalar

    Each residual block: CliffordLayerNorm -> CliffordLinear -> GeometricGELU
                         -> RotorLayer -> BladeSelector -> +skip

    Output head: final norm -> BladeSelector -> CliffordLinear(C->1)
                 -> grade-0 scalar extraction

    The sparsity loss on rotor bivectors encourages parsimonious
    decompositions (few active rotation planes = simple symbolic structure).
    """

    @staticmethod
    def single_rotor(algebra, in_features, channels=4):
        """Minimal 1-block SRGBN with no skip connection.

        Factory for the iterative unbender pipeline where each stage
        uses a single rotor to discover one geometric transformation.
        Skip connections are disabled so the rotor must be engaged.

        Args:
            algebra: CliffordAlgebra instance.
            in_features: Number of scalar inputs.
            channels: Number of hidden channels.

        Returns:
            SRGBN with num_layers=1, no skip connections.
        """
        return SRGBN(algebra, in_features, channels=channels,
                     num_layers=1, use_skip=False)

    def svd_warmstart(self, Vt, algebra):
        """Initialize first rotor's bivector weights from SVD rotation.

        Decomposes Vt into approximate Givens rotations and maps each
        to a bivector plane angle. Only sets elliptic components;
        hyperbolic/parabolic start at zero.

        Args:
            Vt: np.ndarray [k, k] right-singular vectors (orthogonal).
            algebra: CliffordAlgebra instance.
        """
        if Vt is None:
            return

        n = algebra.n
        k = min(Vt.shape[0], Vt.shape[1], n)

        # Extract rotation angles from Vt via Givens-like decomposition
        # For each pair (i,j), estimate the rotation angle from Vt
        bv_mask = algebra.grade_masks[2]
        n_bv = bv_mask.sum().item()
        bv_indices = bv_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        # Map each bivector to its (i,j) pair
        bv_map = {}
        for bv_idx, blade_idx in enumerate(bv_indices):
            bits = [pos for pos in range(n) if (blade_idx >> pos) & 1]
            if len(bits) == 2:
                bv_map[(bits[0], bits[1])] = bv_idx

        # Build bivector weights from Vt rotation angles
        # Start from existing weights (don't overwrite random init with zeros)
        bv_weights = self.blocks[0].rotor.bivector_weights.detach().mean(0).clone()
        any_set = False

        for i in range(min(k, n)):
            for j in range(i + 1, min(k, n)):
                if (i, j) in bv_map:
                    # Rotation angle in (i,j) plane from Vt
                    if i < Vt.shape[0] and j < Vt.shape[1]:
                        angle = float(np.arctan2(Vt[i, j] - Vt[j, i],
                                                  Vt[i, i] + Vt[j, j]))
                        if abs(angle) > 1e-8:
                            bv_weights[bv_map[(i, j)]] = angle / 2.0
                            any_set = True

        # Only overwrite if we found meaningful angles
        if any_set:
            with torch.no_grad():
                self.blocks[0].rotor.bivector_weights.copy_(
                    bv_weights.unsqueeze(0).expand_as(
                        self.blocks[0].rotor.bivector_weights
                    )
                )

    @staticmethod
    def auto_config(n_train: int, n_vars: int, dim: int) -> dict:
        """Select model capacity based on training set size.

        Args:
            n_train: Estimated number of training samples.
            n_vars: Number of input variables.
            dim: Algebra dimension (2^n).

        Returns:
            Dict with channels, num_layers.
        """
        if n_train < 16:
            return {"channels": 4, "num_layers": 2}
        elif n_train < 50:
            return {"channels": 8, "num_layers": 2}
        elif n_train < 150:
            return {"channels": 12, "num_layers": 3}
        else:
            return {"channels": 16, "num_layers": 3}

    def __init__(
        self,
        algebra: CliffordAlgebra,
        in_features: int,
        channels: int = 16,
        num_layers: int = 3,
        use_skip: bool = True,
    ):
        super().__init__(algebra)
        self.in_features = in_features
        self.channels = channels

        self.embedding = SRMultiGradeEmbedding(
            algebra, in_features, channels
        )

        self.blocks = nn.ModuleList([
            _ResidualBlock(algebra, channels,
                           use_skip=use_skip, activation_type="square")
            for _ in range(num_layers)
        ])

        # Output head also uses recover=False for symbolic traceability.
        self.output_norm   = CliffordLayerNorm(algebra, channels, recover=False)
        self.output_blade  = BladeSelector(algebra, channels)
        self.output_linear = CliffordLinear(algebra, channels, 1)

        # Readout: learned weighted sum across all blade components.
        # Grade-0 extraction alone cannot capture linear functions (y=c*x)
        # because inputs live in grade-1 and rotors preserve grade.
        # The readout projects all grades to scalar, letting grade-1
        # (rotor-transformed vectors) contribute to the output.
        self.readout = nn.Parameter(torch.zeros(algebra.dim))
        # Initialize: grade-0 weight = 1 (backward compatible), rest = 0
        with torch.no_grad():
            self.readout[0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): [B, k] normalised scalar inputs.

        Returns:
            torch.Tensor: [B, 1] predicted (normalised) scalar output.
        """
        out = self.embedding(x)      # [B, C, dim]

        for block in self.blocks:
            out = block(out)         # [B, C, dim]

        # Cache last hidden state for analysis / ortho monitoring.
        self._last_hidden = out.detach()   # [B, C, dim]
        self._hidden_for_curvature = out   # non-detached, for curvature loss

        out = self.output_norm(out)
        out = self.output_blade(out)
        out = self.output_linear(out)          # [B, 1, dim]
        # Readout: weighted sum across all blade components
        out = (out[:, 0, :] * self.readout).sum(-1, keepdim=True)  # [B, 1]

        return out

    def get_rotor_analysis(self) -> list:
        """Per-block rotor activity and dominant rotation planes."""
        plane_names = bivector_plane_names(self.algebra)
        results = []
        for i, block in enumerate(self.blocks):
            rotor = block.rotor
            bv = rotor.bivector_weights.detach().cpu()  # [C, n_bv]
            bv_mean = bv.abs().mean(0)  # [n_bv]
            dom = bv_mean.argmax().item()
            results.append({
                "layer": i,
                "bivectors": bv,  # [C, n_bv]
                "plane_names": plane_names,
                "dominant_plane": plane_names[dom],
            })
        return results

    def get_output_blade_weights(self, algebra) -> dict:
        """Map each basis-blade name -> its weight in the final CliffordLinear bias."""
        # output_linear bias: [out_channels=1, dim]
        w     = self.output_linear.bias.detach()[0]  # [dim]
        names = blade_names_for_algebra(algebra)
        return {name: w[i].item() for i, name in enumerate(names)}

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 sparsity losses over all RotorLayer instances."""
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, RotorLayer):
                total = total + module.sparsity_loss()
        return total

    @torch.no_grad()
    def structural_analysis(self, x_sample: torch.Tensor):
        """Stage 1: Determine polynomial degree and active variables from GA structure."""
        self.eval()
        self(x_sample)  # populates _last_hidden
        hidden = self._last_hidden  # [B, C, dim]

        # Grade energy spectrum
        n_grades = self.algebra.n + 1
        dim = self.algebra.dim
        grade_energies = []
        for g in range(n_grades):
            g_idx = [i for i in range(dim) if bin(i).count("1") == g]
            e = hidden[..., g_idx].pow(2).mean().item() if g_idx else 0.0
            grade_energies.append(e)
        total_e = sum(grade_energies) + 1e-12
        grade_fracs = [e / total_e for e in grade_energies]

        # max_degree from grade energy (grade-k energy > 1% -> include degree k)
        max_degree = 1
        for g in range(1, n_grades):
            if grade_fracs[g] > 0.01:
                max_degree = g
        max_degree = min(max_degree, 4)

        # Variable importance via gradient
        x_grad = x_sample.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            self(x_grad).sum().backward()
        imp = x_grad.grad.abs().mean(0)  # [k]
        total_imp = imp.sum() + 1e-12
        imp_frac = imp / total_imp
        active_vars = [i for i in range(imp.shape[0]) if imp_frac[i].item() > 0.01]
        if not active_vars:
            active_vars = list(range(imp.shape[0]))

        return max_degree, grade_fracs, active_vars
