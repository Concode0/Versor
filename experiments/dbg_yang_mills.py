# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""
==============================================================================
VERSOR EXPERIMENT: MATHEMATICAL DEBUGGER
==============================================================================

This script is designed to validate topological and algebraic phenomena 
rather than to achieve State-of-the-Art (SOTA) on traditional benchmarks. 
Our primary focus here is to explore pure geometric intuition within the 
Clifford Algebra framework.

Please kindly note that as an experimental module, formal mathematical proofs 
and exhaustive literature reviews may still be in progress. We warmly invite 
you to run the code, test your own hypotheses, and open a GitHub Issue if you 
discover any fascinating geometric behaviors or encounter structural limitations.

==============================================================================

Yang-Mills SU(2) Gauge Theory in CGA Cl(4,1) — Conformal Geometric Algebra.

Validates Geometric Algebra as a natural framework for non-abelian gauge theory.
SU(2) ≅ Spin(3) embedded in the conformal algebra Cl(4,1).

The gauge field A_μ(x) is an su(2)-valued 1-form: at each spacetime point x ∈ ℝ⁴,
there are 4 bivectors A₁, A₂, A₃, A₄ (one per spacetime direction).

In CGA Cl(4,1) with 32 basis elements (5 generators: e₁,e₂,e₃ spatial, e₄ positive,
e₅ negative):
  The su(2) gauge algebra lives in the spatial bivectors:
  Grade 2 [idx 3,5,6]:   su(2) gauge potential / field strength (e₁₂, e₁₃, e₂₃)
  Conformal bivectors:    e₁₄,e₂₄,e₃�� (translations), e₄₅ (dilations)

The CGA structure provides conformal versors (translations, dilations, special
conformal transformations) as rotors. The BPST instanton's rational decay envelope
1/(|x|²+ρ²) emerges implicitly from the conformal algebraic structure — no
external multiplication needed. Spatial coordinates are embedded into the CGA
null cone via P(x) = x + ½|x|²e∞ + e₀.

Gauge transformations are rotors R ∈ Spin(3) ⊂ Cl(4,1): A_μ → R A_μ R̃
The Hermitian inner product satisfies <RAR̃, RBR̃>_H = <A,B>_H (gauge invariance).

Intrinsic symmetry guarantees (no auxiliary loss needed):
  - Grade-2 su(2) purity: enforced by SU2BladeSelector
  - Gauge covariance: algebraic property of Hermitian inner product
  - Self-duality: the core physics constraint (F = *F determines instanton)
  - Conformal decay: handled by CGA rotor structure (no hardcoded envelope)

CGA integration:
  - Field strength via geometric calculus: F_μν = J[ν,:,μ] - J[μ,:,ν] + [A_μ, A_ν]
    where J is the Jacobian of su(2) components (12 autograd calls vs 96 naive)
  - Commutator via algebra.commutator() (single-pass precomputed antisymmetric signs)
  - Hodge dual via precomputed spacetime permutation map
  - Topological charge via Hermitian inner product: Q ∝ <F, *F>_H
  - Conformal embedding provides inductive bias for radial structure

Test case: BPST instanton — exact SU(2) solution with topological charge Q=1.

Usage:
    uv run python -m experiments.dbg_yang_mills --epochs 300
    uv run python -m experiments.dbg_yang_mills --strict-ortho --rho 1.0 --save-plots
    uv run python -m experiments.dbg_yang_mills --save-plots --output-dir ym_plots
"""

from __future__ import annotations

import sys
import os
import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from core.metric import (
    hermitian_inner_product, hermitian_grade_spectrum,
)
from layers import CliffordLinear
from layers import RotorLayer
from layers import CliffordLayerNorm
from layers.adapters.conformal import ConformalEmbedding
from layers.primitives.base import CliffordModule
from functional.activation import GeometricGELU
from optimizers.riemannian import RiemannianAdam
from functional.orthogonality import StrictOrthogonality, OrthogonalitySettings


# ============================================================================ #
# Constants: su(2) Bivector Indices (same in Cl(3,0) and Cl(4,1))
# ============================================================================ #

# su(2) ≅ spatial bivectors: e₁₂(idx 3), e₁₃(idx 5), e₂₃(idx 6)
# These indices are identical in Cl(3,0) and CGA Cl(4,1) since they
# only involve basis vectors e₁, e₂, e₃ (bits 0-2).
_BV_INDICES = [3, 5, 6]

# Color a ∈ {0,1,2} → bivector index, with sign convention
# a=0 (color 1) → e₂₃(6), a=1 (color 2) → e₁₃(5) with sign flip, a=2 (color 3) → e₁₂(3)
_BV_MAP_INDICES = torch.tensor([6, 5, 3])
_BV_MAP_SIGNS = torch.tensor([1.0, -1.0, 1.0])


# ============================================================================ #
# SU2BladeSelector: restrict to su(2) subalgebra within CGA
# ============================================================================ #

class SU2BladeSelector(CliffordModule):
    """Restricts multivector output to su(2) bivectors {e₁₂, e₁₃, e₂₃}.

    In CGA Cl(4,1) there are 10 grade-2 bivectors, but the SU(2) gauge
    algebra only uses the 3 spatial bivectors at indices [3, 5, 6].
    This selector zeros all other components and applies learnable
    sigmoid gates to the su(2) components.

    Attributes:
        su2_gates (nn.Parameter): Learnable gates [channels, 3].
        su2_mask (Tensor): Binary mask selecting su(2) indices.
    """

    def __init__(self, algebra: CliffordAlgebra, channels: int):
        super().__init__(algebra)
        self.channels = channels
        # Learnable gates for the 3 su(2) components, init to pass-through
        self.su2_gates = nn.Parameter(torch.ones(channels, 3))
        # Binary mask: 1 at su(2) indices, 0 elsewhere
        mask = torch.zeros(algebra.dim)
        for idx in _BV_INDICES:
            mask[idx] = 1.0
        self.register_buffer('su2_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate output to su(2) bivectors only.

        Args:
            x: [Batch, Channels, Dim] multivector input.

        Returns:
            [Batch, Channels, Dim] with only su(2) components non-zero.
        """
        gates = torch.sigmoid(self.su2_gates)  # [C, 3]
        # Build full gate: 0 everywhere except gated su(2) indices
        full_gate = torch.zeros(
            self.channels, self.algebra.dim,
            device=x.device, dtype=x.dtype,
        )
        for i, bv_idx in enumerate(_BV_INDICES):
            full_gate[:, bv_idx] = gates[:, i]
        return x * full_gate.unsqueeze(0)  # [B, C, D] * [1, C, D]


# ============================================================================ #
# 't Hooft Symbols and BPST Instanton (Vectorized)
# ============================================================================ #

def _build_thooft_eta() -> torch.Tensor:
    """Precompute 't Hooft eta symbol η^a_μν as tensor [3, 4, 4].

    0-indexed: a ∈ {0,1,2} maps to color {1,2,3},
    μ,ν ∈ {0,1,2,3} maps to spacetime {1,2,3,4}.
    Antisymmetric in (μ, ν).

    Rules (1-indexed):
      η^a_ij = ε_aij   for spatial i,j ∈ {1,2,3}
      η^a_i4 = δ_ai    for i ∈ {1,2,3}
      η^a_4i = -δ_ai   for i ∈ {1,2,3}
    """
    eta = torch.zeros(3, 4, 4)

    # Spatial part: ε_aij (Levi-Civita, 0-indexed)
    # a=0: η^1_{12}=ε_{123}=1 → eta[0,1,2]=1
    # a=1: η^2_{20}=ε_{231}=1 → eta[1,2,0]=1
    # a=2: η^3_{01}=ε_{312}=1 → eta[2,0,1]=1
    eta[0, 1, 2] = 1.0;  eta[0, 2, 1] = -1.0
    eta[1, 2, 0] = 1.0;  eta[1, 0, 2] = -1.0
    eta[2, 0, 1] = 1.0;  eta[2, 1, 0] = -1.0

    # Temporal part: η^a_{i,4} = δ_{a,i} (0-indexed: μ=a, ν=3)
    for a in range(3):
        eta[a, a, 3] = 1.0
        eta[a, 3, a] = -1.0

    return eta


# Module-level constant — precomputed once
_ETA_THOOFT = _build_thooft_eta()


def bpst_gauge_potential(x: torch.Tensor, rho: float = 1.0,
                         algebra_dim: int = 32) -> torch.Tensor:
    """Vectorized BPST instanton gauge potential A_μ^a(x) in regular gauge.

    A_μ^a(x) = η^a_μν x_ν / (|x|² + ρ²)

    Args:
        x: [B, 4] spacetime coordinates (x₁, x₂, x₃, x₄).
        rho: Instanton size parameter.
        algebra_dim: Algebra dimension (32 for Cl(4,1), 8 for Cl(3,0)).

    Returns:
        A_mu: [B, 4, algebra_dim] — 4 gauge potential components as multivectors.
              Each is a grade-2 bivector in su(2).
    """
    B = x.shape[0]
    r2 = (x ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    denom = r2 + rho ** 2  # [B, 1]

    eta = _ETA_THOOFT.to(x.device, x.dtype)  # [3, 4, 4]
    bv_idx = _BV_MAP_INDICES.to(x.device)
    bv_signs = _BV_MAP_SIGNS.to(x.device, x.dtype)

    # Contract: coeff[b, a, mu] = Σ_ν η[a, mu, ν] * x[b, ν]
    coeff = torch.einsum('amn, bn -> bam', eta, x)  # [B, 3, 4]
    coeff = coeff / denom.unsqueeze(-1)  # [B, 1, 1] broadcast to [B, 3, 4]

    # Scatter to bivector indices via mask multiplication
    # (fully functional — no in-place ops to preserve autograd graph)
    A_mu = torch.zeros(B, 4, algebra_dim, dtype=x.dtype, device=x.device)
    for a in range(3):
        mask = torch.zeros(algebra_dim, device=x.device, dtype=x.dtype)
        mask[bv_idx[a]] = 1.0
        A_mu = A_mu + bv_signs[a] * coeff[:, a, :].unsqueeze(-1) * mask

    return A_mu


def bpst_field_strength(x: torch.Tensor, rho: float = 1.0,
                        algebra_dim: int = 32
                        ) -> Dict[Tuple[int, int], torch.Tensor]:
    """Vectorized BPST instanton field strength F_μν^a(x).

    F_μν^a(x) = -2ρ² η^a_μν / (|x|² + ρ²)²

    Sign convention: matches GA commutator [T_a,T_b]_GA = -2ε_{abc}T_c
    with F = abelian - [A_μ, A_ν]_GA (regular gauge).

    Args:
        x: [B, 4] spacetime coordinates.
        rho: Instanton size parameter.
        algebra_dim: Algebra dimension (32 for Cl(4,1), 8 for Cl(3,0)).

    Returns:
        Dict mapping (mu, nu) pairs (0-indexed, mu < nu) to [B, algebra_dim] MVs.
    """
    B = x.shape[0]
    r2 = (x ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    prefactor = -2.0 * rho ** 2 / (r2 + rho ** 2) ** 2  # [B, 1]

    eta = _ETA_THOOFT.to(x.device, x.dtype)  # [3, 4, 4]
    bv_idx = _BV_MAP_INDICES.to(x.device)
    bv_signs = _BV_MAP_SIGNS.to(x.device, x.dtype)

    F_dict = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F_mv = torch.zeros(B, algebra_dim, dtype=x.dtype, device=x.device)
            for a in range(3):
                e = eta[a, mu, nu].item()
                if e != 0.0:
                    F_mv[:, bv_idx[a]] = bv_signs[a] * e * prefactor.squeeze(-1)
            F_dict[(mu, nu)] = F_mv

    return F_dict


def bpst_action_density(x: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
    """Exact BPST action density: 48ρ⁴ / (|x|² + ρ²)⁴.

    Args:
        x: [B, 4] spacetime coordinates.
        rho: Instanton size.

    Returns:
        [B] action density at each point.
    """
    r2 = (x ** 2).sum(dim=-1)
    return 48.0 * rho ** 4 / (r2 + rho ** 2) ** 4


# ============================================================================ #
# Hodge Dual (Spacetime, Precomputed Map)
# ============================================================================ #

# 4D Euclidean spacetime Hodge dual: maps (mu,nu) → (src_pair, sign)
# This acts on spacetime indices, NOT on internal gauge algebra components.
_HODGE_DUAL_MAP: Dict[Tuple[int, int], Tuple[Tuple[int, int], float]] = {
    (0, 1): ((2, 3),  1.0),
    (0, 2): ((1, 3), -1.0),
    (0, 3): ((1, 2),  1.0),
    (1, 2): ((0, 3),  1.0),
    (1, 3): ((0, 2), -1.0),
    (2, 3): ((0, 1),  1.0),
}


def hodge_dual_4d(F_dict: Dict[Tuple[int, int], torch.Tensor]
                  ) -> Dict[Tuple[int, int], torch.Tensor]:
    """Hodge dual of 2-form F in 4D Euclidean spacetime.

    Acts on spacetime indices (μ,ν), preserving internal gauge algebra structure.
    Maps: *F₀₁ = F₂₃, *F₀₂ = -F₁₃, *F₀₃ = F₁₂, etc.
    """
    return {key: sign * F_dict[src]
            for key, (src, sign) in _HODGE_DUAL_MAP.items()
            if src in F_dict}


# ============================================================================ #
# Dataset
# ============================================================================ #

class BPSTInstantonDataset(Dataset):
    """Dataset of BPST instanton samples.

    Points sampled from exponential distribution concentrated near
    instanton core (origin), with exact A_μ and F_μν as targets.

    Each sample: (coords [4], A_mu [4, algebra_dim], action_density [1]).
    """

    def __init__(self, num_samples: int, rho: float = 1.0,
                 sampling_radius: float = 5.0, seed: int = 42,
                 algebra_dim: int = 32):
        rng = np.random.RandomState(seed)

        # Sample points concentrated near instanton core
        directions = rng.randn(num_samples, 4).astype(np.float32)
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = directions / (norms + 1e-8)
        radii = rng.exponential(scale=2.0 * rho, size=(num_samples, 1)).astype(np.float32)
        radii = np.clip(radii, 0.1 * rho, sampling_radius)
        coords_np = directions * radii

        self.coords = torch.tensor(coords_np)  # [N, 4]

        # Compute exact gauge potential (vectorized)
        self.A_mu = bpst_gauge_potential(
            self.coords, rho, algebra_dim=algebra_dim)  # [N, 4, algebra_dim]

        # Compute exact action density
        self.action_density = bpst_action_density(self.coords, rho)  # [N]

        self.rho = rho

        # Stats
        r = self.coords.norm(dim=-1)
        print(f"  BPST dataset: {num_samples} points, rho={rho:.2f}, "
              f"r range=[{r.min().item():.3f}, {r.max().item():.3f}], "
              f"mean={r.mean().item():.3f}")

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int):
        return self.coords[idx], self.A_mu[idx], self.action_density[idx]


# ============================================================================ #
# Network
# ============================================================================ #

class YangMillsNet(CliffordModule):
    """CGA network for SU(2) gauge fields in Cl(4,1).

    Architecture:
        1. Conformal embedding of spatial coords + Fourier features + time
        2. Lift to multivector hidden space [B, C, 32]
        3. Pre-LN residual blocks: norm → rotor → act → linear + skip
           (CGA rotors can learn translations, dilations, spatial rotations)
        4. Output: CliffordLinear(C, 4) → [B, 4, 32] — 4 gauge potential channels
        5. SU2BladeSelector enforces su(2) purity (A_μ ∈ {e₁₂, e₁₃, e₂₃})
        6. No external envelope — conformal structure handles decay implicitly

    Returns A_μ [B, 4, 32] and optionally intermediates.
    """

    def __init__(self, algebra, hidden_dim: int = 64, num_layers: int = 6,
                 num_freqs: int = 32, rho: float = 1.0):
        super().__init__(algebra)
        self.hidden_dim = hidden_dim
        self.rho = rho

        # Conformal embedding for spatial coordinates (R^3 → CGA null cone)
        self.conformal_embed = ConformalEmbedding(algebra, euclidean_dim=3)

        # Fourier embedding: learned frequencies for all 4 coords
        self.register_buffer('freq_bands', torch.randn(4, num_freqs) * 2.0)
        # raw(4) + sin/cos(2*num_freqs) + conformal(32) + time(t, t²) = 38 + 2*num_freqs
        input_dim = 4 + 2 * num_freqs + algebra.dim + 2

        # Lift
        self.input_lift = nn.Linear(input_dim, hidden_dim * algebra.dim)
        self.input_norm = CliffordLayerNorm(algebra, hidden_dim)

        # Residual GA blocks (Pre-LN) — CGA rotors learn conformal versors
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'norm': CliffordLayerNorm(algebra, hidden_dim),
                'rotor': RotorLayer(algebra, hidden_dim),
                'act': GeometricGELU(algebra, channels=hidden_dim),
                'linear': CliffordLinear(algebra, hidden_dim, hidden_dim),
            }))

        # Output: 4 gauge potential channels
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.output_proj = CliffordLinear(algebra, hidden_dim, 4)

        # SU(2) blade selector — restrict to su(2) bivectors within Cl(4,1)
        self.su2_selector = SU2BladeSelector(algebra, channels=4)

    def forward(self, coords: torch.Tensor,
                store_intermediates: bool = False):
        """Forward pass.

        Args:
            coords: [B, 4] spacetime coordinates (x₁, x₂, x₃, t).
            store_intermediates: If True, return list of hidden states
                for orthogonality diagnostics. Default False to save memory.

        Returns:
            A_mu: [B, 4, 32] predicted gauge potential (4 su(2) channels).
            intermediates: list of hidden states (empty if store_intermediates=False).
        """
        B = coords.shape[0]

        # Split spatial and temporal coordinates
        x_spatial = coords[:, :3]  # [B, 3]
        t = coords[:, 3:4]  # [B, 1]

        # Conformal embedding: spatial coords → CGA null cone
        # P(x) = x + 0.5|x|²·e_inf + e_o  encodes radial info algebraically
        P = self.conformal_embed.embed(x_spatial)  # [B, 32]

        # Fourier features
        proj = coords @ self.freq_bands  # [B, num_freqs]

        # Time features
        t_sq = t ** 2  # [B, 1]

        # Assemble: [coords, sin, cos, P_conformal, t, t²]
        features = torch.cat([
            coords,  # [B, 4]
            torch.sin(proj), torch.cos(proj),  # [B, 2*num_freqs]
            P,  # [B, 32] — conformal embedding
            t, t_sq,  # [B, 2] — time features
        ], dim=-1)

        # Lift to multivector space
        h = self.input_lift(features)
        h = h.reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)

        # Residual blocks — CGA rotors learn conformal transformations
        intermediates: List[torch.Tensor] = []
        for block in self.blocks:
            residual = h
            h = block['norm'](h)
            h = block['rotor'](h)
            h = block['act'](h)
            h = block['linear'](h)
            h = residual + h
            if store_intermediates:
                intermediates.append(h.detach())

        # Output: [B, 4, 32]
        h = self.output_norm(h)
        A_mu = self.output_proj(h)

        # Enforce su(2) purity — conformal decay is implicit in the
        # CGA rotor structure, no external envelope needed
        A_mu = self.su2_selector(A_mu)

        return A_mu, intermediates


# ============================================================================ #
# Field Strength Computation (Jacobian-Cached, 12 autograd calls)
# ============================================================================ #

def _compute_jacobian(A_mu: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian of grade-2 components: J[b, nu, c_idx, mu] = ∂A_ν[bv_c]/∂x_μ.

    Exploits grade-2 purity: only 3 non-zero components (indices 3, 5, 6)
    instead of all D. Each autograd.grad call returns all 4 partial derivatives.

    Total: 4 directions × 3 components = 12 autograd.grad calls (vs 96 naive).

    Args:
        A_mu: [B, 4, D] gauge potential (must be in autograd graph).
        coords: [B, 4] with requires_grad=True.

    Returns:
        jacobian: [B, 4, 3, 4] where [b, nu, c_idx, mu] = ∂A_ν[_BV_INDICES[c_idx]]/∂x_μ
    """
    B = A_mu.shape[0]
    jacobian = torch.zeros(B, 4, 3, 4, device=A_mu.device, dtype=A_mu.dtype)

    for nu in range(4):
        for c_idx, bv_comp in enumerate(_BV_INDICES):
            grad = torch.autograd.grad(
                A_mu[:, nu, bv_comp].sum(), coords,
                create_graph=True, retain_graph=True
            )[0]  # [B, 4] — all 4 partial derivatives at once
            jacobian[:, nu, c_idx, :] = grad

    return jacobian


def compute_field_strength(algebra, A_mu: torch.Tensor,
                           coords: torch.Tensor
                           ) -> Dict[Tuple[int, int], torch.Tensor]:
    """Compute Yang-Mills field strength F_μν = ∂_μA_ν - ∂_νA_μ - [A_μ, A_ν]_GA.

    CGA geometric calculus formulation:
    - Jacobian J encodes the vector derivative ∇A (12 autograd calls)
    - Abelian part: exterior derivative ∇∧A from antisymmetric J
    - Non-abelian part: commutator [A_μ, A_ν] via algebra.commutator()
      (single-pass using precomputed comm_gp_signs, vs 2 GP calls naive)

    Args:
        algebra: CliffordAlgebra instance.
        A_mu: [B, 4, D] gauge potential (must be in autograd graph).
        coords: [B, 4] with requires_grad=True.

    Returns:
        Dict mapping (mu, nu) pairs (mu < nu) to [B, D] field strength bivectors.
    """
    J = _compute_jacobian(A_mu, coords)  # [B, 4, 3, 4]
    B = A_mu.shape[0]
    D = A_mu.shape[-1]
    F_dict = {}

    for mu in range(4):
        for nu in range(mu + 1, 4):
            # Abelian part: ∂_μA_ν - ∂_νA_μ (assemble from cached Jacobian)
            abelian = torch.zeros(B, D, device=A_mu.device, dtype=A_mu.dtype)
            for c_idx, bv_comp in enumerate(_BV_INDICES):
                abelian[:, bv_comp] = J[:, nu, c_idx, mu] - J[:, mu, c_idx, nu]

            # Non-abelian part: [A_μ, A_ν] via single-pass commutator
            commutator = algebra.commutator(A_mu[:, mu], A_mu[:, nu])

            # GA commutator gives [T_a,T_b]_GA = -2ε_{abc}T_c, but BPST
            # convention uses [T_a,T_b] = +2ε_{abc}T_c, so:
            #   GA_commutator = -(physics commutator)
            #   F = abelian + physics_comm = abelian - GA_comm
            F_dict[(mu, nu)] = abelian - commutator

    return F_dict


# ============================================================================ #
# Training Losses (Lean: supervised + self-duality + action)
# ============================================================================ #

def compute_training_losses(algebra, model: YangMillsNet, coords: torch.Tensor,
                            A_exact: torch.Tensor, action_exact: torch.Tensor,
                            rho: float = 1.0,
                            store_intermediates: bool = False
                            ) -> Dict[str, torch.Tensor]:
    """Compute training loss terms (no expensive 2nd-order derivatives).

    Loss terms:
      - supervised: MSE(A_pred, A_exact) — direct gauge potential matching
      - self_duality: |F - *F|² — core physics constraint (F = *F for instanton)
      - action: MSE(Σ<F,F>_H, exact) — action density matching
      - sobolev: MSE(F_pred, F_exact) — derivative-level supervision
      - q_loss: (Q_pred/Q_exact - 1)² — differentiable topological charge

    Total autograd cost: 12 grad calls (field strength Jacobian only).

    Args:
        algebra: CliffordAlgebra instance.
        model: YangMillsNet.
        coords: [B, 4] with requires_grad=True.
        A_exact: [B, 4, D] exact gauge potential.
        action_exact: [B] exact action density.
        rho: instanton size parameter (for exact F computation).
        store_intermediates: pass through to model.forward().

    Returns:
        Dict with 'supervised', 'self_duality', 'action', 'sobolev',
        'q_loss', 'A_pred', 'F_dict', 'intermediates'.
    """
    A_pred, intermediates = model(coords, store_intermediates=store_intermediates)

    # --- Supervised loss on A ---
    supervised_loss = nn.functional.mse_loss(A_pred, A_exact)

    # --- Field strength (12 autograd calls) ---
    F_dict = compute_field_strength(algebra, A_pred, coords)

    # --- Self-duality: F = *F (core instanton constraint) ---
    F_dual = hodge_dual_4d(F_dict)
    sd_loss = torch.tensor(0.0, device=coords.device)
    n_pairs = 0
    for key in F_dict:
        if key in F_dual:
            sd_loss = sd_loss + ((F_dict[key] - F_dual[key]) ** 2).mean()
            n_pairs += 1
    if n_pairs > 0:
        sd_loss = sd_loss / n_pairs

    # --- Action density: Σ <F_μν, F_μν>_H ---
    action_pred = torch.zeros(coords.shape[0], device=coords.device)
    for key, F in F_dict.items():
        action_pred = action_pred + hermitian_inner_product(algebra, F, F).squeeze(-1)
    action_loss = nn.functional.mse_loss(action_pred, action_exact)

    # --- Sobolev loss: match field strength derivatives ---
    F_exact = bpst_field_strength(coords.detach(), rho,
                                  algebra_dim=algebra.dim)
    sobolev_loss = torch.tensor(0.0, device=coords.device)
    n_f = 0
    for key in F_dict:
        if key in F_exact:
            sobolev_loss = sobolev_loss + ((F_dict[key] - F_exact[key]) ** 2).mean()
            n_f += 1
    if n_f > 0:
        sobolev_loss = sobolev_loss / n_f

    # --- Differentiable topological charge loss ---
    # Q density: Σ <F_μν, *F_μν>_H per point
    q_density = torch.zeros(coords.shape[0], device=coords.device)
    for key in F_dict:
        if key in F_dual:
            q_density = q_density + hermitian_inner_product(
                algebra, F_dict[key], F_dual[key]).squeeze(-1)

    # Volume-weighted integral: r³ weighting for 4D spherical shell
    r = coords.detach().norm(dim=-1).clamp(min=0.1)
    vol_weight = r ** 3
    vol_weight = vol_weight / vol_weight.sum()

    # Scale-invariant: match ratio of weighted sums (pred vs exact)
    pred_q_proxy = (q_density * vol_weight).sum()
    exact_q_proxy = (action_exact * vol_weight).sum()
    q_loss = (pred_q_proxy / (exact_q_proxy.detach() + 1e-8) - 1.0) ** 2

    return {
        'supervised': supervised_loss,
        'self_duality': sd_loss,
        'action': action_loss,
        'sobolev': sobolev_loss,
        'q_loss': q_loss,
        'A_pred': A_pred,
        'F_dict': F_dict,
        'intermediates': intermediates,
    }


# ============================================================================ #
# Diagnostic Losses (Eval-only: YM equation, Bianchi, gauge, purity)
# ============================================================================ #

def compute_diagnostic_losses(algebra, A_pred: torch.Tensor,
                              F_dict: Dict[Tuple[int, int], torch.Tensor],
                              coords: torch.Tensor
                              ) -> Dict[str, torch.Tensor]:
    """Compute expensive diagnostic losses (eval-only, not for training).

    These require 2nd-order derivatives (YM equation, Bianchi) or random
    sampling (gauge covariance). They verify physical constraints that are
    either automatically satisfied (YM, Bianchi for self-dual solutions)
    or algebraically guaranteed (gauge covariance, grade purity).

    Args:
        algebra: CliffordAlgebra instance.
        A_pred: [B, 4, D] predicted gauge potential.
        F_dict: precomputed field strength dict.
        coords: [B, 4] with requires_grad=True.

    Returns:
        Dict with 'ym_equation', 'bianchi', 'gauge', 'purity'.
    """
    B = coords.shape[0]
    D = A_pred.shape[-1]
    device = coords.device

    # --- Grade purity (A should be pure grade-2, enforced by SU2BladeSelector) ---
    purity_loss = torch.tensor(0.0, device=device)
    for mu in range(4):
        A_mu_proj = algebra.grade_projection(A_pred[:, mu], 2)
        residual = A_pred[:, mu] - A_mu_proj
        purity_loss = purity_loss + (residual ** 2).mean()
    purity_loss = purity_loss / 4.0

    # --- Gauge covariance: <RFR̃, RFR̃>_H = <F, F>_H ---
    gauge_loss = _gauge_covariance_loss(algebra, F_dict)

    # --- YM equation residual: D_μ F^μν = 0 ---
    ym_loss = torch.tensor(0.0, device=device)
    for nu_target in range(4):
        residual_nu = torch.zeros(B, D, device=device)
        for mu in range(4):
            if mu == nu_target:
                continue
            if mu < nu_target:
                F_mn = F_dict.get((mu, nu_target),
                                  torch.zeros(B, D, device=device))
            else:
                F_mn = -F_dict.get((nu_target, mu),
                                   torch.zeros(B, D, device=device))

            # ∂F^μν/∂x_μ (only grade-2 components)
            for _, bv_comp in enumerate(_BV_INDICES):
                grad = torch.autograd.grad(
                    F_mn[:, bv_comp].sum(), coords,
                    create_graph=False, retain_graph=True
                )[0]  # [B, 4]
                residual_nu[:, bv_comp] = residual_nu[:, bv_comp] + grad[:, mu]

            # [A_μ, F^μν] via single-pass commutator
            comm = algebra.commutator(A_pred[:, mu], F_mn)
            residual_nu = residual_nu + comm

        ym_loss = ym_loss + (residual_nu ** 2).mean()
    ym_loss = ym_loss / 4.0

    # --- Bianchi identity: D_{[μ} F_{νρ]} = 0 ---
    bianchi_loss = torch.tensor(0.0, device=device)
    triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for mu, nu, rho_idx in triples:
        F_nu_rho = F_dict.get((nu, rho_idx),
                              torch.zeros(B, D, device=device))
        F_rho_mu = F_dict.get((min(rho_idx, mu), max(rho_idx, mu)),
                              torch.zeros(B, D, device=device))
        if rho_idx > mu:
            pass  # correct sign
        else:
            F_rho_mu = -F_rho_mu
        F_mu_nu = F_dict.get((mu, nu),
                             torch.zeros(B, D, device=device))

        # D_μ F_{νρ} = ∂_μ F_{νρ} + [A_μ, F_{νρ}]
        dF_nu_rho = torch.zeros_like(F_nu_rho)
        dF_rho_mu = torch.zeros_like(F_rho_mu)
        dF_mu_nu = torch.zeros_like(F_mu_nu)

        for _, bv_comp in enumerate(_BV_INDICES):
            g1 = torch.autograd.grad(
                F_nu_rho[:, bv_comp].sum(), coords,
                create_graph=False, retain_graph=True
            )[0]
            dF_nu_rho[:, bv_comp] = g1[:, mu]

            g2 = torch.autograd.grad(
                F_rho_mu[:, bv_comp].sum(), coords,
                create_graph=False, retain_graph=True
            )[0]
            dF_rho_mu[:, bv_comp] = g2[:, nu]

            g3 = torch.autograd.grad(
                F_mu_nu[:, bv_comp].sum(), coords,
                create_graph=False, retain_graph=True
            )[0]
            dF_mu_nu[:, bv_comp] = g3[:, rho_idx]

        comm1 = algebra.commutator(A_pred[:, mu], F_nu_rho)
        comm2 = algebra.commutator(A_pred[:, nu], F_rho_mu)
        comm3 = algebra.commutator(A_pred[:, rho_idx], F_mu_nu)

        bianchi_residual = ((dF_nu_rho + comm1) + (dF_rho_mu + comm2)
                           + (dF_mu_nu + comm3))
        bianchi_loss = bianchi_loss + (bianchi_residual ** 2).mean()
    bianchi_loss = bianchi_loss / len(triples)

    return {
        'ym_equation': ym_loss,
        'bianchi': bianchi_loss,
        'gauge': gauge_loss,
        'purity': purity_loss,
    }


def _gauge_covariance_loss(algebra, F_dict: Dict[Tuple[int, int], torch.Tensor]
                           ) -> torch.Tensor:
    """Test gauge covariance: <RFR̃, RFR̃>_H = <F, F>_H for random R.

    This should be zero by algebraic construction of the Hermitian inner product.
    Non-zero values indicate numerical precision issues, not model failure.
    """
    device = next(iter(F_dict.values())).device
    bv = torch.zeros(1, algebra.dim, device=device)
    bv_coeffs = torch.randn(3, device=device) * 0.3
    bv[0, 3] = bv_coeffs[0]
    bv[0, 5] = bv_coeffs[1]
    bv[0, 6] = bv_coeffs[2]

    rotor = algebra.exp(-0.5 * bv)
    rotor_rev = algebra.reverse(rotor)

    total_err = torch.tensor(0.0, device=device)
    n = 0
    for _, F in F_dict.items():
        B = F.shape[0]
        # Sandwich product via algebra
        temp = algebra.geometric_product(rotor.expand(B, -1), F)
        F_transformed = algebra.geometric_product(temp, rotor_rev.expand(B, -1))

        norm_orig = hermitian_inner_product(algebra, F, F)
        norm_trans = hermitian_inner_product(algebra, F_transformed, F_transformed)
        total_err = total_err + (norm_orig - norm_trans).abs().mean()
        n += 1

    return total_err / max(n, 1)


# ============================================================================ #
# Gauge YM Metric
# ============================================================================ #

class GaugeYMMetric:
    """Gauge-theoretic metrics for Yang-Mills in CGA Cl(4,1).

    All metrics use the Hermitian inner product, which is algebraically
    gauge-invariant: <RAR̃, RBR̃>_H = <A,B>_H for any rotor R ∈ Spin(3).
    """

    def __init__(self, algebra):
        self.algebra = algebra

    def action_density(self, F_dict: Dict[Tuple[int, int], torch.Tensor]
                       ) -> torch.Tensor:
        """Σ_{μ<ν} <F_μν, F_μν>_H per point."""
        result = None
        for F in F_dict.values():
            contrib = hermitian_inner_product(self.algebra, F, F).squeeze(-1)
            result = contrib if result is None else result + contrib
        return result

    def topological_charge(self, F_dict: Dict[Tuple[int, int], torch.Tensor]
                           ) -> torch.Tensor:
        """Topological charge density: Σ <F_μν, *F_μν>_H.

        For instanton: integrated over all space should give 8π².
        Uses Hodge dual in spacetime indices, Hermitian inner product
        in gauge algebra — both intrinsic GA operations.
        """
        F_dual = hodge_dual_4d(F_dict)
        result = torch.tensor(0.0, device=next(iter(F_dict.values())).device)
        for key in F_dict:
            if key in F_dual:
                contrib = hermitian_inner_product(self.algebra, F_dict[key],
                                                  F_dual[key]).squeeze(-1)
                result = result + contrib.mean()
        return result

    def self_duality_error(self, F_dict: Dict[Tuple[int, int], torch.Tensor]
                           ) -> float:
        """Mean |F - *F|² across all pairs."""
        F_dual = hodge_dual_4d(F_dict)
        total = 0.0
        n = 0
        for key in F_dict:
            if key in F_dual:
                err = ((F_dict[key] - F_dual[key]) ** 2).mean().item()
                total += err
                n += 1
        return total / max(n, 1)

    def gauge_covariance_error(self, F_dict: Dict[Tuple[int, int], torch.Tensor]
                               ) -> float:
        """Random rotor test for gauge covariance (should be ~0 by construction)."""
        with torch.no_grad():
            return _gauge_covariance_loss(self.algebra, F_dict).item()

    def grade_purity(self, A_mu: torch.Tensor) -> float:
        """Fraction of energy in grade-2 vs total for A_mu."""
        grade2_energy = 0.0
        total_energy = 0.0
        for mu in range(A_mu.shape[1]):
            A = A_mu[:, mu]
            A_proj = self.algebra.grade_projection(A, 2)
            grade2_energy += (A_proj ** 2).sum().item()
            total_energy += (A ** 2).sum().item()
        return grade2_energy / (total_energy + 1e-12)

    def format_report(self, A_mu: torch.Tensor,
                      F_dict: Dict[Tuple[int, int], torch.Tensor]) -> str:
        """ASCII diagnostics report."""
        action = self.action_density(F_dict)
        topo = self.topological_charge(F_dict)
        sd_err = self.self_duality_error(F_dict)
        gauge_err = self.gauge_covariance_error(F_dict)
        purity = self.grade_purity(A_mu)

        lines = [
            "  --- Gauge YM Metric Report ---",
            f"  Action density <F,F>_H:     mean={action.mean().item():.6f}",
            f"  Topological charge density:  {topo.item():.6f}  "
            f"(target: 8pi^2~={8*math.pi**2:.4f})",
            f"  Self-duality error |F-*F|^2: {sd_err:.6e}",
            f"  Gauge covariance error:      {gauge_err:.6e}  "
            f"(algebraically guaranteed ~0)",
            f"  Grade-2 purity:              {purity:.4%}  "
            f"(enforced by SU2BladeSelector)",
        ]
        return '\n'.join(lines)


# ============================================================================ #
# Instanton Debugger
# ============================================================================ #

class YMInstantonDebugger:
    """Physics diagnostics for BPST instanton in CGA Cl(4,1)."""

    def __init__(self, algebra, metric: GaugeYMMetric, rho: float = 1.0):
        self.algebra = algebra
        self.metric = metric
        self.rho = rho

    def action_density_profile(self, model: YangMillsNet,
                               r_grid: torch.Tensor) -> Dict[str, np.ndarray]:
        """Batched radial profile of action density: predicted vs exact.

        All radii processed in a single forward pass + field strength computation.
        """
        model.eval()
        N = len(r_grid)
        device = next(model.parameters()).device

        # Create all points along x₁ axis at once
        coords = torch.zeros(N, 4, device=device, dtype=torch.float32)
        coords[:, 0] = r_grid.to(device)
        coords = coords.requires_grad_(True)

        with torch.enable_grad():
            A_mu, _ = model(coords)
            F_dict = compute_field_strength(self.algebra, A_mu, coords)

        # Predicted action: Σ <F_μν, F_μν>_H per point
        pred_action = torch.zeros(N, device=device)
        for F in F_dict.values():
            pred_action = pred_action + hermitian_inner_product(
                self.algebra, F.detach(), F.detach()
            ).squeeze(-1)

        # Exact action density at each radius
        r2 = r_grid ** 2
        exact_action = 48.0 * self.rho ** 4 / (r2 + self.rho ** 2) ** 4

        return {
            'r': r_grid.cpu().numpy(),
            'pred_action': pred_action.cpu().detach().numpy(),
            'exact_action': exact_action.cpu().numpy(),
        }

    @torch.no_grad()
    def topological_charge_integration(self, model: YangMillsNet,
                                       R_max: float = 5.0,
                                       n_radii: int = 10
                                       ) -> Dict[str, np.ndarray]:
        """Vectorized topological charge as function of integration radius.

        All MC points generated and processed in a single batch.
        """
        model.eval()
        radii = np.linspace(0.5, R_max, n_radii)
        pts_per_radius = 100
        device = next(model.parameters()).device

        # Generate all MC points at once
        rng = np.random.RandomState(123)
        all_pts = []
        radius_labels = []
        for i, R in enumerate(radii):
            pts = rng.randn(pts_per_radius, 4).astype(np.float32)
            norms = np.linalg.norm(pts, axis=-1, keepdims=True)
            pts = pts / (norms + 1e-8) * rng.uniform(
                0, R, (pts_per_radius, 1)).astype(np.float32)
            all_pts.append(pts)
            radius_labels.extend([i] * pts_per_radius)

        all_pts_np = np.concatenate(all_pts, axis=0)  # [n_radii*100, 4]
        coords = torch.tensor(all_pts_np, device=device)

        # Single batch: exact action density
        action_exact = bpst_action_density(coords, self.rho)  # [N]

        # Aggregate by radius
        radius_labels_t = torch.tensor(radius_labels, device=device)
        Q_values = []
        for i, R in enumerate(radii):
            mask = radius_labels_t == i
            a_mean = action_exact[mask].mean().item()
            Q_approx = a_mean * (R ** 4) * (math.pi ** 2 / 2.0)
            Q_values.append(Q_approx / (8.0 * math.pi ** 2))

        return {
            'R': np.array(radii),
            'Q': np.array(Q_values),
        }

    def self_duality_check(self, model: YangMillsNet,
                           coords: torch.Tensor) -> Dict[str, float]:
        """Check F = *F for predicted fields."""
        model.eval()
        coords = coords.clone().requires_grad_(True)

        with torch.enable_grad():
            A_mu, _ = model(coords)
            F_dict = compute_field_strength(self.algebra, A_mu, coords)

        sd_err = self.metric.self_duality_error(
            {k: v.detach() for k, v in F_dict.items()}
        )
        return {'self_duality_error': sd_err}

    def gauge_covariance_test(self, model: YangMillsNet,
                              coords: torch.Tensor) -> float:
        """Apply random rotors, check gauge invariance."""
        model.eval()
        coords = coords.clone().requires_grad_(True)

        with torch.enable_grad():
            A_mu, _ = model(coords)
            F_dict = compute_field_strength(self.algebra, A_mu, coords)

        return self.metric.gauge_covariance_error(
            {k: v.detach() for k, v in F_dict.items()}
        )

    def field_strength_spectrum(self, model: YangMillsNet,
                                coords: torch.Tensor) -> Dict[str, float]:
        """Grade spectrum of F (should be pure grade-2)."""
        model.eval()
        coords = coords.clone().requires_grad_(True)

        with torch.enable_grad():
            A_mu, _ = model(coords)
            F_dict = compute_field_strength(self.algebra, A_mu, coords)

        # Average spectrum over all F components
        total_spectrum = None
        n = 0
        for F in F_dict.values():
            spec = hermitian_grade_spectrum(self.algebra, F.detach())
            if total_spectrum is None:
                total_spectrum = spec.mean(dim=0)
            else:
                total_spectrum = total_spectrum + spec.mean(dim=0)
            n += 1

        if total_spectrum is not None:
            total_spectrum = total_spectrum / n
            n_grades = total_spectrum.shape[0]
            labels = [f'G{k}' for k in range(n_grades)]
            return {labels[k]: total_spectrum[k].item() for k in range(n_grades)}
        return {}

    def format_report(self, model: YangMillsNet,
                      coords: torch.Tensor) -> str:
        """Human-readable diagnostic report."""
        model.eval()
        with torch.no_grad():
            A_mu, _ = model(coords)

        lines = ["  --- YM Instanton Debugger Report ---"]

        # Grade purity of A
        purity = self.metric.grade_purity(A_mu.detach())
        lines.append(f"  A_mu grade-2 purity:         {purity:.4%}")

        # Supervised error
        A_exact = bpst_gauge_potential(
            coords.detach(), self.rho,
            algebra_dim=self.algebra.dim)
        sup_err = nn.functional.mse_loss(A_mu.detach(), A_exact).item()
        lines.append(f"  Supervised MSE(A):          {sup_err:.6e}")

        # Action density comparison (batched)
        r_grid = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0])
        profile = self.action_density_profile(model, r_grid)
        lines.append("  Action density profile (|A|^2 proxy):")
        for r, pred, exact in zip(profile['r'], profile['pred_action'],
                                  profile['exact_action']):
            lines.append(f"    r={r:.1f}: pred={pred:.6f}, exact={exact:.6f}")

        return '\n'.join(lines)


# ============================================================================ #
# Visualization
# ============================================================================ #

def _save_plots(history: dict, model: YangMillsNet,
                debugger: YMInstantonDebugger,
                algebra, ortho: StrictOrthogonality,
                last_intermediates: List[torch.Tensor],
                test_coords: torch.Tensor,
                rho: float,
                output_dir: str) -> None:
    """Save diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    device = test_coords.device

    # ------------------------------------------------------------------ #
    # 1. Convergence curves
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history['epochs']

    ax = axes[0]
    for key, label, color in [
        ('supervised', 'Supervised', 'steelblue'),
        ('action', 'Action density', 'tomato'),
        ('sobolev', 'Sobolev (F)', 'forestgreen'),
        ('q_loss', 'Q (topo)', 'darkorange'),
        ('total', 'Total', 'black'),
    ]:
        if history.get(key):
            ax.semilogy(epochs, history[key], label=label, color=color)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.set_title('Training Convergence'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key, label, color in [
        ('self_duality', 'Self-duality', 'purple'),
    ]:
        if history.get(key):
            ax.semilogy(epochs, history[key], label=label, color=color)
    # Overlay Q tracking on secondary axis
    if history.get('Q_estimates'):
        ax2 = ax.twinx()
        ax2.plot(epochs[:len(history['Q_estimates'])],
                 history['Q_estimates'], 'go-', markersize=3,
                 label='Q estimate', alpha=0.8)
        ax2.axhline(1.0, color='red', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Topological Charge Q', color='green')
        ax2.legend(loc='upper left')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.set_title('Physics Losses + Q Tracking'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Yang-Mills SU(2) Gauge Theory - CGA Cl(4,1)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Action density profile
    # ------------------------------------------------------------------ #
    r_grid = torch.linspace(0.1, 8.0, 50)
    profile = debugger.action_density_profile(model, r_grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(profile['r'], profile['exact_action'], 'b-', linewidth=2,
            label=r'Exact: $48\rho^4/(r^2+\rho^2)^4$')
    ax.plot(profile['r'], profile['pred_action'], 'r--o', markersize=3,
            label='Predicted')
    # Error band
    pred = profile['pred_action']
    exact = profile['exact_action']
    rel_err = np.abs(pred - exact) / (np.abs(exact) + 1e-12)
    ax.fill_between(profile['r'], exact * (1 - rel_err), exact * (1 + rel_err),
                    alpha=0.15, color='red', label='Error band')
    ax.set_xlabel('r = |x|')
    ax.set_ylabel('Action density')
    ax.set_title(f'Action Density Profile (rho={rho:.1f})')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'action_density.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. Grade spectrum for A
    # ------------------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        A_mu, _ = model(test_coords)

    # Average spectrum over all 4 components
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # A spectrum — Cl(4,1) has 6 grades (0..5)
    n_grades = algebra.n + 1  # n=5 for Cl(4,1) → 6 grades
    a_spec_total = torch.zeros(n_grades)
    for mu in range(4):
        spec = hermitian_grade_spectrum(algebra, A_mu[:, mu].detach()).mean(dim=0).cpu()
        a_spec_total = a_spec_total + spec
    a_spec_total = a_spec_total / 4.0

    ax = axes[0]
    grade_names = ['G0\n(scalar)', 'G1\n(vector)', 'G2\n(bivector)',
                   'G3\n(trivec)', 'G4\n(quadvec)', 'G5\n(pseudo)']
    labels = grade_names[:n_grades]
    # Highlight grade-2 (su(2) target)
    colors = ['gray'] * n_grades
    colors[2] = 'mediumseagreen'  # grade-2 is the target
    bars = ax.bar(range(n_grades), a_spec_total.numpy(), color=colors, alpha=0.8,
                  edgecolor='white')
    for bar, val in zip(bars, a_spec_total.numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(n_grades)); ax.set_xticklabels(labels)
    ax.set_ylabel('Hermitian Energy')
    ax.set_title('A_mu Grade Spectrum\n(should be G2-dominated)')
    ax.grid(True, alpha=0.2, axis='y')

    # Exact A spectrum for comparison
    A_exact = bpst_gauge_potential(test_coords.detach(), rho,
                                   algebra_dim=algebra.dim)
    exact_spec_total = torch.zeros(n_grades)
    for mu in range(4):
        spec = hermitian_grade_spectrum(algebra, A_exact[:, mu]).mean(dim=0).cpu()
        exact_spec_total = exact_spec_total + spec
    exact_spec_total = exact_spec_total / 4.0

    ax = axes[1]
    bars = ax.bar(range(n_grades), exact_spec_total.numpy(), color=colors, alpha=0.8,
                  edgecolor='white')
    for bar, val in zip(bars, exact_spec_total.numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(n_grades)); ax.set_xticklabels(labels)
    ax.set_ylabel('Hermitian Energy'); ax.set_title('A_mu Exact Grade Spectrum')
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'grade_spectrum.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. Predicted A vs exact A scatter + error histogram
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pred_flat = A_mu.detach().reshape(-1).cpu().numpy()
    exact_flat = A_exact.reshape(-1).cpu().numpy()

    n_pts = min(5000, len(pred_flat))
    idx = np.random.choice(len(pred_flat), n_pts, replace=False)

    ax = axes[0]
    ax.scatter(exact_flat[idx], pred_flat[idx], alpha=0.3, s=5, color='steelblue')
    lim = max(abs(exact_flat[idx]).max(), abs(pred_flat[idx]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Exact A'); ax.set_ylabel('Predicted A')
    ax.set_title('A_mu: Predicted vs Exact'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    errors = (A_mu.detach() - A_exact).reshape(-1).cpu().numpy()
    ax.hist(errors, bins=50, color='tomato', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Error (pred - exact)'); ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution (std={errors.std():.4f})')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'self_duality.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5. Topological charge vs integration radius
    # ------------------------------------------------------------------ #
    topo = debugger.topological_charge_integration(model, R_max=6.0, n_radii=15)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(topo['R'], topo['Q'], 'bo-', markersize=4)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Q=1 (target)')
    ax.set_xlabel('Integration Radius R')
    ax.set_ylabel('Topological Charge Q')
    ax.set_title('Topological Charge vs Integration Radius')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'topological_charge.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 6. Coupling heatmap
    # ------------------------------------------------------------------ #
    if last_intermediates:
        h_flat = last_intermediates[-1].reshape(-1, algebra.dim).to(device)
        fig_coup = ortho.visualize_coupling(
            h_flat, title="Cross-Grade Coupling (Last Hidden Layer)"
        )
        if fig_coup is not None:
            fig_coup.savefig(os.path.join(output_dir, 'coupling_heatmap.png'),
                             dpi=150, bbox_inches='tight')
            plt.close(fig_coup)

    # ------------------------------------------------------------------ #
    # 7. Q(epoch) convergence
    # ------------------------------------------------------------------ #
    if history.get('Q_estimates'):
        fig, ax = plt.subplots(figsize=(8, 5))
        q_epochs = epochs[:len(history['Q_estimates'])]
        ax.plot(q_epochs, history['Q_estimates'], 'bo-', markersize=4,
                linewidth=1.5)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.7,
                   label='Q=1 (target)')
        ax.fill_between(q_epochs,
                        [q - 0.05 for q in history['Q_estimates']],
                        [q + 0.05 for q in history['Q_estimates']],
                        alpha=0.15, color='steelblue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Topological Charge Q')
        ax.set_title('Topological Charge Convergence\n'
                     '(Q=1 for single BPST instanton)')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'q_convergence.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # 8. Topological charge density heatmap (2D slice)
    # ------------------------------------------------------------------ #
    _plot_topo_charge_density(model, algebra, rho, output_dir, plt)

    # ------------------------------------------------------------------ #
    # 9. Bivector field visualization (2D slice)
    # ------------------------------------------------------------------ #
    _plot_bivector_field(model, algebra, rho, output_dir, plt)

    # ------------------------------------------------------------------ #
    # 10. Gauge field structure (2D slice)
    # ------------------------------------------------------------------ #
    _plot_gauge_field(model, algebra, rho, output_dir, plt)

    print(f"  Plots saved to {output_dir}/")


def _plot_topo_charge_density(model, algebra, rho, output_dir, plt):
    """Topological charge density Q(x) on x₁-x₂ slice (x₃=x₄=0)."""
    device = next(model.parameters()).device
    model.eval()

    grid_n = 30
    grid_range = 4.0
    x1 = torch.linspace(-grid_range, grid_range, grid_n)
    x2 = torch.linspace(-grid_range, grid_range, grid_n)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

    coords_grid = torch.zeros(grid_n * grid_n, 4, device=device)
    coords_grid[:, 0] = X1.reshape(-1).to(device)
    coords_grid[:, 1] = X2.reshape(-1).to(device)
    coords_grid = coords_grid.requires_grad_(True)

    with torch.enable_grad():
        A_pred, _ = model(coords_grid)
        F_dict = compute_field_strength(algebra, A_pred, coords_grid)

    # Predicted charge density
    F_dual = hodge_dual_4d({k: v.detach() for k, v in F_dict.items()})
    F_det = {k: v.detach() for k, v in F_dict.items()}
    pred_q = torch.zeros(grid_n * grid_n, device=device)
    for key in F_det:
        if key in F_dual:
            pred_q += hermitian_inner_product(
                algebra, F_det[key], F_dual[key]
            ).squeeze(-1)

    # Exact charge density (proportional to action density for self-dual fields)
    exact_q = bpst_action_density(coords_grid.detach(), rho)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pred_q_grid = pred_q.detach().cpu().reshape(grid_n, grid_n).numpy()
    exact_q_grid = exact_q.detach().cpu().reshape(grid_n, grid_n).numpy()

    vmax = max(np.abs(pred_q_grid).max(), np.abs(exact_q_grid).max())
    x1_np = x1.numpy()
    x2_np = x2.numpy()

    ax = axes[0]
    im = ax.pcolormesh(x1_np, x2_np, pred_q_grid.T, cmap='inferno',
                       vmin=0, vmax=vmax, shading='auto')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title('Predicted Q(x) density')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.pcolormesh(x1_np, x2_np, exact_q_grid.T, cmap='inferno',
                       vmin=0, vmax=vmax, shading='auto')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title(r'Exact $48\rho^4/(r^2+\rho^2)^4$')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    fig.suptitle('Topological Charge Density (x3=x4=0 slice)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'topo_charge_density.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def _plot_bivector_field(model, algebra, rho, output_dir, plt):
    """Bivector plane decomposition of F₀₁ on x₁-x₂ slice.

    Each F bivector is decomposed into su(2) components (e₁₂, e₁₃, e₂₃)
    and visualized as colored quiver arrows showing rotation plane orientation.
    """
    device = next(model.parameters()).device
    model.eval()

    grid_n = 16
    grid_range = 3.5
    x1 = torch.linspace(-grid_range, grid_range, grid_n)
    x2 = torch.linspace(-grid_range, grid_range, grid_n)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

    coords_grid = torch.zeros(grid_n * grid_n, 4, device=device)
    coords_grid[:, 0] = X1.reshape(-1).to(device)
    coords_grid[:, 1] = X2.reshape(-1).to(device)
    coords_grid = coords_grid.requires_grad_(True)

    with torch.enable_grad():
        A_pred, _ = model(coords_grid)
        F_dict = compute_field_strength(algebra, A_pred, coords_grid)

    # Exact field strength for comparison
    F_exact_dict = bpst_field_strength(coords_grid.detach(), rho,
                                       algebra_dim=algebra.dim)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, F_source, title in [
        (axes[0], {k: v.detach() for k, v in F_dict.items()}, 'Predicted F_01'),
        (axes[1], F_exact_dict, 'Exact F_01'),
    ]:
        F_01 = F_source.get((0, 1))
        if F_01 is None:
            ax.text(0.5, 0.5, 'No F_01 data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        F_01_cpu = F_01.cpu()

        # Extract su(2) bivector components
        e12 = F_01_cpu[:, 3].numpy()  # e₁₂
        e13 = F_01_cpu[:, 5].numpy()  # e₁₃
        e23 = F_01_cpu[:, 6].numpy()  # e₂₃

        x1_np = X1.reshape(-1).numpy()
        x2_np = X2.reshape(-1).numpy()

        # Magnitude for arrow scaling
        mag = np.sqrt(e12 ** 2 + e13 ** 2 + e23 ** 2)

        # Quiver: use e12, e13 as arrow direction (projection of rotation plane)
        scale = np.percentile(mag, 95) + 1e-12
        u = e12 / scale
        v = e13 / scale

        ax.quiver(x1_np, x2_np, u, v, mag, cmap='hot', alpha=0.8,
                  scale=grid_n * 0.5, width=0.004)
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        ax.set_title(f'{title}\nbivector components: '
                     r'$e_{12}$ (red), $e_{13}$ (green), $e_{23}$ (blue)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Bivector Field Decomposition (x3=x4=0 slice)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bivector_field.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def _plot_gauge_field(model, algebra, rho, output_dir, plt):
    """Gauge field A_μ structure on x₁-x₂ slice.

    Shows A₁ and A₂ components with su(2) color coding.
    """
    device = next(model.parameters()).device
    model.eval()

    grid_n = 16
    grid_range = 3.5
    x1 = torch.linspace(-grid_range, grid_range, grid_n)
    x2 = torch.linspace(-grid_range, grid_range, grid_n)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')

    coords_grid = torch.zeros(grid_n * grid_n, 4, device=device)
    coords_grid[:, 0] = X1.reshape(-1).to(device)
    coords_grid[:, 1] = X2.reshape(-1).to(device)

    with torch.no_grad():
        A_pred, _ = model(coords_grid)

    A_exact = bpst_gauge_potential(coords_grid.detach(), rho,
                                   algebra_dim=algebra.dim)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    x1_np = X1.reshape(-1).numpy()
    x2_np = X2.reshape(-1).numpy()

    for row, (A_src, src_label) in enumerate([
        (A_pred.detach().cpu(), 'Predicted'),
        (A_exact.cpu(), 'Exact'),
    ]):
        for col, mu in enumerate([0, 1]):
            ax = axes[row, col]
            A_mu_slice = A_src[:, mu]  # [N, D]

            # su(2) components
            e12 = A_mu_slice[:, 3].numpy()
            e13 = A_mu_slice[:, 5].numpy()
            e23 = A_mu_slice[:, 6].numpy()

            # Quiver: e12 as U, e13 as V, e23 as color
            mag = np.sqrt(e12**2 + e13**2 + e23**2)
            scale_val = np.percentile(mag, 95) + 1e-12

            q = ax.quiver(x1_np, x2_np, e12 / scale_val, e13 / scale_val,
                          e23, cmap='coolwarm', alpha=0.8,
                          scale=grid_n * 0.5, width=0.004)
            plt.colorbar(q, ax=ax, label=r'$e_{23}$ component')
            ax.set_xlabel('x1'); ax.set_ylabel('x2')
            ax.set_title(f'{src_label} A_{mu+1}\n'
                         r'arrows: ($e_{12}$, $e_{13}$), '
                         r'color: $e_{23}$')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)

    fig.suptitle('Gauge Field Structure (x3=x4=0 slice)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'gauge_field.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ============================================================================ #
# Self-Duality Component Visualization
# ============================================================================ #

def _plot_self_duality_components(F_dict, output_dir, plt):
    """Per-channel F vs *F comparison for each (mu,nu) pair."""
    os.makedirs(output_dir, exist_ok=True)
    F_dual = hodge_dual_4d(F_dict)

    n_pairs = sum(1 for k in F_dict if k in F_dual)
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    bv_labels = [r'$e_{12}$', r'$e_{13}$', r'$e_{23}$']
    idx = 0

    for key in sorted(F_dict.keys()):
        if key not in F_dual or idx >= 6:
            continue
        ax = axes_flat[idx]

        F_vals = F_dict[key].detach().cpu()
        Fd_vals = F_dual[key].detach().cpu()

        # Extract grade-2 components
        f_bv = torch.stack([F_vals[:, c].mean() for c in _BV_INDICES]).numpy()
        fd_bv = torch.stack([Fd_vals[:, c].mean() for c in _BV_INDICES]).numpy()

        x_pos = np.arange(3)
        width = 0.35
        ax.bar(x_pos - width / 2, f_bv, width, label='F', color='steelblue',
               alpha=0.8)
        ax.bar(x_pos + width / 2, fd_bv, width, label='*F', color='tomato',
               alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bv_labels)
        ax.set_title(f'F_{{{key[0]}{key[1]}}} vs *F_{{{key[0]}{key[1]}}}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')
        idx += 1

    fig.suptitle('Self-Duality: F vs *F per (mu,nu) pair\n'
                 '(should match for instanton)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'self_duality_components.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================ #
# Training
# ============================================================================ #

def train(args):
    """Main training loop for Yang-Mills instanton learning.

    Optimized pipeline:
    - Training: 12 autograd calls/step (Jacobian-cached field strength)
    - Diagnostics: YM equation, Bianchi, gauge covariance computed at intervals
    - Intrinsic symmetry: SU2BladeSelector (grade-2 su(2)), Hermitian IP (gauge)
    - Conformal decay: implicit via CGA rotor structure (no external envelope)
    - Topological charge tracked every diagnostic interval
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Algebra: CGA Cl(4,1), dim=32
    algebra = CliffordAlgebra(p=4, q=1, device=device)
    print(f"\n{'='*60}")
    print(f" Yang-Mills SU(2) Gauge Theory - CGA Cl(4,1)")
    print(f" SU(2) = Spin(3) embedded in conformal algebra (dim=32)")
    print(f" su(2) bivectors: e12(3), e13(5), e23(6) — same indices")
    print(f" Conformal bivectors: e14,e24,e34 (trans), e45 (dilation)")
    print(f" BPST instanton: rho={args.rho}, Q=1")
    print(f" Intrinsic symmetry: SU2BladeSelector + Hermitian IP (gauge)")
    print(f" Conformal decay: implicit (no external envelope)")
    print(f" Losses: supervised + self-duality + action + Sobolev(F) + Q(topo)")
    print(f" Autograd budget: 12 calls/step (Jacobian-cached, su(2) restricted)")
    print(f" Strict Orthogonality: {'ON' if args.strict_ortho else 'OFF'}"
          f"{f' (weight={args.ortho_weight}, mode={args.ortho_mode})' if args.strict_ortho else ''}")
    print(f"{'='*60}\n")

    # Orthogonality: even subalgebra {0, 2}
    ortho_settings = OrthogonalitySettings(
        enabled=args.strict_ortho,
        mode=args.ortho_mode,
        weight=args.ortho_weight,
        target_grades=[0, 2],  # even subalgebra for SU(2)
        tolerance=1e-3,
        monitor_interval=args.diag_interval,
        coupling_warn_threshold=0.3,
    )
    ortho = StrictOrthogonality(algebra, ortho_settings).to(device)

    # Metric and debugger
    ym_metric = GaugeYMMetric(algebra)
    debugger = YMInstantonDebugger(algebra, ym_metric, rho=args.rho)

    # Dataset
    print("Generating datasets...")
    train_ds = BPSTInstantonDataset(
        num_samples=args.num_train, rho=args.rho,
        sampling_radius=args.sampling_radius, seed=args.seed,
        algebra_dim=algebra.dim,
    )
    test_ds = BPSTInstantonDataset(
        num_samples=args.num_test, rho=args.rho,
        sampling_radius=args.sampling_radius, seed=args.seed + 1,
        algebra_dim=algebra.dim,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Network
    model = YangMillsNet(
        algebra,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_freqs=args.num_freqs,
        rho=args.rho,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nYangMillsNet [CGA]: {args.hidden_dim} hidden, {args.num_layers} layers, "
          f"{n_params:,} parameters, dim={algebra.dim}\n")

    # Optimizer
    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Training history
    history: Dict[str, list] = {
        'epochs': [], 'total': [], 'supervised': [], 'self_duality': [],
        'action': [], 'sobolev': [], 'q_loss': [], 'Q_estimates': [],
    }

    best_test_loss = float('inf')
    last_intermediates: List[torch.Tensor] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Short PDE warmup (20 epochs), then all losses at full strength
        pde_scale = min(epoch / 20.0, 1.0)
        sup_scale = 1.0

        epoch_losses = {k: 0.0 for k in ['total', 'supervised', 'self_duality',
                                           'action', 'sobolev', 'q_loss', 'ortho']}
        n_batches = 0

        need_intermediates = args.strict_ortho

        for coords, A_exact, action_exact in train_loader:
            coords = coords.to(device).requires_grad_(True)
            A_exact = A_exact.to(device)
            action_exact = action_exact.to(device)

            losses = compute_training_losses(
                algebra, model, coords, A_exact, action_exact,
                rho=args.rho,
                store_intermediates=need_intermediates,
            )

            # Orthogonality
            ortho_loss = torch.tensor(0.0, device=device)
            intermediates = losses['intermediates']
            if args.strict_ortho and intermediates:
                eff_weight = ortho.anneal_weight(epoch,
                                                  warmup_epochs=args.ortho_warmup,
                                                  total_epochs=args.epochs)
                for h in intermediates:
                    h_flat = h.reshape(-1, algebra.dim)
                    ortho_loss = ortho_loss + eff_weight * ortho.parasitic_energy(h_flat)
                ortho_loss = ortho_loss / len(intermediates)

            # Total loss (5 terms + ortho)
            total = (args.supervised_weight * sup_scale * losses['supervised'] +
                     args.sd_weight * pde_scale * losses['self_duality'] +
                     args.action_weight * pde_scale * losses['action'] +
                     args.sobolev_weight * pde_scale * losses['sobolev'] +
                     args.q_weight * pde_scale * losses['q_loss'] +
                     ortho_loss)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['total'] += total.item()
            epoch_losses['supervised'] += losses['supervised'].item()
            epoch_losses['self_duality'] += losses['self_duality'].item()
            epoch_losses['action'] += losses['action'].item()
            epoch_losses['sobolev'] += losses['sobolev'].item()
            epoch_losses['q_loss'] += losses['q_loss'].item()
            epoch_losses['ortho'] += ortho_loss.item()
            n_batches += 1

        scheduler.step()
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # Evaluate periodically
        if epoch % args.diag_interval == 0 or epoch == 1 or epoch == args.epochs:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg['total']:.6f} "
                  f"(sup:{avg['supervised']:.4f} "
                  f"sd:{avg['self_duality']:.4f} "
                  f"act:{avg['action']:.4f} "
                  f"sob:{avg['sobolev']:.4f} "
                  f"Q:{avg['q_loss']:.4f}) | "
                  f"LR: {lr:.6f}")

            # Diagnostics
            model.eval()
            test_coords = test_ds.coords.to(device)

            with torch.no_grad():
                report = debugger.format_report(model, test_coords)
                print(report)

                _, last_inters = model(test_coords, store_intermediates=True)
                last_intermediates = [h.cpu() for h in last_inters]

            # Topological charge tracking
            test_sample = test_ds.coords[:200].to(device).requires_grad_(True)
            with torch.enable_grad():
                A_test, _ = model(test_sample)
                F_test = compute_field_strength(algebra, A_test, test_sample)
            Q_est = ym_metric.topological_charge(
                {k: v.detach() for k, v in F_test.items()}
            ).item() / (8.0 * math.pi ** 2)
            print(f"  Q estimate: {Q_est:.4f} (target: 1.0)")
            history['Q_estimates'].append(Q_est)

            # Diagnostic losses (eval-only, expensive)
            if args.diagnostic_losses:
                diag_coords = test_ds.coords[:100].to(device).requires_grad_(True)
                with torch.enable_grad():
                    A_diag, _ = model(diag_coords)
                    F_diag = compute_field_strength(algebra, A_diag, diag_coords)
                    diag_losses = compute_diagnostic_losses(
                        algebra, A_diag, F_diag, diag_coords)
                print(f"  [Diagnostic] YM eqn: {diag_losses['ym_equation'].item():.6e}, "
                      f"Bianchi: {diag_losses['bianchi'].item():.6e}, "
                      f"Gauge: {diag_losses['gauge'].item():.6e}, "
                      f"Purity: {diag_losses['purity'].item():.6e}")

            if args.strict_ortho and last_intermediates:
                last_h = last_intermediates[-1].reshape(-1, algebra.dim)
                print(ortho.format_diagnostics(last_h))

            # Record history
            history['epochs'].append(epoch)
            for key in ['total', 'supervised', 'self_duality', 'action',
                         'sobolev', 'q_loss']:
                history[key].append(avg[key])

            if avg['total'] < best_test_loss:
                best_test_loss = avg['total']

    # ------------------------------------------------------------------ #
    # Final evaluation
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f" Final Evaluation")
    print(f"{'='*60}")

    model.eval()
    test_coords = test_ds.coords.to(device)

    with torch.no_grad():
        A_pred, final_inters = model(test_coords, store_intermediates=True)
        last_intermediates = [h.cpu() for h in final_inters]

        A_exact = bpst_gauge_potential(test_coords, args.rho,
                                       algebra_dim=algebra.dim)
        sup_err = nn.functional.mse_loss(A_pred, A_exact)
        print(f"  Supervised MSE(A):    {sup_err.item():.6e}")
        print(f"  Best training loss:   {best_test_loss:.6f}")

        print(debugger.format_report(model, test_coords))

    # Final Q estimate
    test_sample = test_ds.coords[:200].to(device).requires_grad_(True)
    with torch.enable_grad():
        A_test, _ = model(test_sample)
        F_test = compute_field_strength(algebra, A_test, test_sample)
    Q_final = ym_metric.topological_charge(
        {k: v.detach() for k, v in F_test.items()}
    ).item() / (8.0 * math.pi ** 2)
    print(f"  Final Q estimate: {Q_final:.4f} (target: 1.0)")

    # Final diagnostic losses
    if args.diagnostic_losses:
        diag_coords = test_ds.coords[:100].to(device).requires_grad_(True)
        with torch.enable_grad():
            A_diag, _ = model(diag_coords)
            F_diag = compute_field_strength(algebra, A_diag, diag_coords)
            diag_losses = compute_diagnostic_losses(
                algebra, A_diag, F_diag, diag_coords)
        print(f"  [Final Diagnostic] YM eqn: {diag_losses['ym_equation'].item():.6e}, "
              f"Bianchi: {diag_losses['bianchi'].item():.6e}, "
              f"Gauge: {diag_losses['gauge'].item():.6e}, "
              f"Purity: {diag_losses['purity'].item():.6e}")

    if args.strict_ortho and last_intermediates:
        last_h = last_intermediates[-1].reshape(-1, algebra.dim)
        print(ortho.format_diagnostics(last_h))

    # Self-duality component analysis
    if args.save_plots:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None

        if plt is not None:
            # Compute F for component analysis
            sd_coords = test_ds.coords[:200].to(device).requires_grad_(True)
            with torch.enable_grad():
                A_sd, _ = model(sd_coords)
                F_sd = compute_field_strength(algebra, A_sd, sd_coords)
            _plot_self_duality_components(
                {k: v.detach() for k, v in F_sd.items()},
                args.output_dir, plt)

    # Save plots
    if args.save_plots:
        print("\nGenerating plots...")
        _save_plots(
            history=history,
            model=model,
            debugger=debugger,
            algebra=algebra,
            ortho=ortho,
            last_intermediates=last_intermediates,
            test_coords=test_coords,
            rho=args.rho,
            output_dir=args.output_dir,
        )

    print(f"\n{'='*60}\n")
    return model, best_test_loss


# ============================================================================ #
# CLI
# ============================================================================ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Yang-Mills SU(2) Gauge Theory in CGA Cl(4,1) — Conformal Formulation')

    # Data
    p.add_argument('--rho', type=float, default=1.0,
                   help='Instanton size parameter')
    p.add_argument('--sampling-radius', type=float, default=5.0,
                   help='Maximum sampling radius around instanton core')
    p.add_argument('--num-train', type=int, default=3000)
    p.add_argument('--num-test', type=int, default=500)

    # Model
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--num-freqs', type=int, default=32)

    # Training
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')

    # Loss weights
    p.add_argument('--supervised-weight', type=float, default=1.0,
                   help='Supervised A matching weight')
    p.add_argument('--sd-weight', type=float, default=5.0,
                   help='Self-duality weight (core instanton constraint)')
    p.add_argument('--action-weight', type=float, default=1.0,
                   help='Action density weight')
    p.add_argument('--sobolev-weight', type=float, default=2.0,
                   help='Sobolev (field strength matching) weight')
    p.add_argument('--q-weight', type=float, default=1.0,
                   help='Topological charge loss weight')

    # Diagnostics
    p.add_argument('--diagnostic-losses', action='store_true',
                   help='Compute YM equation/Bianchi/gauge/purity at '
                        'diagnostic intervals (expensive, eval-only)')

    # Orthogonality
    p.add_argument('--strict-ortho', action='store_true',
                   help='Enable strict orthogonality enforcement')
    p.add_argument('--ortho-weight', type=float, default=0.1,
                   help='Orthogonality penalty weight')
    p.add_argument('--ortho-mode', choices=['loss', 'project'], default='loss',
                   help='Enforcement mode: soft loss or hard projection')
    p.add_argument('--ortho-warmup', type=int, default=20,
                   help='Epochs to linearly ramp orthogonality weight from 0')
    p.add_argument('--diag-interval', type=int, default=20,
                   help='Epochs between diagnostic reports')

    # Output / visualization
    p.add_argument('--save-plots', action='store_true',
                   help='Save diagnostic plots to --output-dir')
    p.add_argument('--output-dir', type=str, default='ym_plots',
                   help='Directory for saving plots')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
