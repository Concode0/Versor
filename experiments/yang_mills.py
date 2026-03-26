# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Yang-Mills SU(2) Gauge Theory in Cl(3,0).

Validates Geometric Algebra as a natural framework for non-abelian gauge theory.
SU(2) ≅ Spin(3) = unit even subalgebra of Cl(3,0).

The gauge field A_μ(x) is an su(2)-valued 1-form: at each spacetime point x ∈ ℝ⁴,
there are 4 bivectors A₁, A₂, A₃, A₄ (one per spacetime direction).

In Cl(3,0) with basis {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}:
  Grade 0 [idx 0]:       Scalar
  Grade 1 [idx 1,2,4]:   (unused for gauge fields)
  Grade 2 [idx 3,5,6]:   su(2) gauge potential / field strength (e₁₂, e₁₃, e₂₃)
  Grade 3 [idx 7]:       Pseudoscalar

Gauge transformations are rotors R ∈ Spin(3): A_μ → R A_μ R̃
The Hermitian inner product satisfies <RAR̃, RBR̃>_H = <A,B>_H (gauge invariance).

Test case: BPST instanton — exact SU(2) solution with topological charge Q=1.

Usage:
    uv run python -m experiments.yang_mills --epochs 300
    uv run python -m experiments.yang_mills --strict-ortho --rho 1.0 --save-plots
    uv run python -m experiments.yang_mills --save-plots --output-dir ym_plots
"""

from __future__ import annotations

import sys
import os
import argparse
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from core.metric import (
    hermitian_inner_product, hermitian_norm, hermitian_grade_spectrum,
)
from layers import CliffordLinear
from layers import RotorLayer
from layers import CliffordLayerNorm
from layers import BladeSelector
from functional.activation import GeometricGELU
from functional.loss import BivectorRegularization
from optimizers.riemannian import RiemannianAdam
from functional.orthogonality import StrictOrthogonality, OrthogonalitySettings


# ============================================================================ #
# Vectorised Jacobian helper
# ============================================================================ #

def _batch_jacobian(
    output: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Per-sample Jacobian via K VJPs.

    Args:
        output:  [B, K] — each row output[b,:] depends only on inputs[b,:].
        inputs:  [B, D] leaf tensor with requires_grad=True.
        create_graph: retain meta-graph for higher-order differentiation.
        retain_graph: keep the original graph between successive calls.

    Returns:
        jac: [B, K, D]  where jac[b,k,d] = ∂output[b,k] / ∂inputs[b,d].
    """
    B, K = output.shape
    D = inputs.shape[1]
    jac = output.new_zeros(B, K, D)
    for k in range(K):
        (g,) = torch.autograd.grad(
            output[:, k].sum(),
            inputs,
            create_graph=create_graph,
            retain_graph=retain_graph or (k < K - 1),
        )
        jac[:, k, :] = g
    return jac


# ============================================================================ #
# 't Hooft Symbols and BPST Instanton
# ============================================================================ #

def t_hooft_eta(a: int, mu: int, nu: int) -> float:
    """'t Hooft eta symbol η^a_μν (self-dual).

    Antisymmetric in μ,ν. Indices are 1-based: a ∈ {1,2,3}, μ,ν ∈ {1,2,3,4}.

    Rules:
      η^a_ij = ε_aij   for spatial i,j ∈ {1,2,3}
      η^a_i4 = δ_ai     for i ∈ {1,2,3}
      η^a_4i = -δ_ai    for i ∈ {1,2,3}
    """
    if mu == nu:
        return 0.0
    if mu > nu:
        return -t_hooft_eta(a, nu, mu)

    # mu < nu guaranteed below
    if mu <= 3 and nu <= 3:
        # Spatial: ε_aij (Levi-Civita)
        if (a, mu, nu) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
            return 1.0
        elif (a, mu, nu) in [(1, 3, 2), (2, 1, 3), (3, 2, 1)]:
            return -1.0
        return 0.0
    elif nu == 4:
        # η^a_i4 = δ_ai
        return 1.0 if a == mu else 0.0
    return 0.0


def bpst_gauge_potential(x: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
    """BPST instanton gauge potential A_μ^a(x) in regular gauge.

    A_μ^a(x) = η^a_μν x_ν / (|x|² + ρ²)

    Args:
        x: [B, 4] spacetime coordinates (x₁, x₂, x₃, x₄).
        rho: Instanton size parameter.

    Returns:
        A_mu: [B, 4, 8] — 4 gauge potential components as Cl(3,0) multivectors.
              Each is a grade-2 bivector in su(2).
    """
    B = x.shape[0]
    r2 = (x ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    denom = r2 + rho ** 2  # [B, 1]

    # su(2) → Cl(3,0) bivector index mapping:
    # a=1 → e₂₃ (index 6), a=2 → e₁₃ (index 5), a=3 → e₁₂ (index 3)
    bv_indices = {1: 6, 2: 5, 3: 3}

    A_mu = torch.zeros(B, 4, 8, dtype=x.dtype, device=x.device)

    for mu in range(4):  # 0-indexed spacetime direction
        for a in range(1, 4):  # su(2) color index
            coeff = 0.0
            # A_mu^a = Σ_ν η^a_{mu+1,ν} x_{ν-1} / denom
            acc = torch.zeros(B, dtype=x.dtype, device=x.device)
            for nu in range(1, 5):
                eta = t_hooft_eta(a, mu + 1, nu)
                if eta != 0.0:
                    acc = acc + eta * x[:, nu - 1]

            bv_idx = bv_indices[a]
            # Sign convention for a=2 (e₁₃): the standard mapping uses
            # a sign flip because e₁₃ has a different orientation than
            # the right-hand rule. We keep it consistent with the 't Hooft
            # symbol convention.
            sign = -1.0 if a == 2 else 1.0
            A_mu[:, mu, bv_idx] = sign * acc / denom.squeeze(-1)

    return A_mu


def bpst_field_strength(x: torch.Tensor, rho: float = 1.0) -> Dict[Tuple[int, int], torch.Tensor]:
    """BPST instanton field strength F_μν^a(x).

    F_μν^a(x) = 2ρ² η^a_μν / (|x|² + ρ²)²

    Args:
        x: [B, 4] spacetime coordinates.
        rho: Instanton size parameter.

    Returns:
        Dict mapping (mu, nu) pairs (0-indexed, mu < nu) to [B, 8] bivector multivectors.
    """
    B = x.shape[0]
    r2 = (x ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    denom = (r2 + rho ** 2) ** 2  # [B, 1]
    prefactor = 2.0 * rho ** 2 / denom  # [B, 1]

    bv_indices = {1: 6, 2: 5, 3: 3}

    F_dict = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F_mv = torch.zeros(B, 8, dtype=x.dtype, device=x.device)
            for a in range(1, 4):
                eta = t_hooft_eta(a, mu + 1, nu + 1)
                if eta != 0.0:
                    bv_idx = bv_indices[a]
                    sign = -1.0 if a == 2 else 1.0
                    F_mv[:, bv_idx] = sign * eta * prefactor.squeeze(-1)
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
# Dataset
# ============================================================================ #

class BPSTInstantonDataset(Dataset):
    """Dataset of BPST instanton samples.

    Points sampled from Gaussian distribution concentrated near
    instanton core (origin), with exact A_μ and F_μν as targets.

    Each sample: (coords [4], A_mu [4, 8], action_density [1]).
    """

    def __init__(self, num_samples: int, rho: float = 1.0,
                 sampling_radius: float = 5.0, seed: int = 42):
        rng = np.random.RandomState(seed)

        # Sample points concentrated near instanton core
        # Use Gaussian with std = rho to sampling_radius
        sigma = rng.uniform(rho, sampling_radius, num_samples).astype(np.float32)
        directions = rng.randn(num_samples, 4).astype(np.float32)
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = directions / (norms + 1e-8)
        radii = rng.exponential(scale=2.0 * rho, size=(num_samples, 1)).astype(np.float32)
        # Clip to sampling_radius
        radii = np.clip(radii, 0.1 * rho, sampling_radius)
        coords_np = directions * radii

        self.coords = torch.tensor(coords_np)  # [N, 4]

        # Compute exact gauge potential
        self.A_mu = bpst_gauge_potential(self.coords, rho)  # [N, 4, 8]

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

class YangMillsNet(nn.Module):
    """GA network for SU(2) gauge fields in Cl(3,0).

    Architecture:
        1. Fourier feature embedding of 4D spacetime coordinates
        2. Lift to multivector hidden space [B, C, 8]
        3. Pre-LN residual blocks: norm → rotor → act → linear + skip
        4. Output: CliffordLinear(C, 4) → [B, 4, 8] — 4 gauge potential channels
        5. BladeSelector enforces grade-2 purity (A_μ ∈ su(2))

    Returns A_mu [B, 4, 8] and intermediates.
    """

    def __init__(self, algebra, hidden_dim: int = 32, num_layers: int = 4,
                 num_freqs: int = 16):
        super().__init__()
        self.algebra = algebra
        self.hidden_dim = hidden_dim

        # Fourier embedding: learned frequencies for all 4 coords
        self.register_buffer('freq_bands', torch.randn(4, num_freqs) * 2.0)
        input_dim = 4 + 2 * num_freqs  # raw + sin + cos

        # Lift
        self.input_lift = nn.Linear(input_dim, hidden_dim * algebra.dim)
        self.input_norm = CliffordLayerNorm(algebra, hidden_dim)

        # Residual GA blocks (Pre-LN)
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

        # Grade-2 blade selector for output (enforce su(2))
        self.blade_selector = BladeSelector(algebra, channels=4)

        self._intermediates: List[torch.Tensor] = []

    def forward(self, coords: torch.Tensor):
        """Forward pass.

        Args:
            coords: [B, 4] spacetime coordinates.

        Returns:
            A_mu: [B, 4, 8] predicted gauge potential (4 bivector channels).
            intermediates: list of hidden states.
        """
        B = coords.shape[0]

        # Fourier features
        proj = coords @ self.freq_bands  # [B, num_freqs]
        features = torch.cat([coords, torch.sin(proj), torch.cos(proj)], dim=-1)

        # Lift
        h = self.input_lift(features)
        h = h.reshape(B, self.hidden_dim, self.algebra.dim)
        h = self.input_norm(h)

        # Residual blocks
        intermediates: List[torch.Tensor] = []
        self._intermediates = []
        for block in self.blocks:
            residual = h
            h = block['norm'](h)
            h = block['rotor'](h)
            h = block['act'](h)
            h = block['linear'](h)
            h = residual + h
            intermediates.append(h)
            self._intermediates.append(h.detach())

        # Output: [B, 4, 8]
        h = self.output_norm(h)
        A_mu = self.output_proj(h)  # [B, 4, 8]

        # Enforce grade-2 purity
        A_mu = self.blade_selector(A_mu)

        return A_mu, intermediates


# ============================================================================ #
# Field Strength Computation
# ============================================================================ #

def compute_field_strength(
    algebra,
    A_mu: torch.Tensor,
    coords: torch.Tensor,
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
    """Compute F_μν = ∂_μA_ν - ∂_νA_μ + [A_μ, A_ν] and the A-Jacobian.

    Uses a single batch of 32 VJPs (one per A_mu output component) instead of
    96 separate per-component grad calls, giving ~3× fewer autograd operations.

    Args:
        algebra: CliffordAlgebra(3, 0).
        A_mu: [B, 4, 8] gauge potential (must be in autograd graph).
        coords: [B, 4] with requires_grad=True.

    Returns:
        F_dict: (mu, nu) → [B, 8] field-strength tensors.
        jac_A:  [B, 4, 8, 4] — jac_A[:, chan, comp, x_dim].
    """
    B = A_mu.shape[0]

    # 32 VJPs to get the full Jacobian ∂A/∂x (was 96 per-component calls)
    A_flat = A_mu.reshape(B, 32)                                    # [B, 32]
    jac_flat = _batch_jacobian(A_flat, coords,
                               create_graph=True, retain_graph=True)  # [B, 32, 4]
    jac_A = jac_flat.reshape(B, 4, 8, 4)                           # [B, chan, comp, x_dim]

    F_dict: Dict[Tuple[int, int], torch.Tensor] = {}
    for mu in range(4):
        for nu in range(mu + 1, 4):
            dAnu_dxmu = jac_A[:, nu, :, mu]   # [B, 8]
            dAmu_dxnu = jac_A[:, mu, :, nu]   # [B, 8]
            AB = algebra.geometric_product(A_mu[:, mu], A_mu[:, nu])
            BA = algebra.geometric_product(A_mu[:, nu], A_mu[:, mu])
            F_dict[(mu, nu)] = (dAnu_dxmu - dAmu_dxnu) + (AB - BA)

    return F_dict, jac_A


def hodge_dual_4d(F_dict: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
    """Hodge dual of 2-form F in 4D Euclidean space.

    Maps:
      *F₀₁ = F₂₃,  *F₀₂ = -F₁₃,  *F₀₃ = F₁₂
      *F₁₂ = F₀₃,  *F₁₃ = -F₀₂,  *F₂₃ = F₀₁

    Note: Using 0-indexed (μ,ν) pairs.
    """
    dual = {}
    # *F_{01} = F_{23}
    if (2, 3) in F_dict:
        dual[(0, 1)] = F_dict[(2, 3)]
    # *F_{02} = -F_{13}
    if (1, 3) in F_dict:
        dual[(0, 2)] = -F_dict[(1, 3)]
    # *F_{03} = F_{12}
    if (1, 2) in F_dict:
        dual[(0, 3)] = F_dict[(1, 2)]
    # *F_{12} = F_{03}
    if (0, 3) in F_dict:
        dual[(1, 2)] = F_dict[(0, 3)]
    # *F_{13} = -F_{02}
    if (0, 2) in F_dict:
        dual[(1, 3)] = -F_dict[(0, 2)]
    # *F_{23} = F_{01}
    if (0, 1) in F_dict:
        dual[(2, 3)] = F_dict[(0, 1)]

    return dual


# ============================================================================ #
# Yang-Mills Losses
# ============================================================================ #

def compute_ym_losses(algebra, model: YangMillsNet, coords: torch.Tensor,
                      A_exact: torch.Tensor, action_exact: torch.Tensor,
                      rho: float) -> Dict[str, torch.Tensor]:
    """Compute all Yang-Mills loss terms.

    Optimised autograd schedule:
      • Field strength : 32 VJPs  (was 96  per-component calls)
      • F Jacobians    : 48 VJPs  (was 224 per-component calls for YM eq + Bianchi)
      Total            : 80 VJPs  (was 320+)

    Args:
        algebra: CliffordAlgebra(3, 0).
        model: YangMillsNet.
        coords: [B, 4] with requires_grad=True.
        A_exact: [B, 4, 8] exact gauge potential.
        action_exact: [B] exact action density.
        rho: Instanton size.

    Returns:
        Dict of loss terms.
    """
    A_pred, intermediates = model(coords)

    # --- Supervised loss on A ---
    supervised_loss = nn.functional.mse_loss(A_pred, A_exact)

    # --- Field strength (32 VJPs via _batch_jacobian) ---
    F_dict, _ = compute_field_strength(algebra, A_pred, coords)

    # --- Self-duality: F = *F ---
    F_dual = hodge_dual_4d(F_dict)
    sd_loss = torch.tensor(0.0, device=coords.device)
    n_pairs = 0
    for key in F_dict:
        if key in F_dual:
            sd_loss = sd_loss + ((F_dict[key] - F_dual[key]) ** 2).mean()
            n_pairs += 1
    if n_pairs > 0:
        sd_loss = sd_loss / n_pairs

    # --- Action density ---
    action_pred = torch.zeros(coords.shape[0], device=coords.device)
    for key, F in F_dict.items():
        action_pred = action_pred + hermitian_inner_product(algebra, F, F).squeeze(-1)
    action_loss = nn.functional.mse_loss(action_pred, action_exact)

    # --- Grade purity (A should be pure grade-2) ---
    purity_loss = torch.tensor(0.0, device=coords.device)
    for mu in range(4):
        A_mu_proj = algebra.grade_projection(A_pred[:, mu], 2)
        residual = A_pred[:, mu] - A_mu_proj
        purity_loss = purity_loss + (residual ** 2).mean()
    purity_loss = purity_loss / 4.0

    # --- Gauge covariance ---
    gauge_loss = _gauge_covariance_loss(algebra, F_dict)

    # --- Precompute F Jacobians once: 48 VJPs for all 6 pairs ---
    # jac_F[(mu,nu)][b, comp, d] = ∂F_{μν}[b,comp] / ∂coords[b,d]
    # retain_graph=True on all calls: the same forward graph (A_pred) is also
    # needed for supervised_loss and commutator terms in loss.backward().
    # The graph is freed by loss.backward() at the end of the training step.
    pairs = list(F_dict.keys())  # 6 pairs
    jac_F: Dict[Tuple[int, int], torch.Tensor] = {}
    for key in pairs:
        jac_F[key] = _batch_jacobian(
            F_dict[key], coords,
            create_graph=True,
            retain_graph=True,
        )  # [B, 8, 4]

    _zeros8 = torch.zeros(coords.shape[0], 8, device=coords.device)

    # --- YM equation: D_μ F^μν = ∂_μ F^μν + [A_μ, F^μν] = 0 ---
    ym_loss = torch.tensor(0.0, device=coords.device)
    for nu_target in range(4):
        residual_nu = torch.zeros_like(_zeros8)
        for mu in range(4):
            if mu == nu_target:
                continue
            if mu < nu_target:
                key = (mu, nu_target)
                sign = 1.0
            else:
                key = (nu_target, mu)
                sign = -1.0
            F_mn = F_dict.get(key, _zeros8)
            jac = jac_F.get(key)
            # ∂_μ F^μν: column mu of the Jacobian
            dF_dxmu = sign * jac[:, :, mu] if jac is not None else _zeros8
            residual_nu = residual_nu + dF_dxmu
            comm = (algebra.geometric_product(A_pred[:, mu], sign * F_mn) -
                    algebra.geometric_product(sign * F_mn, A_pred[:, mu]))
            residual_nu = residual_nu + comm
        ym_loss = ym_loss + (residual_nu ** 2).mean()
    ym_loss = ym_loss / 4.0

    # --- Bianchi identity: D_{[μ}F_{νρ]} = 0 ---
    bianchi_loss = torch.tensor(0.0, device=coords.device)
    triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for mu, nu, rho_idx in triples:
        F_nu_rho = F_dict.get((nu, rho_idx), _zeros8)
        j_nu_rho = jac_F.get((nu, rho_idx))

        key_rmu = (min(rho_idx, mu), max(rho_idx, mu))
        sign_rmu = 1.0 if rho_idx < mu else -1.0
        F_rho_mu_raw = F_dict.get(key_rmu, _zeros8)
        j_rho_mu_raw = jac_F.get(key_rmu)
        F_rho_mu = sign_rmu * F_rho_mu_raw

        F_mu_nu = F_dict.get((mu, nu), _zeros8)
        j_mu_nu = jac_F.get((mu, nu))

        # ∂_μ F_{νρ}, ∂_ν F_{ρμ}, ∂_ρ F_{μν} — reuse precomputed Jacobians
        dF_nu_rho = j_nu_rho[:, :, mu] if j_nu_rho is not None else _zeros8
        dF_rho_mu = (sign_rmu * j_rho_mu_raw[:, :, nu]
                     if j_rho_mu_raw is not None else _zeros8)
        dF_mu_nu  = j_mu_nu[:, :, rho_idx] if j_mu_nu is not None else _zeros8

        comm1 = (algebra.geometric_product(A_pred[:, mu], F_nu_rho) -
                 algebra.geometric_product(F_nu_rho, A_pred[:, mu]))
        comm2 = (algebra.geometric_product(A_pred[:, nu], F_rho_mu) -
                 algebra.geometric_product(F_rho_mu, A_pred[:, nu]))
        comm3 = (algebra.geometric_product(A_pred[:, rho_idx], F_mu_nu) -
                 algebra.geometric_product(F_mu_nu, A_pred[:, rho_idx]))

        bianchi_residual = (dF_nu_rho + comm1) + (dF_rho_mu + comm2) + (dF_mu_nu + comm3)
        bianchi_loss = bianchi_loss + (bianchi_residual ** 2).mean()
    bianchi_loss = bianchi_loss / len(triples)

    return {
        'supervised': supervised_loss,
        'self_duality': sd_loss,
        'action': action_loss,
        'ym_equation': ym_loss,
        'bianchi': bianchi_loss,
        'gauge': gauge_loss,
        'purity': purity_loss,
        'A_pred': A_pred,
        'F_dict': F_dict,
        'intermediates': intermediates,
    }


def _gauge_covariance_loss(algebra, F_dict: Dict[Tuple[int, int], torch.Tensor]) -> torch.Tensor:
    """Test gauge covariance: <RFR̃, RFR̃>_H = <F, F>_H for random R."""
    # Random bivector for rotor
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
    for key, F in F_dict.items():
        B = F.shape[0]
        # Sandwich product
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
    """Gauge-theoretic metrics for Yang-Mills in Cl(3,0)."""

    def __init__(self, algebra):
        self.algebra = algebra

    def action_density(self, F_dict: Dict[Tuple[int, int], torch.Tensor]) -> torch.Tensor:
        """Σ_{μ<ν} <F_μν, F_μν>_H per point."""
        result = None
        for key, F in F_dict.items():
            contrib = hermitian_inner_product(self.algebra, F, F).squeeze(-1)
            result = contrib if result is None else result + contrib
        return result

    def topological_charge(self, F_dict: Dict[Tuple[int, int], torch.Tensor]) -> torch.Tensor:
        """Topological charge density: Σ <F_μν, *F_μν>_H.

        For instanton: integrated over all space should give 8π².
        """
        F_dual = hodge_dual_4d(F_dict)
        result = torch.tensor(0.0, device=next(iter(F_dict.values())).device)
        for key in F_dict:
            if key in F_dual:
                contrib = hermitian_inner_product(self.algebra, F_dict[key],
                                                  F_dual[key]).squeeze(-1)
                result = result + contrib.mean()
        return result

    def self_duality_error(self, F_dict: Dict[Tuple[int, int], torch.Tensor]) -> float:
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

    def gauge_covariance_error(self, F_dict: Dict[Tuple[int, int], torch.Tensor]) -> float:
        """Random rotor test for gauge covariance."""
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
            f"  Topological charge density:  {topo.item():.6f}  (target: 8π²≈{8*math.pi**2:.4f})",
            f"  Self-duality error |F-*F|²:  {sd_err:.6e}",
            f"  Gauge covariance error:      {gauge_err:.6e}",
            f"  Grade-2 purity:              {purity:.4%}",
        ]
        return '\n'.join(lines)


# ============================================================================ #
# Instanton Debugger
# ============================================================================ #

class YMInstantonDebugger:
    """Physics diagnostics for BPST instanton in Cl(3,0)."""

    def __init__(self, algebra, metric: GaugeYMMetric, rho: float = 1.0):
        self.algebra = algebra
        self.metric = metric
        self.rho = rho

    def action_density_profile(self, model: YangMillsNet,
                               r_grid: torch.Tensor) -> Dict[str, np.ndarray]:
        """Radial profile of action density: predicted vs exact."""
        model.eval()
        pred_action = []
        exact_action = []

        for r_val in r_grid:
            # Sample points on sphere of radius r
            coords = torch.zeros(1, 4, device=r_grid.device, requires_grad=True)
            with torch.no_grad():
                coords.data[0, 0] = r_val.item()  # point along x₁ axis

            with torch.enable_grad():
                A_mu, _ = model(coords)
                F_dict = compute_field_strength(self.algebra, A_mu, coords)

            # Exact action density at this radius
            r2 = r_val.item() ** 2
            exact_a = 48.0 * self.rho ** 4 / (r2 + self.rho ** 2) ** 4
            exact_action.append(exact_a)

            # Predicted action: sum of <F_μν, F_μν>_H
            a_energy = 0.0
            for key, F in F_dict.items():
                a_energy += hermitian_inner_product(
                    self.algebra, F.detach(), F.detach()
                ).squeeze(-1).item()
            pred_action.append(a_energy)

        return {
            'r': r_grid.numpy(),
            'pred_action': np.array(pred_action),
            'exact_action': np.array(exact_action),
        }

    @torch.no_grad()
    def topological_charge_integration(self, model: YangMillsNet,
                                       R_max: float = 5.0,
                                       n_radii: int = 10) -> Dict[str, np.ndarray]:
        """Integrate topological charge as function of integration radius."""
        model.eval()
        radii = np.linspace(0.5, R_max, n_radii)
        Q_values = []

        for R in radii:
            # Sample points within ball of radius R
            rng = np.random.RandomState(123)
            pts = rng.randn(100, 4).astype(np.float32)
            norms = np.linalg.norm(pts, axis=-1, keepdims=True)
            pts = pts / (norms + 1e-8) * rng.uniform(0, R, (100, 1)).astype(np.float32)

            coords = torch.tensor(pts, device=next(model.parameters()).device)
            coords.requires_grad_(True)

            A_mu, _ = model(coords)

            # Exact topological charge at these points
            action_exact = bpst_action_density(coords.detach(), self.rho)
            # Q ∝ mean action × volume
            Q_approx = action_exact.mean().item() * (R ** 4) * (math.pi ** 2 / 2.0)
            # Normalize by 8π²
            Q_values.append(Q_approx / (8.0 * math.pi ** 2))

        return {
            'R': np.array(radii),
            'Q': np.array(Q_values),
        }

    @torch.no_grad()
    def self_duality_check(self, model: YangMillsNet,
                           coords: torch.Tensor) -> Dict[str, float]:
        """Check F = *F for predicted fields."""
        model.eval()
        coords = coords.clone().requires_grad_(True)
        A_mu, _ = model(coords)

        F_dict = compute_field_strength(self.algebra, A_mu, coords)
        sd_err = self.metric.self_duality_error(
            {k: v.detach() for k, v in F_dict.items()}
        )
        return {'self_duality_error': sd_err}

    @torch.no_grad()
    def gauge_covariance_test(self, model: YangMillsNet,
                              coords: torch.Tensor) -> float:
        """Apply random rotors, check gauge invariance."""
        model.eval()
        coords = coords.clone().requires_grad_(True)
        A_mu, _ = model(coords)

        F_dict = compute_field_strength(self.algebra, A_mu, coords)
        return self.metric.gauge_covariance_error(
            {k: v.detach() for k, v in F_dict.items()}
        )

    @torch.no_grad()
    def field_strength_spectrum(self, model: YangMillsNet,
                                coords: torch.Tensor) -> Dict[str, float]:
        """Grade spectrum of F (should be pure grade-2)."""
        model.eval()
        coords = coords.clone().requires_grad_(True)
        A_mu, _ = model(coords)

        F_dict = compute_field_strength(self.algebra, A_mu, coords)

        # Average spectrum over all F components
        total_spectrum = None
        n = 0
        for key, F in F_dict.items():
            spec = hermitian_grade_spectrum(self.algebra, F.detach())
            if total_spectrum is None:
                total_spectrum = spec.mean(dim=0)
            else:
                total_spectrum = total_spectrum + spec.mean(dim=0)
            n += 1

        if total_spectrum is not None:
            total_spectrum = total_spectrum / n
            labels = ['G0', 'G1', 'G2', 'G3']
            return {labels[k]: total_spectrum[k].item() for k in range(4)}
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
        lines.append(f"  A_μ grade-2 purity:         {purity:.4%}")

        # Supervised error
        A_exact = bpst_gauge_potential(coords.detach(), self.rho)
        sup_err = nn.functional.mse_loss(A_mu.detach(), A_exact).item()
        lines.append(f"  Supervised MSE(A):          {sup_err:.6e}")

        # Action density comparison at origin
        r_grid = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0])
        profile = self.action_density_profile(model, r_grid)
        lines.append("  Action density profile (|A|² proxy):")
        for r, pred, exact in zip(profile['r'], profile['pred_action'], profile['exact_action']):
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
    """Save diagnostic plots.

    Plots:
        1. convergence.png       - Training loss curves
        2. action_density.png    - Radial profile: predicted vs exact
        3. grade_spectrum.png    - Energy per grade for A and F
        4. self_duality.png      - F vs *F scatter + error histogram
        5. topological_charge.png - Q(R) vs integration radius
        6. coupling_heatmap.png  - Cross-grade coupling
    """
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
        ('total', 'Total', 'black'),
    ]:
        if history.get(key):
            ax.semilogy(epochs, history[key], label=label, color=color)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.set_title('Training Convergence'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key, label, color in [
        ('self_duality', 'Self-duality', 'purple'),
        ('gauge', 'Gauge covariance', 'orange'),
        ('purity', 'Grade purity', 'teal'),
        ('ym_equation', 'YM equation', 'brown'),
    ]:
        if history.get(key):
            ax.semilogy(epochs, history[key], label=label, color=color)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
    ax.set_title('Physics Losses'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Yang-Mills SU(2) Gauge Theory - Cl(3,0)', fontsize=13)
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
            label='Exact: 48ρ⁴/(r²+ρ²)⁴')
    ax.plot(profile['r'], profile['pred_action'], 'r--o', markersize=3,
            label='Predicted (|A|² proxy)')
    ax.set_xlabel('r = |x|')
    ax.set_ylabel('Action density')
    ax.set_title(f'Action Density Profile (ρ={rho:.1f})')
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # A spectrum
    a_spec_total = torch.zeros(4)
    for mu in range(4):
        spec = hermitian_grade_spectrum(algebra, A_mu[:, mu].detach()).mean(dim=0).cpu()
        a_spec_total = a_spec_total + spec
    a_spec_total = a_spec_total / 4.0

    ax = axes[0]
    labels = ['G0\n(scalar)', 'G1\n(vector)', 'G2\n(bivector)', 'G3\n(pseudo)']
    colors = ['royalblue', 'gray', 'mediumseagreen', 'gray']
    bars = ax.bar(range(4), a_spec_total.numpy(), color=colors, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, a_spec_total.numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel('Hermitian Energy'); ax.set_title('A_μ Grade Spectrum\n(should be G2-dominated)')
    ax.grid(True, alpha=0.2, axis='y')

    # Exact A spectrum for comparison
    A_exact = bpst_gauge_potential(test_coords.detach(), rho)
    exact_spec_total = torch.zeros(4)
    for mu in range(4):
        spec = hermitian_grade_spectrum(algebra, A_exact[:, mu]).mean(dim=0).cpu()
        exact_spec_total = exact_spec_total + spec
    exact_spec_total = exact_spec_total / 4.0

    ax = axes[1]
    bars = ax.bar(range(4), exact_spec_total.numpy(), color=colors, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, exact_spec_total.numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels)
    ax.set_ylabel('Hermitian Energy'); ax.set_title('A_μ Exact Grade Spectrum')
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'grade_spectrum.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. Self-duality: predicted A vs exact A scatter
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pred_flat = A_mu.detach().reshape(-1).cpu().numpy()
    exact_flat = A_exact.reshape(-1).cpu().numpy()

    # Subsample for scatter
    n_pts = min(5000, len(pred_flat))
    idx = np.random.choice(len(pred_flat), n_pts, replace=False)

    ax = axes[0]
    ax.scatter(exact_flat[idx], pred_flat[idx], alpha=0.3, s=5, color='steelblue')
    lim = max(abs(exact_flat[idx]).max(), abs(pred_flat[idx]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Exact A'); ax.set_ylabel('Predicted A')
    ax.set_title('A_μ: Predicted vs Exact'); ax.legend(); ax.grid(True, alpha=0.3)

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
            import matplotlib.pyplot as _plt
            _plt.close(fig_coup)

    print(f"  Plots saved to {output_dir}/")


# ============================================================================ #
# Training
# ============================================================================ #

def train(args):
    """Main training loop for Yang-Mills instanton learning."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Algebra: Cl(3,0), dim=8
    algebra = CliffordAlgebra(p=3, q=0, device=device)
    print(f"\n{'='*60}")
    print(f" Yang-Mills SU(2) Gauge Theory - Cl(3,0)")
    print(f" SU(2) ≅ Spin(3) = even subalgebra {{1, e₁₂, e₁₃, e₂₃}}")
    print(f" BPST instanton: ρ={args.rho}, Q=1")
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

    # Bivector regularization
    bv_reg = BivectorRegularization(algebra, grade=2).to(device)

    # Dataset
    print("Generating datasets...")
    train_ds = BPSTInstantonDataset(
        num_samples=args.num_train, rho=args.rho,
        sampling_radius=args.sampling_radius, seed=args.seed,
    )
    test_ds = BPSTInstantonDataset(
        num_samples=args.num_test, rho=args.rho,
        sampling_radius=args.sampling_radius, seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Network
    model = YangMillsNet(
        algebra,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_freqs=args.num_freqs,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nYangMillsNet: {args.hidden_dim} hidden, {args.num_layers} layers, "
          f"{n_params:,} parameters\n")

    # Optimizer
    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history
    history: Dict[str, list] = {
        'epochs': [], 'total': [], 'supervised': [], 'self_duality': [],
        'action': [], 'ym_equation': [], 'bianchi': [],
        'gauge': [], 'purity': [],
    }

    best_test_loss = float('inf')
    last_intermediates: List[torch.Tensor] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Curriculum: start supervised-heavy, shift to PDE losses
        sup_scale = max(1.0 - epoch / (args.epochs * 0.5), 0.2)
        pde_scale = min(epoch / (args.epochs * 0.3), 1.0)

        epoch_losses = {k: 0.0 for k in ['total', 'supervised', 'self_duality',
                                           'action', 'ym_equation', 'bianchi',
                                           'gauge', 'purity', 'ortho']}
        n_batches = 0

        for coords, A_exact, action_exact in train_loader:
            coords = coords.to(device).requires_grad_(True)
            A_exact = A_exact.to(device)
            action_exact = action_exact.to(device)

            losses = compute_ym_losses(algebra, model, coords,
                                       A_exact, action_exact, args.rho)

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

            # Total loss with curriculum weighting
            total = (args.supervised_weight * sup_scale * losses['supervised'] +
                     args.sd_weight * pde_scale * losses['self_duality'] +
                     args.action_weight * pde_scale * losses['action'] +
                     args.ym_weight * pde_scale * losses['ym_equation'] +
                     args.bianchi_weight * pde_scale * losses['bianchi'] +
                     args.gauge_weight * losses['gauge'] +
                     args.purity_weight * losses['purity'] +
                     ortho_loss)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['total'] += total.item()
            epoch_losses['supervised'] += losses['supervised'].item()
            epoch_losses['self_duality'] += losses['self_duality'].item()
            epoch_losses['action'] += losses['action'].item()
            epoch_losses['ym_equation'] += losses['ym_equation'].item()
            epoch_losses['bianchi'] += losses['bianchi'].item()
            epoch_losses['gauge'] += losses['gauge'].item()
            epoch_losses['purity'] += losses['purity'].item()
            epoch_losses['ortho'] += ortho_loss.item()
            n_batches += 1

        scheduler.step()
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # Evaluate periodically
        if epoch % args.diag_interval == 0 or epoch == 1 or epoch == args.epochs:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg['total']:.6f} "
                  f"(sup:{avg['supervised']:.4f} sd:{avg['self_duality']:.4f} "
                  f"act:{avg['action']:.4f} ym:{avg['ym_equation']:.4f} "
                  f"gauge:{avg['gauge']:.4f} pur:{avg['purity']:.4f}) | "
                  f"LR: {lr:.6f}")

            # Diagnostics
            model.eval()
            with torch.no_grad():
                test_coords = test_ds.coords.to(device)
                report = debugger.format_report(model, test_coords)
                print(report)

                _, last_inters = model(test_coords)
                last_intermediates = [h.cpu() for h in last_inters]

            if args.strict_ortho and last_intermediates:
                last_h = last_intermediates[-1].reshape(-1, algebra.dim)
                print(ortho.format_diagnostics(last_h))

            # Record history
            history['epochs'].append(epoch)
            for key in ['total', 'supervised', 'self_duality', 'action',
                        'ym_equation', 'bianchi', 'gauge', 'purity']:
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
        A_pred, final_inters = model(test_coords)
        last_intermediates = [h.cpu() for h in final_inters]

        A_exact = bpst_gauge_potential(test_coords, args.rho)
        sup_err = nn.functional.mse_loss(A_pred, A_exact)
        print(f"  Supervised MSE(A):    {sup_err.item():.6e}")
        print(f"  Best training loss:   {best_test_loss:.6f}")

        print(debugger.format_report(model, test_coords))

    if args.strict_ortho and last_intermediates:
        last_h = last_intermediates[-1].reshape(-1, algebra.dim)
        print(ortho.format_diagnostics(last_h))

    # Save plots
    if args.save_plots:
        print("\nGenerating plots...")
        with torch.no_grad():
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
        description='Yang-Mills SU(2) Gauge Theory in Cl(3,0)')

    # Data
    p.add_argument('--rho', type=float, default=1.0,
                   help='Instanton size parameter')
    p.add_argument('--sampling-radius', type=float, default=5.0,
                   help='Maximum sampling radius around instanton core')
    p.add_argument('--num-train', type=int, default=2000)
    p.add_argument('--num-test', type=int, default=300)

    # Model
    p.add_argument('--hidden-dim', type=int, default=32)
    p.add_argument('--num-layers', type=int, default=4)
    p.add_argument('--num-freqs', type=int, default=16)

    # Training
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cpu')

    # Loss weights
    p.add_argument('--supervised-weight', type=float, default=1.0,
                   help='Supervised A matching weight')
    p.add_argument('--sd-weight', type=float, default=5.0,
                   help='Self-duality weight')
    p.add_argument('--action-weight', type=float, default=1.0,
                   help='Action density weight')
    p.add_argument('--ym-weight', type=float, default=1.0,
                   help='YM equation residual weight')
    p.add_argument('--bianchi-weight', type=float, default=0.1,
                   help='Bianchi identity weight')
    p.add_argument('--gauge-weight', type=float, default=0.1,
                   help='Gauge covariance weight')
    p.add_argument('--purity-weight', type=float, default=1.0,
                   help='Grade-2 purity weight')

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
