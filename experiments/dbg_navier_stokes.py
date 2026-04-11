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

Navier-Stokes as Gauge Theory in Cl(3,0) — Continuum Multivector Formulation.

Reinterprets fluid dynamics through the lens of gauge theory, treating the
entire fluid state Ψ = p + u + ω + h as a single continuum multivector in
Cl(3,0). PDE constraints emerge from the vector derivative ∇Ψ and grade
projections, rather than component-wise scalar differentiation:

  ∇Ψ = GP(e₁, ∂Ψ/∂x) + GP(e₂, ∂Ψ/∂y) + GP(e₃, ∂Ψ/∂z)

  ⟨∇u⟩₀ = div(u) = 0          — incompressibility (grade-0 projection)
  ⟨∇u⟩₂ = curl(u) = ω         — vorticity consistency (grade-2 projection)
  ∂ₜu + (u·∇)u + ∇p − ν∇²u   — momentum equation (grade-1 residual)

Gauge-theoretic interpretation:
  - Velocity field u is a gauge connection (grade-1 vector)
  - Vorticity ω = ∇×u is the gauge curvature (grade-2 bivector)
  - Incompressibility ∇·u = 0 is gauge fixing
  - Pressure p is a scalar potential (grade-0)
  - Helicity h = u·ω is a topological invariant (grade-3 pseudoscalar)

In Cl(3,0) with basis {1, e₁, e₂, e₃, e₁₂, e₁₃, e₂₃, e₁₂₃}:
  Grade 0 [idx 0]:       Pressure (scalar)
  Grade 1 [idx 1,2,4]:   Velocity (e₁, e₂, e₃)
  Grade 2 [idx 3,5,6]:   Vorticity (e₁₂, e₁₃, e₂₃)
  Grade 3 [idx 7]:       Helicity density (e₁₂₃)

The gauge metric is the Hermitian inner product <·,·>_H from core/metric.py.
Key property: <RuR̃, RuR̃>_H = <u, u>_H (rotor isometry → gauge invariance).

Test case: Taylor-Green vortex with viscous decay (analytical solution).

Usage:
    uv run python -m experiments.navier_stokes --epochs 300
    uv run python -m experiments.navier_stokes --strict-ortho --re-max 10000 --save-plots
    uv run python -m experiments.navier_stokes --re-min 1000 --re-max 100000 --epochs 500
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
from optimizers.riemannian import RiemannianAdam
from functional.orthogonality import StrictOrthogonality, OrthogonalitySettings


# ============================================================================ #
# Taylor-Green Vortex — Analytical Solution
# ============================================================================ #

def taylor_green_velocity(x: torch.Tensor, y: torch.Tensor,
                          z: torch.Tensor, t: torch.Tensor,
                          nu: float, A: float = 1.0) -> Tuple[torch.Tensor, ...]:
    """Analytical Taylor-Green vortex velocity with viscous decay.

    u₁ =  A sin(x)cos(y)cos(z) exp(-3νt)
    u₂ = -A cos(x)sin(y)cos(z) exp(-3νt)
    u₃ =  0

    Args:
        x, y, z, t: Coordinate tensors (same shape).
        nu: Kinematic viscosity (1/Re).
        A: Amplitude.

    Returns:
        (u1, u2, u3) velocity components.
    """
    decay = torch.exp(-3.0 * nu * t)
    u1 = A * torch.sin(x) * torch.cos(y) * torch.cos(z) * decay
    u2 = -A * torch.cos(x) * torch.sin(y) * torch.cos(z) * decay
    u3 = torch.zeros_like(x)
    return u1, u2, u3


def taylor_green_pressure(x: torch.Tensor, y: torch.Tensor,
                          z: torch.Tensor, t: torch.Tensor,
                          nu: float, A: float = 1.0) -> torch.Tensor:
    """Analytical Taylor-Green pressure.

    p = (A²/16)(cos(2x) + cos(2y))(cos(2z) + 2) exp(-6νt)
    """
    decay = torch.exp(-6.0 * nu * t)
    p = (A ** 2 / 16.0) * (torch.cos(2 * x) + torch.cos(2 * y)) * \
        (torch.cos(2 * z) + 2.0) * decay
    return p


def taylor_green_vorticity(x: torch.Tensor, y: torch.Tensor,
                           z: torch.Tensor, t: torch.Tensor,
                           nu: float, A: float = 1.0) -> Tuple[torch.Tensor, ...]:
    """Analytical Taylor-Green vorticity ω = ∇×u.

    ω₁ = ∂u₃/∂y - ∂u₂/∂z =  A cos(x)sin(y)sin(z) exp(-3νt)
    ω₂ = ∂u₁/∂z - ∂u₃/∂x = -A sin(x)cos(y)sin(z) exp(-3νt)
    ω₃ = ∂u₂/∂x - ∂u₁/∂y =  0

    Note: ω₃ = ∂u₂/∂x - ∂u₁/∂y = A sin(x)sin(y)cos(z) - (-A)sin(x)sin(y)cos(z)
        Wait, let me recalculate:
        ∂u₂/∂x = A sin(x)sin(y)cos(z) exp(-3νt)
        ∂u₁/∂y = -A sin(x)sin(y)cos(z) exp(-3νt)
        ω₃ = ∂u₂/∂x - ∂u₁/∂y = 2A sin(x)sin(y)cos(z) exp(-3νt)
    Actually for the standard TGV the z-component of vorticity is nonzero. Let's compute correctly:
        ∂u₁/∂y = -A sin(x)sin(y)cos(z) exp
        ∂u₁/∂z = -A sin(x)cos(y)sin(z) exp
        ∂u₂/∂x = A sin(x)sin(y)cos(z) exp
        ∂u₂/∂z = A cos(x)sin(y)sin(z) exp
        ∂u₃/∂x = 0, ∂u₃/∂y = 0

    ω₁ = ∂u₃/∂y - ∂u₂/∂z = -A cos(x)sin(y)sin(z) exp(-3νt)  [wait, sign]
    Actually: ω₁ = ∂u₃/∂y - ∂u₂/∂z = 0 - A cos(x)sin(y)sin(z) exp(-3νt)
                 = A cos(x)sin(y)sin(z) exp(-3νt)
    ∂u₂/∂z = -A cos(x)sin(y)·(-sin(z))·exp = A cos(x)sin(y)sin(z) exp  ... hmm no.
    ∂u₂/∂z = -A cos(x)sin(y)·(∂cos(z)/∂z)·exp = -A cos(x)sin(y)·(-sin(z))·exp = A cos(x)sin(y)sin(z) exp
    So ω₁ = 0 - A cos(x)sin(y)sin(z) exp = -A cos(x)sin(y)sin(z) exp

    Let me just return the standard result:
    """
    decay = torch.exp(-3.0 * nu * t)
    # ω₁ = ∂u₃/∂y - ∂u₂/∂z = 0 - (-A cos(x)sin(y)(-sin(z))) exp = -A cos(x)sin(y)sin(z) exp
    w1 = -A * torch.cos(x) * torch.sin(y) * torch.sin(z) * decay
    # ω₂ = ∂u₁/∂z - ∂u₃/∂x = A sin(x)cos(y)(-sin(z)) exp - 0 = -A sin(x)cos(y)sin(z) exp
    w2 = -A * torch.sin(x) * torch.cos(y) * torch.sin(z) * decay
    # ω₃ = ∂u₂/∂x - ∂u₁/∂y = A sin(x)sin(y)cos(z) exp - (-A sin(x)sin(y)cos(z) exp)
    #     = 2A sin(x)sin(y)cos(z) exp
    # Actually, no. u₂ = -A cos(x)sin(y)cos(z) exp, so ∂u₂/∂x = A sin(x)sin(y)cos(z) exp
    # u₁ = A sin(x)cos(y)cos(z) exp, so ∂u₁/∂y = -A sin(x)sin(y)cos(z) exp
    # ω₃ = A sin(x)sin(y)cos(z) - (-A sin(x)sin(y)cos(z)) = 2A sin(x)sin(y)cos(z) exp
    # But wait — this doesn't match the standard TGV. For the standard 2D TGV (uz=0 and
    # no z-dependence of vorticity source), it should be that ω₃ contributions cancel.
    # Actually ω₃ = ∂u₂/∂x - ∂u₁/∂y is indeed nonzero.
    # Hmm, no. The standard 2D TGV has u₁ = sin(x)cos(y), u₂ = -cos(x)sin(y), with
    # ω₃ = ∂u₂/∂x - ∂u₁/∂y = sin(x)sin(y) - (-sin(x)sin(y)) = 2sin(x)sin(y).
    # In 3D TGV with cos(z) factor, ω₃ = 2A sin(x)sin(y)cos(z) exp.
    # But actually the standard reference uses: ω₃ = 0 for the "symmetric" TGV
    # where the 3D structure is chosen to make ω₃=0. Let me use the simpler version
    # where A_u1 = A, A_u2 = -A, u3=0.
    # With cos(z) multiplier: ∂u₂/∂x = A sin(x)sin(y)cos(z) exp, ∂u₁/∂y = -A sin(x)sin(y)cos(z) exp
    # ω₃ = 2A sin(x)sin(y)cos(z) exp ... this is nonzero.
    # This is fine — the TGV does have 3 nonzero vorticity components.
    w3 = 2.0 * A * torch.sin(x) * torch.sin(y) * torch.cos(z) * decay
    return w1, w2, w3


def pack_multivector(p: torch.Tensor,
                     u1: torch.Tensor, u2: torch.Tensor, u3: torch.Tensor,
                     w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
                     helicity: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pack physical fields into Cl(3,0) multivector [B, 8].

    Index mapping:
        0: scalar (pressure)
        1: e₁ (u₁)
        2: e₂ (u₂)
        3: e₁₂ (ω₃ component — mapped to e₁₂ bivector)
        4: e₃ (u₃)
        5: e₁₃ (ω₂ component — mapped to e₁₃ bivector)
        6: e₂₃ (ω₁ component — mapped to e₂₃ bivector)
        7: e₁₂₃ (helicity density)

    Note: Vorticity ω = (ω₁, ω₂, ω₃) maps to bivectors via Hodge dual:
        ω₁ → +e₂₃ (index 6), ω₂ → -e₁₃ (index 5), ω₃ → +e₁₂ (index 3)
    Sign convention: *(e₂) = e₃₁ = -e₁₃, so ω₂ is stored negated.
    """
    B = p.shape[0]
    mv = torch.zeros(B, 8, dtype=p.dtype, device=p.device)
    mv[:, 0] = p
    mv[:, 1] = u1
    mv[:, 2] = u2
    mv[:, 4] = u3
    mv[:, 3] = w3   # e₁₂: *(e₃) = e₁₂, sign +1
    mv[:, 5] = -w2  # e₁₃: *(e₂) = e₃₁ = -e₁₃, sign -1
    mv[:, 6] = w1   # e₂₃: *(e₁) = e₂₃, sign +1
    if helicity is not None:
        mv[:, 7] = helicity
    return mv


# ============================================================================ #
# Continuum Multivector Derivative
# ============================================================================ #

class MultivectorDerivative:
    """Vector derivative operator ∇Ψ in Cl(3,0) via geometric product.

    The vector derivative ∇Ψ = GP(e₁, ∂Ψ/∂x) + GP(e₂, ∂Ψ/∂y) + GP(e₃, ∂Ψ/∂z)
    is the fundamental differential operator in Geometric Algebra. Grade
    projections of ∇Ψ unify multiple PDE constraints into a single algebraic
    object:

        ⟨∇u⟩₀ = div(u)      — incompressibility constraint
        ⟨∇u⟩₂ = curl(u)     — vorticity (as bivector)

    This replaces 25 separate scalar autograd calls with 8 multivector-valued
    calls (one per basis element), yielding all 32 first-order partials.
    """

    def __init__(self, algebra: CliffordAlgebra):
        self.algebra = algebra
        # Basis vector indices in Cl(3,0): e₁=1, e₂=2, e₃=4 (binary encoding)
        self._basis_indices = [1 << i for i in range(3)]  # [1, 2, 4]
        # Velocity component indices in the multivector
        self._vel_indices = self._basis_indices  # [1, 2, 4]

    def _make_basis_vector(self, idx: int, device, dtype) -> torch.Tensor:
        """Create a unit basis vector eᵢ as a [1, dim] multivector."""
        ei = torch.zeros(1, self.algebra.dim, device=device, dtype=dtype)
        ei[0, idx] = 1.0
        return ei

    def compute_partials(
        self, mv: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
        z: torch.Tensor, t: torch.Tensor, create_graph: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ∂Ψ/∂x, ∂Ψ/∂y, ∂Ψ/∂z, ∂Ψ/∂t as full [B, 8] multivectors.

        Uses 8 autograd calls (one per MV component j), each returning
        grads w.r.t. [x, y, z, t] simultaneously. This yields all 32
        first-order partial derivatives.

        Args:
            mv: [B, 8] full multivector output from the network.
            x, y, z, t: [B, 1] leaf tensors with requires_grad=True.
            create_graph: Whether to create graph for higher-order derivatives.

        Returns:
            (dΨ/dx, dΨ/dy, dΨ/dz, dΨ/dt): each [B, 8].
        """
        B, D = mv.shape
        device, dtype = mv.device, mv.dtype

        # Collect per-component gradients as lists, then stack.
        # In-place assignment to torch.zeros would break the autograd graph.
        cols_dx, cols_dy, cols_dz, cols_dt = [], [], [], []
        zero = torch.zeros(B, device=device, dtype=dtype)

        for j in range(D):
            grads = torch.autograd.grad(
                mv[:, j].sum(), [x, y, z, t],
                create_graph=create_graph,
                allow_unused=True,
            )
            cols_dx.append(grads[0].squeeze(-1) if grads[0] is not None else zero)
            cols_dy.append(grads[1].squeeze(-1) if grads[1] is not None else zero)
            cols_dz.append(grads[2].squeeze(-1) if grads[2] is not None else zero)
            cols_dt.append(grads[3].squeeze(-1) if grads[3] is not None else zero)

        dPsi_dx = torch.stack(cols_dx, dim=-1)  # [B, D]
        dPsi_dy = torch.stack(cols_dy, dim=-1)
        dPsi_dz = torch.stack(cols_dz, dim=-1)
        dPsi_dt = torch.stack(cols_dt, dim=-1)

        return dPsi_dx, dPsi_dy, dPsi_dz, dPsi_dt

    def vector_derivative(
        self, dPsi_dx: torch.Tensor, dPsi_dy: torch.Tensor,
        dPsi_dz: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ∇Ψ = GP(e₁, ∂Ψ/∂x) + GP(e₂, ∂Ψ/∂y) + GP(e₃, ∂Ψ/∂z).

        The grade decomposition of the result encodes:
            Grade 0: scalar contributions (div of vectors)
            Grade 1: vector contributions (grad of scalars, curl of bivectors)
            Grade 2: bivector contributions (curl of vectors)
            Grade 3: pseudoscalar contributions

        When applied to velocity (grade-1 input only):
            ⟨∇u⟩₀ = ∇·u (divergence)
            ⟨∇u⟩₂ = ∇∧u (vorticity as bivector)

        Args:
            dPsi_dx, dPsi_dy, dPsi_dz: [B, 8] multivector partials.

        Returns:
            [B, 8] the vector derivative ∇Ψ.
        """
        device, dtype = dPsi_dx.device, dPsi_dx.dtype
        partials = [dPsi_dx, dPsi_dy, dPsi_dz]

        result = torch.zeros_like(dPsi_dx)
        for i, dPsi_di in enumerate(partials):
            ei = self._make_basis_vector(self._basis_indices[i], device, dtype)
            # GP(eᵢ, ∂Ψ/∂xᵢ): broadcast eᵢ [1, 8] with dPsi_di [B, 8]
            result = result + self.algebra.geometric_product(
                ei.expand(dPsi_di.shape[0], -1), dPsi_di
            )
        return result

    def velocity_laplacian(
        self, dPsi_dx: torch.Tensor, dPsi_dy: torch.Tensor,
        dPsi_dz: torch.Tensor,
        x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Compute ∇²u (Laplacian of velocity) as a grade-1 multivector.

        Requires 9 second-order autograd calls: velocity indices {1,2,4} ×
        spatial variables {x,y,z}.

        Args:
            dPsi_dx, dPsi_dy, dPsi_dz: [B, 8] first-order partials.
            x, y, z: [B, 1] leaf tensors.
            create_graph: Whether to retain graph (needed if loss needs grads).

        Returns:
            [B, 8] grade-1 multivector containing ∇²u.
        """
        B = dPsi_dx.shape[0]
        device, dtype = dPsi_dx.device, dPsi_dx.dtype

        # Accumulate second derivatives per velocity index without in-place ops.
        accum = {vi: [] for vi in self._vel_indices}

        partials_and_vars = [
            (dPsi_dx, x), (dPsi_dy, y), (dPsi_dz, z),
        ]
        for dPsi_di, var_i in partials_and_vars:
            for vel_idx in self._vel_indices:  # [1, 2, 4]
                d2 = torch.autograd.grad(
                    dPsi_di[:, vel_idx].sum(), var_i,
                    create_graph=create_graph,
                    allow_unused=True,
                )[0]
                if d2 is not None:
                    accum[vel_idx].append(d2.squeeze(-1))

        # Build the Laplacian multivector additively (graph-safe, no in-place ops)
        laplacian = torch.zeros(B, self.algebra.dim, device=device, dtype=dtype)
        for vel_idx, parts in accum.items():
            if parts:
                col = torch.stack(parts, dim=0).sum(dim=0)  # [B]
                mask = torch.zeros(self.algebra.dim, device=device, dtype=dtype)
                mask[vel_idx] = 1.0
                laplacian = laplacian + col.unsqueeze(-1) * mask

        return laplacian


# ============================================================================ #
# Dataset
# ============================================================================ #

class TaylorGreenDataset(Dataset):
    """Dataset of Taylor-Green vortex collocation and IC points.

    Generates:
      - Collocation points (x,y,z,t) uniformly in [0,2π]³ × [0,T]
      - Initial condition points at t=0 with analytical targets

    Each sample is (coords [5], target_mv [8]) where coords = [x,y,z,t,log(Re)].
    """

    def __init__(self, num_collocation: int, num_ic: int,
                 re: float, t_max: float = 1.0,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        nu = 1.0 / re
        self.nu = nu
        self.re = re

        # Collocation points (PDE residual enforced here)
        x_c = rng.uniform(0, 2 * np.pi, num_collocation).astype(np.float32)
        y_c = rng.uniform(0, 2 * np.pi, num_collocation).astype(np.float32)
        z_c = rng.uniform(0, 2 * np.pi, num_collocation).astype(np.float32)
        t_c = rng.uniform(0, t_max, num_collocation).astype(np.float32)

        # IC points (exact match enforced here)
        x_i = rng.uniform(0, 2 * np.pi, num_ic).astype(np.float32)
        y_i = rng.uniform(0, 2 * np.pi, num_ic).astype(np.float32)
        z_i = rng.uniform(0, 2 * np.pi, num_ic).astype(np.float32)
        t_i = np.zeros(num_ic, dtype=np.float32)

        # Combine
        x = np.concatenate([x_c, x_i])
        y = np.concatenate([y_c, y_i])
        z = np.concatenate([z_c, z_i])
        t = np.concatenate([t_c, t_i])

        x_t = torch.tensor(x)
        y_t = torch.tensor(y)
        z_t = torch.tensor(z)
        t_t = torch.tensor(t)

        # Compute analytical targets
        u1, u2, u3 = taylor_green_velocity(x_t, y_t, z_t, t_t, nu)
        p = taylor_green_pressure(x_t, y_t, z_t, t_t, nu)
        w1, w2, w3 = taylor_green_vorticity(x_t, y_t, z_t, t_t, nu)
        targets = pack_multivector(p, u1, u2, u3, w1, w2, w3)

        # Coords: [x, y, z, t, log(Re)]
        log_re = np.full(len(x), np.log(re), dtype=np.float32)
        self.coords = torch.stack([x_t, y_t, z_t, t_t,
                                   torch.tensor(log_re)], dim=-1)  # [N, 5]
        self.targets = targets  # [N, 8]
        self.is_ic = torch.zeros(len(x), dtype=torch.bool)
        self.is_ic[num_collocation:] = True

        print(f"  TGV dataset: {num_collocation} collocation + {num_ic} IC points, "
              f"Re={re:.0f}, nu={nu:.6f}, t_max={t_max}")

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int):
        return self.coords[idx], self.targets[idx], self.is_ic[idx]


# ============================================================================ #
# Network
# ============================================================================ #

class GaugeFluidNet(nn.Module):
    """GA network for Navier-Stokes in Cl(3,0).

    Architecture:
        1. Fourier feature embedding of (x, y, z, t, log(Re))
           - Integer spatial frequencies for periodicity
           - Learned temporal frequencies
        2. Lift to multivector hidden space [B, C, 8]
        3. Pre-LN residual blocks: norm → rotor → act → linear + skip
        4. Output head: norm → blade_selector → linear → [B, 8]

    Returns full multivector encoding pressure, velocity, vorticity, helicity.
    """

    def __init__(self, algebra, hidden_dim: int = 64, num_layers: int = 6,
                 num_spatial_freqs: int = 8, num_temporal_freqs: int = 16):
        super().__init__()
        self.algebra = algebra
        self.hidden_dim = hidden_dim

        # Fourier embedding
        # Spatial: integer frequencies for periodicity [0, 2π]
        spatial_freqs = torch.arange(1, num_spatial_freqs + 1, dtype=torch.float32)
        self.register_buffer('spatial_freqs', spatial_freqs)  # [F_s]
        # Temporal: learned frequencies
        self.register_buffer('temporal_freqs',
                             torch.randn(num_temporal_freqs) * 2.0)
        # Re embedding: learned
        self.register_buffer('re_freqs',
                             torch.randn(num_temporal_freqs) * 0.5)

        # Input dim: raw(5) + sin/cos spatial(3*2*F_s) + sin/cos temporal(2*F_t) + sin/cos Re(2*F_t)
        input_dim = 5 + 3 * 2 * num_spatial_freqs + 2 * num_temporal_freqs + 2 * num_temporal_freqs

        # Lift to multivector hidden space
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

        # Output head
        self.output_norm = CliffordLayerNorm(algebra, hidden_dim)
        self.blade_selector = BladeSelector(algebra, channels=hidden_dim)
        self.output_proj = CliffordLinear(algebra, hidden_dim, 1)

        self._intermediates: List[torch.Tensor] = []

    def forward(self, coords: torch.Tensor):
        """Forward pass.

        Args:
            coords: [B, 5] — (x, y, z, t, log(Re)).

        Returns:
            mv: [B, 8] predicted multivector.
            intermediates: list of hidden states.
        """
        B = coords.shape[0]
        x, y, z, t, log_re = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3], coords[:, 4]

        # Spatial Fourier features (periodic)
        spatial_feats = []
        for s_coord in [x, y, z]:
            proj = s_coord.unsqueeze(-1) * self.spatial_freqs  # [B, F_s]
            spatial_feats.extend([torch.sin(proj), torch.cos(proj)])
        spatial_feats = torch.cat(spatial_feats, dim=-1)  # [B, 3*2*F_s]

        # Temporal Fourier features
        t_proj = t.unsqueeze(-1) * self.temporal_freqs  # [B, F_t]
        temporal_feats = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

        # Re Fourier features
        re_proj = log_re.unsqueeze(-1) * self.re_freqs  # [B, F_t]
        re_feats = torch.cat([torch.sin(re_proj), torch.cos(re_proj)], dim=-1)

        # Concatenate all features
        features = torch.cat([coords, spatial_feats, temporal_feats, re_feats], dim=-1)

        # Lift to multivector space
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

        # Output
        h = self.output_norm(h)
        h = self.blade_selector(h)
        out = self.output_proj(h)  # [B, 1, 8]
        out = out.squeeze(1)       # [B, 8]

        return out, intermediates


# ============================================================================ #
# Physics Losses via Autograd
# ============================================================================ #

def compute_ns_residual(model: GaugeFluidNet, coords_raw: torch.Tensor,
                        algebra, nu: float) -> Dict[str, torch.Tensor]:
    """Compute Navier-Stokes PDE residuals via autograd.

    Args:
        model: The gauge fluid network.
        coords_raw: [B, 5] — (x, y, z, t, log_re). Spatial/temporal coords
            are re-created as leaf tensors with requires_grad=True.
        algebra: CliffordAlgebra(3, 0).
        nu: Kinematic viscosity.

    Returns:
        Dict of loss terms: ns_residual, div_residual, lagrangian, vorticity_consistency.
    """
    # Create leaf tensors for autograd differentiation
    x = coords_raw[:, 0:1].detach().requires_grad_(True)
    y = coords_raw[:, 1:2].detach().requires_grad_(True)
    z = coords_raw[:, 2:3].detach().requires_grad_(True)
    t = coords_raw[:, 3:4].detach().requires_grad_(True)
    log_re = coords_raw[:, 4:5].detach()

    # Reconstruct coords from leaf tensors
    coords = torch.cat([x, y, z, t, log_re], dim=-1)

    # Forward pass
    mv, _ = model(coords)

    # Extract fields from multivector
    p = mv[:, 0]     # pressure (grade-0)
    u1 = mv[:, 1]    # velocity e₁
    u2 = mv[:, 2]    # velocity e₂
    u3 = mv[:, 4]    # velocity e₃
    w3_pred = mv[:, 3]   # vorticity e₁₂ (ω₃)
    w2_pred = -mv[:, 5]  # vorticity e₁₃: stored as -ω₂, negate to recover ω₂
    w1_pred = mv[:, 6]   # vorticity e₂₃ (ω₁)

    # --- First derivatives ---
    # ∂u_i/∂x_j and ∂u_i/∂t
    du1_dx = torch.autograd.grad(u1.sum(), x, create_graph=True)[0].squeeze(-1)
    du1_dy = torch.autograd.grad(u1.sum(), y, create_graph=True)[0].squeeze(-1)
    du1_dz = torch.autograd.grad(u1.sum(), z, create_graph=True)[0].squeeze(-1)
    du1_dt = torch.autograd.grad(u1.sum(), t, create_graph=True)[0].squeeze(-1)

    du2_dx = torch.autograd.grad(u2.sum(), x, create_graph=True)[0].squeeze(-1)
    du2_dy = torch.autograd.grad(u2.sum(), y, create_graph=True)[0].squeeze(-1)
    du2_dz = torch.autograd.grad(u2.sum(), z, create_graph=True)[0].squeeze(-1)
    du2_dt = torch.autograd.grad(u2.sum(), t, create_graph=True)[0].squeeze(-1)

    du3_dx = torch.autograd.grad(u3.sum(), x, create_graph=True)[0].squeeze(-1)
    du3_dy = torch.autograd.grad(u3.sum(), y, create_graph=True)[0].squeeze(-1)
    du3_dz = torch.autograd.grad(u3.sum(), z, create_graph=True)[0].squeeze(-1)
    du3_dt = torch.autograd.grad(u3.sum(), t, create_graph=True)[0].squeeze(-1)

    # Pressure gradients
    dp_dx = torch.autograd.grad(p.sum(), x, create_graph=True)[0].squeeze(-1)
    dp_dy = torch.autograd.grad(p.sum(), y, create_graph=True)[0].squeeze(-1)
    dp_dz = torch.autograd.grad(p.sum(), z, create_graph=True)[0].squeeze(-1)

    # --- Second derivatives (Laplacian) ---
    d2u1_dx2 = torch.autograd.grad(du1_dx.sum(), x, create_graph=True)[0].squeeze(-1)
    d2u1_dy2 = torch.autograd.grad(du1_dy.sum(), y, create_graph=True)[0].squeeze(-1)
    d2u1_dz2 = torch.autograd.grad(du1_dz.sum(), z, create_graph=True)[0].squeeze(-1)

    d2u2_dx2 = torch.autograd.grad(du2_dx.sum(), x, create_graph=True)[0].squeeze(-1)
    d2u2_dy2 = torch.autograd.grad(du2_dy.sum(), y, create_graph=True)[0].squeeze(-1)
    d2u2_dz2 = torch.autograd.grad(du2_dz.sum(), z, create_graph=True)[0].squeeze(-1)

    d2u3_dx2 = torch.autograd.grad(du3_dx.sum(), x, create_graph=True)[0].squeeze(-1)
    d2u3_dy2 = torch.autograd.grad(du3_dy.sum(), y, create_graph=True)[0].squeeze(-1)
    d2u3_dz2 = torch.autograd.grad(du3_dz.sum(), z, create_graph=True)[0].squeeze(-1)

    # --- NS momentum residual ---
    # R_i = ∂u_i/∂t + u_j·∂u_i/∂x_j + ∂p/∂x_i - ν∇²u_i
    R1 = du1_dt + u1 * du1_dx + u2 * du1_dy + u3 * du1_dz + dp_dx - nu * (d2u1_dx2 + d2u1_dy2 + d2u1_dz2)
    R2 = du2_dt + u1 * du2_dx + u2 * du2_dy + u3 * du2_dz + dp_dy - nu * (d2u2_dx2 + d2u2_dy2 + d2u2_dz2)
    R3 = du3_dt + u1 * du3_dx + u2 * du3_dy + u3 * du3_dz + dp_dz - nu * (d2u3_dx2 + d2u3_dy2 + d2u3_dz2)

    ns_residual = (R1 ** 2 + R2 ** 2 + R3 ** 2).mean()

    # --- Incompressibility (divergence-free) ---
    div_u = du1_dx + du2_dy + du3_dz
    div_residual = (div_u ** 2).mean()

    # --- Vorticity consistency: ω_pred ≈ ∇×u ---
    # curl(u) components:
    curl_x = du3_dy - du2_dz   # ω₁
    curl_y = du1_dz - du3_dx   # ω₂
    curl_z = du2_dx - du1_dy   # ω₃

    vort_consistency = ((w1_pred - curl_x) ** 2 +
                        (w2_pred - curl_y) ** 2 +
                        (w3_pred - curl_z) ** 2).mean()

    # --- Lagrangian energy balance: dE/dt + 2ν·Ω = 0 ---
    # E = ½(u1² + u2² + u3²), Ω = ½(ω1² + ω2² + ω3²) (enstrophy)
    E = 0.5 * (u1 ** 2 + u2 ** 2 + u3 ** 2)
    dE_dt = torch.autograd.grad(E.sum(), t, create_graph=True)[0].squeeze(-1)
    enstrophy = 0.5 * (w1_pred ** 2 + w2_pred ** 2 + w3_pred ** 2)
    lagrangian = ((dE_dt + 2.0 * nu * enstrophy) ** 2).mean()

    return {
        'ns_residual': ns_residual,
        'div_residual': div_residual,
        'vorticity_consistency': vort_consistency,
        'lagrangian': lagrangian,
        'mv': mv,
    }


def compute_ns_residual_mv_from_output(
    mv: torch.Tensor,
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
    algebra, nu: float, mv_deriv: MultivectorDerivative,
    colloc_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute NS residuals via continuum multivector formulation.

    All PDE constraints emerge from the vector derivative ∇Ψ and grade
    projections, rather than component-wise scalar differentiation:

        ∇u = GP(e₁, ∂u/∂x) + GP(e₂, ∂u/∂y) + GP(e₃, ∂u/∂z)
        ⟨∇u⟩₀ = div(u) = 0           (incompressibility)
        ⟨∇u⟩₂ = curl(u) = ω          (vorticity consistency)
        ∂ₜu + (u·∇)u + ∇p - ν∇²u = 0 (momentum, grade-1 equation)

    This operates on a pre-computed mv output (no internal forward pass),
    enabling single-forward-pass training. Derivatives are computed on the
    full batch (preserving the autograd graph from leaf tensors to mv), then
    masked to collocation points before averaging.

    Args:
        mv: [B, 8] multivector output from the network (with grad graph).
        x, y, z, t: [B, 1] leaf tensors with requires_grad=True (full batch).
        algebra: CliffordAlgebra(3, 0).
        nu: Kinematic viscosity.
        mv_deriv: MultivectorDerivative instance.
        colloc_mask: [B] boolean mask for collocation points (PDE enforced here).
            If None, all points are treated as collocation.

    Returns:
        Dict of loss terms: ns_residual, div_residual, vorticity_consistency,
        lagrangian, mv.
    """
    # --- Step 1: Compute all first-order partials as full multivectors ---
    # 8 autograd calls (one per MV component), yielding all 32 first-order
    # partial derivatives on the FULL batch (keeps autograd graph intact).
    dPsi_dx, dPsi_dy, dPsi_dz, dPsi_dt = mv_deriv.compute_partials(
        mv, x, y, z, t
    )

    # --- Step 2: Extract velocity partials via grade projection ---
    du_dx = algebra.grade_projection(dPsi_dx, 1)  # [B, 8]
    du_dy = algebra.grade_projection(dPsi_dy, 1)
    du_dz = algebra.grade_projection(dPsi_dz, 1)

    # --- Step 3: Vector derivative ∇u = GP(eᵢ, ∂u/∂xᵢ) ---
    # This single multivector encodes both divergence and curl.
    nabla_u = mv_deriv.vector_derivative(du_dx, du_dy, du_dz)

    # --- Step 4: Divergence from grade-0 projection ---
    # ⟨∇u⟩₀ = div(u) — the incompressibility constraint
    div_u = algebra.grade_projection(nabla_u, 0)[..., 0]  # [B]

    # --- Step 5: Vorticity consistency from grade-2 projection ---
    # ⟨∇u⟩₂ = curl(u) as bivector — should match predicted vorticity
    curl_u_bv = algebra.grade_projection(nabla_u, 2)   # [B, 8]
    vort_pred = algebra.grade_projection(mv, 2)         # [B, 8]
    vort_diff = (curl_u_bv - vort_pred) ** 2  # [B, 8]

    # --- Step 6: Momentum equation (grade-1 residual) ---
    # ∂ₜu
    dt_u = algebra.grade_projection(dPsi_dt, 1)  # [B, 8]

    # Advection: (u·∇)u = u₁·(∂u/∂x) + u₂·(∂u/∂y) + u₃·(∂u/∂z)
    # Scalar velocity components × grade-1 multivector partials
    u1 = mv[:, 1]  # [B]
    u2 = mv[:, 2]
    u3 = mv[:, 4]
    advection = (u1.unsqueeze(-1) * du_dx +
                 u2.unsqueeze(-1) * du_dy +
                 u3.unsqueeze(-1) * du_dz)  # [B, 8] (grade-1)

    # Pressure gradient: ∇p = embed_vector([dp/dx, dp/dy, dp/dz])
    # dp/dxᵢ is the grade-0 component of ∂Ψ/∂xᵢ
    grad_p = algebra.embed_vector(
        torch.stack([dPsi_dx[:, 0], dPsi_dy[:, 0], dPsi_dz[:, 0]], dim=-1)
    )  # [B, 8] grade-1

    # Laplacian: ν∇²u (9 second-order autograd calls)
    laplacian_u = mv_deriv.velocity_laplacian(
        dPsi_dx, dPsi_dy, dPsi_dz, x, y, z
    )  # [B, 8] grade-1

    # NS momentum residual: R = ∂ₜu + (u·∇)u + ∇p - ν∇²u
    R_momentum = dt_u + advection + grad_p - nu * laplacian_u  # [B, 8]
    # Only grade-1 components carry the momentum equation
    R_momentum = algebra.grade_projection(R_momentum, 1)
    R_sq = (R_momentum ** 2).sum(dim=-1)  # [B]

    # --- Step 7: Lagrangian energy balance ---
    # E = ½|u|², dE/dt = u · ∂ₜu (chain rule, no extra autograd needed)
    dE_dt = (u1 * dt_u[:, 1] + u2 * dt_u[:, 2] + u3 * dt_u[:, 4])
    # Enstrophy from predicted vorticity bivector
    w1_pred = vort_pred[:, 6]    # e₂₃
    w2_pred = -vort_pred[:, 5]   # -e₁₃ → ω₂
    w3_pred = vort_pred[:, 3]    # e₁₂
    enstrophy = 0.5 * (w1_pred ** 2 + w2_pred ** 2 + w3_pred ** 2)
    lagr_sq = (dE_dt + 2.0 * nu * enstrophy) ** 2  # [B]

    # --- Apply collocation mask and average ---
    if colloc_mask is not None:
        div_residual = (div_u[colloc_mask] ** 2).mean()
        vort_consistency = vort_diff[colloc_mask].sum(dim=-1).mean()
        ns_residual = R_sq[colloc_mask].mean()
        lagrangian = lagr_sq[colloc_mask].mean()
    else:
        div_residual = (div_u ** 2).mean()
        vort_consistency = vort_diff.sum(dim=-1).mean()
        ns_residual = R_sq.mean()
        lagrangian = lagr_sq.mean()

    return {
        'ns_residual': ns_residual,
        'div_residual': div_residual,
        'vorticity_consistency': vort_consistency,
        'lagrangian': lagrangian,
        'mv': mv,
    }


def compute_gauge_covariance_loss(mv: torch.Tensor, algebra) -> torch.Tensor:
    """Test gauge covariance: <RuR̃, RuR̃>_H ≈ <u, u>_H.

    Generates a random rotor R = exp(-B/2) and checks that the
    Hermitian norm of the velocity (grade-1) is preserved under
    the sandwich product.

    Args:
        mv: [B, 8] multivector output.
        algebra: CliffordAlgebra(3, 0).

    Returns:
        Scalar gauge covariance error.
    """
    # Extract grade-1 (velocity)
    vel = algebra.grade_projection(mv, 1)  # [B, 8]

    # Random bivector for rotor
    bv = torch.zeros(1, algebra.dim, device=mv.device)
    bv_coeffs = torch.randn(3, device=mv.device) * 0.5
    bv[0, 3] = bv_coeffs[0]  # e₁₂
    bv[0, 5] = bv_coeffs[1]  # e₁₃
    bv[0, 6] = bv_coeffs[2]  # e₂₃

    # Rotor R = exp(-B/2)
    rotor = algebra.exp(-0.5 * bv)  # [1, 8]
    rotor_rev = algebra.reverse(rotor)  # [1, 8]

    # Sandwich product: R v R̃
    temp = algebra.geometric_product(rotor.expand(mv.shape[0], -1), vel)
    transformed = algebra.geometric_product(temp, rotor_rev.expand(mv.shape[0], -1))

    # Compare Hermitian norms
    norm_orig = hermitian_inner_product(algebra, vel, vel)       # [B, 1]
    norm_trans = hermitian_inner_product(algebra, transformed, transformed)  # [B, 1]

    return (norm_orig - norm_trans).abs().mean()


# ============================================================================ #
# Gauge Fluid Metric
# ============================================================================ #

class GaugeFluidMetric:
    """Gauge-theoretic metrics for fluid dynamics in Cl(3,0).

    Interprets:
      - Grade-1 (velocity) as gauge connection → kinetic energy
      - Grade-2 (vorticity) as gauge curvature → enstrophy
      - Grade-3 (helicity) as topological density
    """

    def __init__(self, algebra):
        self.algebra = algebra

    def connection_energy(self, mv: torch.Tensor) -> torch.Tensor:
        """Kinetic energy = <u, u>_H (Hermitian norm of grade-1)."""
        vel = self.algebra.grade_projection(mv, 1)
        return hermitian_inner_product(self.algebra, vel, vel).squeeze(-1)

    def curvature_energy(self, mv: torch.Tensor) -> torch.Tensor:
        """Enstrophy = <ω, ω>_H (Hermitian norm of grade-2)."""
        vort = self.algebra.grade_projection(mv, 2)
        return hermitian_inner_product(self.algebra, vort, vort).squeeze(-1)

    def bkm_criterion(self, mv: torch.Tensor) -> torch.Tensor:
        """Beale-Kato-Majda criterion: max|ω|.

        If ∫₀ᵀ max|ω| dt < ∞, the solution stays regular.
        """
        vort = self.algebra.grade_projection(mv, 2)
        return hermitian_norm(self.algebra, vort).squeeze(-1).max()

    def helicity(self, mv: torch.Tensor) -> torch.Tensor:
        """Helicity density = grade-3 pseudoscalar component."""
        return mv[..., 7]

    def gauge_covariance_error(self, mv: torch.Tensor) -> float:
        """Test <RuR̃, RuR̃>_H = <u, u>_H for random R."""
        with torch.no_grad():
            err = compute_gauge_covariance_loss(mv, self.algebra)
        return err.item()

    def energy_balance(self, dEdt: torch.Tensor,
                       enstrophy: torch.Tensor, nu: float) -> torch.Tensor:
        """Energy-enstrophy balance: |dE/dt + 2ν·Ω|."""
        return (dEdt + 2.0 * nu * enstrophy).abs()

    def format_report(self, mv: torch.Tensor, nu: float) -> str:
        """ASCII formatted diagnostics report."""
        ke = self.connection_energy(mv)
        enst = self.curvature_energy(mv)
        bkm = self.bkm_criterion(mv)
        hel = self.helicity(mv)
        gauge_err = self.gauge_covariance_error(mv)

        lines = [
            "  --- Gauge Fluid Metric Report ---",
            f"  Kinetic energy  <u,u>_H:   mean={ke.mean().item():.6f}, max={ke.max().item():.6f}",
            f"  Enstrophy <ω,ω>_H:         mean={enst.mean().item():.6f}, max={enst.max().item():.6f}",
            f"  BKM criterion max|ω|:      {bkm.item():.6f}",
            f"  Helicity (grade-3):         mean={hel.mean().item():.6f}, std={hel.std().item():.6f}",
            f"  Gauge covariance error:     {gauge_err:.6e}",
        ]
        return '\n'.join(lines)


# ============================================================================ #
# Regularity Debugger
# ============================================================================ #

class NSRegularityDebugger:
    """Physics diagnostics for Navier-Stokes regularity in Cl(3,0)."""

    def __init__(self, algebra, metric: GaugeFluidMetric):
        self.algebra = algebra
        self.metric = metric

    @torch.no_grad()
    def enstrophy_evolution(self, model: GaugeFluidNet, t_grid: torch.Tensor,
                            spatial_pts: torch.Tensor, nu: float,
                            re: float) -> Dict[str, np.ndarray]:
        """Track enstrophy over time: predicted vs analytical."""
        model.eval()
        pred_enst = []
        exact_enst = []

        for t_val in t_grid:
            N = spatial_pts.shape[0]
            t_col = torch.full((N, 1), t_val.item(), device=spatial_pts.device)
            log_re_col = torch.full((N, 1), np.log(re), device=spatial_pts.device)
            coords = torch.cat([spatial_pts, t_col, log_re_col], dim=-1)

            mv, _ = model(coords)
            enst = self.metric.curvature_energy(mv).mean().item()
            pred_enst.append(enst)

            # Analytical enstrophy decay: Ω(t) = Ω₀ exp(-6νt)
            # For TGV, Ω₀ depends on the domain integral
            exact_decay = math.exp(-6.0 * nu * t_val.item())
            exact_enst.append(exact_decay)

        return {
            't': t_grid.numpy(),
            'pred_enstrophy': np.array(pred_enst),
            'exact_decay': np.array(exact_enst),
        }

    @torch.no_grad()
    def bkm_tracking(self, model: GaugeFluidNet, t_grid: torch.Tensor,
                     spatial_pts: torch.Tensor, nu: float,
                     re: float) -> Dict[str, np.ndarray]:
        """Track BKM criterion (max|ω|) over time."""
        model.eval()
        bkm_vals = []

        for t_val in t_grid:
            N = spatial_pts.shape[0]
            t_col = torch.full((N, 1), t_val.item(), device=spatial_pts.device)
            log_re_col = torch.full((N, 1), np.log(re), device=spatial_pts.device)
            coords = torch.cat([spatial_pts, t_col, log_re_col], dim=-1)

            mv, _ = model(coords)
            bkm = self.metric.bkm_criterion(mv).item()
            bkm_vals.append(bkm)

        return {
            't': t_grid.numpy(),
            'bkm': np.array(bkm_vals),
        }

    @torch.no_grad()
    def grade_spectrum_analysis(self, model: GaugeFluidNet,
                                coords: torch.Tensor) -> Dict[str, float]:
        """Grade energy spectrum via Hermitian inner product."""
        model.eval()
        mv, _ = model(coords)
        spectrum = hermitian_grade_spectrum(self.algebra, mv)  # [B, 4]
        mean_spec = spectrum.mean(dim=0)

        labels = ['pressure(G0)', 'velocity(G1)', 'vorticity(G2)', 'helicity(G3)']
        return {labels[k]: mean_spec[k].item() for k in range(4)}

    @torch.no_grad()
    def reynolds_sweep(self, model: GaugeFluidNet,
                       re_values: List[float],
                       spatial_pts: torch.Tensor,
                       t_val: float = 0.5) -> Dict[str, list]:
        """Stability diagnostics across Reynolds numbers."""
        model.eval()
        results = {'re': [], 'enstrophy': [], 'bkm': [],
                   'energy_balance': [], 'gauge_error': []}

        for re in re_values:
            nu = 1.0 / re
            N = spatial_pts.shape[0]
            t_col = torch.full((N, 1), t_val, device=spatial_pts.device)
            log_re_col = torch.full((N, 1), np.log(re), device=spatial_pts.device)
            coords = torch.cat([spatial_pts, t_col, log_re_col], dim=-1)

            mv, _ = model(coords)

            results['re'].append(re)
            results['enstrophy'].append(self.metric.curvature_energy(mv).mean().item())
            results['bkm'].append(self.metric.bkm_criterion(mv).item())
            results['gauge_error'].append(self.metric.gauge_covariance_error(mv))

            # Simple energy balance proxy
            ke = self.metric.connection_energy(mv).mean().item()
            enst = self.metric.curvature_energy(mv).mean().item()
            results['energy_balance'].append(abs(2.0 * nu * enst))

        return results

    @torch.no_grad()
    def gauge_covariance_test(self, model: GaugeFluidNet,
                              coords: torch.Tensor) -> float:
        """Verify <RuR̃, RuR̃>_H = <u, u>_H for random R."""
        model.eval()
        mv, _ = model(coords)
        return self.metric.gauge_covariance_error(mv)

    def format_report(self, model: GaugeFluidNet,
                      coords: torch.Tensor, nu: float, re: float) -> str:
        """Human-readable diagnostic report."""
        model.eval()
        with torch.no_grad():
            mv, _ = model(coords)

        lines = [self.metric.format_report(mv, nu)]

        # Grade spectrum
        spec = self.grade_spectrum_analysis(model, coords)
        lines.append("  Grade spectrum:")
        for label, val in spec.items():
            lines.append(f"    {label}: {val:.6f}")

        # Gauge covariance
        gauge_err = self.gauge_covariance_test(model, coords)
        lines.append(f"  Gauge covariance test:      {gauge_err:.6e}")

        return '\n'.join(lines)


# ============================================================================ #
# Visualization
# ============================================================================ #

def _save_plots(history: dict, model: GaugeFluidNet,
                debugger: NSRegularityDebugger,
                algebra, ortho: StrictOrthogonality,
                last_intermediates: List[torch.Tensor],
                test_coords: torch.Tensor,
                test_targets: torch.Tensor,
                nu: float, re: float,
                output_dir: str) -> None:
    """Save diagnostic plots to disk.

    Plots:
        1. convergence.png         - Loss curves
        2. vorticity_evolution.png - Enstrophy + max|ω| vs time
        3. grade_spectrum.png      - Energy per grade
        4. reynolds_sweep.png      - 2×2 diagnostics vs Re
        5. velocity_field.png      - Predicted vs exact velocity
        6. coupling_heatmap.png    - Cross-grade coupling
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Convergence curves
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history['epochs']

    ax = axes[0]
    for key, label, color in [
        ('ns_loss', 'NS residual', 'steelblue'),
        ('div_loss', 'Divergence', 'tomato'),
        ('ic_loss', 'IC', 'goldenrod'),
        ('total_loss', 'Total', 'black'),
    ]:
        if history.get(key):
            ax.semilogy(epochs, history[key], label=label, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key, label, color in [
        ('lagrangian_loss', 'Lagrangian', 'purple'),
        ('vorticity_loss', 'Vorticity', 'teal'),
        ('gauge_loss', 'Gauge', 'orange'),
    ]:
        if history.get(key):
            ax.semilogy(epochs, history[key], label=label, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.set_title('Physics Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Navier-Stokes Gauge Theory - Cl(3,0)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Vorticity evolution
    # ------------------------------------------------------------------ #
    device = test_coords.device
    rng = np.random.RandomState(99)
    spatial_pts_np = rng.uniform(0, 2 * np.pi, (200, 3)).astype(np.float32)
    spatial_pts = torch.tensor(spatial_pts_np, device=device)
    t_grid = torch.linspace(0, 1.0, 20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    evo = debugger.enstrophy_evolution(model, t_grid, spatial_pts, nu, re)
    ax = axes[0]
    ax.plot(evo['t'], evo['pred_enstrophy'], 'b-o', markersize=3, label='Predicted')
    ax.plot(evo['t'], evo['exact_decay'] * evo['pred_enstrophy'][0], 'r--',
            label='Analytical decay')
    ax.set_xlabel('Time')
    ax.set_ylabel('Enstrophy')
    ax.set_title(f'Enstrophy Evolution (Re={re:.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    bkm = debugger.bkm_tracking(model, t_grid, spatial_pts, nu, re)
    ax = axes[1]
    ax.plot(bkm['t'], bkm['bkm'], 'g-o', markersize=3, label='max|ω|')
    ax.set_xlabel('Time')
    ax.set_ylabel('max|ω| (BKM)')
    ax.set_title('Beale-Kato-Majda Criterion')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'vorticity_evolution.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. Grade spectrum
    # ------------------------------------------------------------------ #
    model.eval()
    with torch.no_grad():
        mv, _ = model(test_coords)
    spectrum = hermitian_grade_spectrum(algebra, mv).mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Pressure\n(G0)', 'Velocity\n(G1)', 'Vorticity\n(G2)', 'Helicity\n(G3)']
    colors = ['royalblue', 'mediumseagreen', 'tomato', 'gold']
    bars = ax.bar(range(4), spectrum, color=colors, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, spectrum):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Hermitian Energy')
    ax.set_title('Grade Energy Spectrum')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'grade_spectrum.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 4. Reynolds sweep
    # ------------------------------------------------------------------ #
    re_values = [100, 1000, 10000, 100000]
    sweep = debugger.reynolds_sweep(model, re_values, spatial_pts)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.semilogx(sweep['re'], sweep['enstrophy'], 'bo-')
    ax.set_xlabel('Re'); ax.set_ylabel('Enstrophy')
    ax.set_title('Enstrophy vs Re'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx(sweep['re'], sweep['bkm'], 'go-')
    ax.set_xlabel('Re'); ax.set_ylabel('max|ω|')
    ax.set_title('BKM vs Re'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(sweep['re'], sweep['energy_balance'], 'ro-')
    ax.set_xlabel('Re'); ax.set_ylabel('2ν·Ω')
    ax.set_title('Energy Balance vs Re'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogx(sweep['re'], sweep['gauge_error'], 'mo-')
    ax.set_xlabel('Re'); ax.set_ylabel('Gauge Error')
    ax.set_title('Gauge Covariance vs Re'); ax.grid(True, alpha=0.3)

    fig.suptitle('Reynolds Number Sweep', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reynolds_sweep.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 5. Velocity field comparison
    # ------------------------------------------------------------------ #
    N_grid = 50
    x_lin = np.linspace(0, 2 * np.pi, N_grid, dtype=np.float32)
    y_lin = np.linspace(0, 2 * np.pi, N_grid, dtype=np.float32)
    xx, yy = np.meshgrid(x_lin, y_lin)
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    zz_flat = np.full_like(xx_flat, np.pi)  # z = π slice
    tt_flat = np.zeros_like(xx_flat)  # t = 0

    grid_coords = torch.tensor(np.stack([xx_flat, yy_flat, zz_flat, tt_flat,
                                         np.full_like(xx_flat, np.log(re))], axis=-1),
                                device=device)

    with torch.no_grad():
        mv_grid, _ = model(grid_coords)
    pred_u1 = mv_grid[:, 1].cpu().numpy().reshape(N_grid, N_grid)
    pred_u2 = mv_grid[:, 2].cpu().numpy().reshape(N_grid, N_grid)
    pred_u3 = mv_grid[:, 4].cpu().numpy().reshape(N_grid, N_grid)

    # Exact
    x_t = torch.tensor(xx_flat)
    y_t = torch.tensor(yy_flat)
    z_t = torch.tensor(zz_flat)
    t_t = torch.tensor(tt_flat)
    eu1, eu2, eu3 = taylor_green_velocity(x_t, y_t, z_t, t_t, nu)
    exact_u1 = eu1.numpy().reshape(N_grid, N_grid)
    exact_u2 = eu2.numpy().reshape(N_grid, N_grid)
    exact_u3 = eu3.numpy().reshape(N_grid, N_grid)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles = ['u₁ (predicted)', 'u₂ (predicted)', 'u₃ (predicted)',
              'u₁ (exact)', 'u₂ (exact)', 'u₃ (exact)']
    data = [pred_u1, pred_u2, pred_u3, exact_u1, exact_u2, exact_u3]

    vmin = min(d.min() for d in data)
    vmax = max(d.max() for d in data)

    for idx, (ax, d, title) in enumerate(zip(axes.ravel(), data, titles)):
        im = ax.imshow(d, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       extent=[0, 2 * np.pi, 0, 2 * np.pi], origin='lower')
        ax.set_title(title)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Velocity Field at z=π, t=0', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'velocity_field.png'), dpi=150,
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

def _build_re_schedule(epoch: int, total_epochs: int,
                       re_min: float, re_max: float) -> float:
    """Curriculum Reynolds number: ramp from re_min to re_max.

    First 20% of epochs: constant at re_min (warmup).
    Remaining 80%: log-linear ramp to re_max.
    """
    warmup_frac = 0.2
    warmup_end = int(total_epochs * warmup_frac)

    if epoch <= warmup_end:
        return re_min

    frac = (epoch - warmup_end) / max(total_epochs - warmup_end, 1)
    log_re = math.log(re_min) + frac * (math.log(re_max) - math.log(re_min))
    return math.exp(log_re)


def train(args):
    """Main training loop with physics-informed losses and diagnostics."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    # Algebra: Cl(3,0), dim=8
    algebra = CliffordAlgebra(p=3, q=0, device=device)
    print(f"\n{'='*60}")
    print(f" Navier-Stokes as Gauge Theory - Cl(3,0)")
    print(f" Grade mapping: G0=pressure, G1=velocity, G2=vorticity, G3=helicity")
    print(f" Re range: [{args.re_min}, {args.re_max}] (curriculum)")
    print(f" Strict Orthogonality: {'ON' if args.strict_ortho else 'OFF'}"
          f"{f' (weight={args.ortho_weight}, mode={args.ortho_mode})' if args.strict_ortho else ''}")
    print(f"{'='*60}\n")

    # Orthogonality
    ortho_settings = OrthogonalitySettings(
        enabled=args.strict_ortho,
        mode=args.ortho_mode,
        weight=args.ortho_weight,
        target_grades=[0, 1, 2, 3],  # all grades valid for NS
        tolerance=1e-3,
        monitor_interval=args.diag_interval,
        coupling_warn_threshold=0.3,
    )
    ortho = StrictOrthogonality(algebra, ortho_settings).to(device)

    # Metric, debugger, and multivector derivative operator
    gauge_metric = GaugeFluidMetric(algebra)
    debugger = NSRegularityDebugger(algebra, gauge_metric)
    mv_deriv = MultivectorDerivative(algebra)

    # Dataset (at initial Re)
    re_init = args.re_min
    nu_init = 1.0 / re_init
    print("Generating datasets...")
    train_ds = TaylorGreenDataset(
        num_collocation=args.num_collocation,
        num_ic=args.num_ic,
        re=re_init, t_max=args.t_max,
        seed=args.seed,
    )
    test_ds = TaylorGreenDataset(
        num_collocation=args.num_collocation // 4,
        num_ic=args.num_ic // 4,
        re=re_init, t_max=args.t_max,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Network
    model = GaugeFluidNet(
        algebra,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nGaugeFluidNet: {args.hidden_dim} hidden, {args.num_layers} layers, "
          f"{n_params:,} parameters\n")

    # Optimizer
    optimizer = RiemannianAdam(model.parameters(), lr=args.lr, algebra=algebra)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training history
    history: Dict[str, list] = {
        'epochs': [], 'total_loss': [], 'ns_loss': [], 'div_loss': [],
        'ic_loss': [], 'lagrangian_loss': [], 'vorticity_loss': [],
        'gauge_loss': [],
    }

    best_test_loss = float('inf')
    last_intermediates: List[torch.Tensor] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Curriculum Reynolds number
        current_re = _build_re_schedule(epoch, args.epochs, args.re_min, args.re_max)
        current_nu = 1.0 / current_re

        epoch_losses = {k: 0.0 for k in ['total', 'ns', 'div', 'ic',
                                           'lagrangian', 'vorticity', 'gauge', 'ortho']}
        n_batches = 0

        for coords, targets, is_ic in train_loader:
            coords = coords.to(device)
            targets = targets.to(device)
            is_ic = is_ic.to(device)

            # Create leaf tensors for autograd before the single forward pass
            x = coords[:, 0:1].detach().requires_grad_(True)
            y = coords[:, 1:2].detach().requires_grad_(True)
            z = coords[:, 2:3].detach().requires_grad_(True)
            t = coords[:, 3:4].detach().requires_grad_(True)
            log_re = torch.full((coords.shape[0], 1), np.log(current_re),
                                device=device, dtype=coords.dtype)
            coords_leaf = torch.cat([x, y, z, t, log_re], dim=-1)

            # === Single forward pass for ALL points ===
            mv_all, intermediates = model(coords_leaf)

            # IC loss (on initial condition points)
            if is_ic.any():
                ic_loss = nn.functional.mse_loss(mv_all[is_ic], targets[is_ic])
            else:
                ic_loss = torch.tensor(0.0, device=device)

            # PDE residuals via continuum multivector formulation.
            # Derivatives computed on full batch (preserving autograd graph),
            # then masked to collocation points for the loss.
            colloc_mask = ~is_ic
            if colloc_mask.any():
                residuals = compute_ns_residual_mv_from_output(
                    mv_all, x, y, z, t,
                    algebra, current_nu, mv_deriv,
                    colloc_mask=colloc_mask,
                )
                ns_loss = residuals['ns_residual']
                div_loss = residuals['div_residual']
                lagrangian_loss = residuals['lagrangian']
                vorticity_loss = residuals['vorticity_consistency']
            else:
                ns_loss = div_loss = lagrangian_loss = vorticity_loss = torch.tensor(0.0, device=device)

            # Gauge covariance loss (reuses mv_all — no extra forward pass)
            gauge_loss = compute_gauge_covariance_loss(mv_all, algebra)

            # Orthogonality loss
            ortho_loss = torch.tensor(0.0, device=device)
            if args.strict_ortho and intermediates:
                eff_weight = ortho.anneal_weight(epoch,
                                                  warmup_epochs=args.ortho_warmup,
                                                  total_epochs=args.epochs)
                for h in intermediates:
                    h_flat = h.reshape(-1, algebra.dim)
                    ortho_loss = ortho_loss + eff_weight * ortho.parasitic_energy(h_flat)
                ortho_loss = ortho_loss / len(intermediates)

            # Total loss
            loss = (args.ns_weight * ns_loss +
                    args.div_weight * div_loss +
                    args.ic_weight * ic_loss +
                    args.lagrangian_weight * lagrangian_loss +
                    args.vorticity_weight * vorticity_loss +
                    args.gauge_weight * gauge_loss +
                    ortho_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['total'] += loss.item()
            epoch_losses['ns'] += ns_loss.item()
            epoch_losses['div'] += div_loss.item()
            epoch_losses['ic'] += ic_loss.item()
            epoch_losses['lagrangian'] += lagrangian_loss.item()
            epoch_losses['vorticity'] += vorticity_loss.item()
            epoch_losses['gauge'] += gauge_loss.item()
            epoch_losses['ortho'] += ortho_loss.item()
            n_batches += 1

        scheduler.step()
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

        # Evaluate periodically
        if epoch % args.diag_interval == 0 or epoch == 1 or epoch == args.epochs:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:4d}/{args.epochs} | Re={current_re:.0f} | "
                  f"Loss: {avg['total']:.6f} "
                  f"(ns:{avg['ns']:.4f} div:{avg['div']:.4f} ic:{avg['ic']:.4f} "
                  f"lag:{avg['lagrangian']:.4f} vort:{avg['vorticity']:.4f} "
                  f"gauge:{avg['gauge']:.4f}) | LR: {lr:.6f}")

            # Diagnostics
            model.eval()
            with torch.no_grad():
                test_coords = test_ds.coords.to(device)
                report = debugger.format_report(model, test_coords, current_nu, current_re)
                print(report)

                # Get intermediates for ortho
                _, last_intermediates_list = model(test_coords)
                last_intermediates = [h.cpu() for h in last_intermediates_list]

            if args.strict_ortho and last_intermediates:
                last_h = last_intermediates[-1].reshape(-1, algebra.dim)
                print(ortho.format_diagnostics(last_h))

            # Record history
            history['epochs'].append(epoch)
            history['total_loss'].append(avg['total'])
            history['ns_loss'].append(avg['ns'])
            history['div_loss'].append(avg['div'])
            history['ic_loss'].append(avg['ic'])
            history['lagrangian_loss'].append(avg['lagrangian'])
            history['vorticity_loss'].append(avg['vorticity'])
            history['gauge_loss'].append(avg['gauge'])

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
    test_targets = test_ds.targets.to(device)

    with torch.no_grad():
        mv_final, final_inters = model(test_coords)
        last_intermediates = [h.cpu() for h in final_inters]

        ic_mask = test_ds.is_ic
        if ic_mask.any():
            ic_err = nn.functional.mse_loss(mv_final[ic_mask], test_targets[ic_mask])
            print(f"  IC MSE:               {ic_err.item():.6f}")

        print(debugger.format_report(model, test_coords, 1.0 / args.re_max, args.re_max))
        print(f"  Best training loss:   {best_test_loss:.6f}")

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
                test_targets=test_targets,
                nu=1.0 / args.re_max,
                re=args.re_max,
                output_dir=args.output_dir,
            )

    print(f"\n{'='*60}\n")
    return model, best_test_loss


# ============================================================================ #
# CLI
# ============================================================================ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Navier-Stokes as Gauge Theory in Cl(3,0)')

    # Data
    p.add_argument('--re-min', type=float, default=100.0,
                   help='Initial Reynolds number')
    p.add_argument('--re-max', type=float, default=10000.0,
                   help='Maximum Reynolds number (curriculum target)')
    p.add_argument('--t-max', type=float, default=1.0,
                   help='Maximum time')
    p.add_argument('--num-collocation', type=int, default=3000,
                   help='Number of collocation points')
    p.add_argument('--num-ic', type=int, default=1000,
                   help='Number of initial condition points')

    # Model
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=6)

    # Training
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='mps')

    # Loss weights
    p.add_argument('--ns-weight', type=float, default=1.0,
                   help='NS momentum residual weight')
    p.add_argument('--div-weight', type=float, default=10.0,
                   help='Incompressibility (divergence) weight')
    p.add_argument('--ic-weight', type=float, default=10.0,
                   help='Initial condition weight')
    p.add_argument('--lagrangian-weight', type=float, default=1.0,
                   help='Energy-enstrophy balance weight')
    p.add_argument('--vorticity-weight', type=float, default=1.0,
                   help='Vorticity consistency weight')
    p.add_argument('--gauge-weight', type=float, default=0.1,
                   help='Gauge covariance weight')

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
    p.add_argument('--output-dir', type=str, default='ns_plots',
                   help='Directory for saving plots')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
