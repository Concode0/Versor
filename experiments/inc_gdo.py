# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#


"""
==============================================================================
VERSOR EXPERIMENT: IDEA INCUBATOR (SPIN-OFF CONCEPT)
==============================================================================

This script serves as an early-stage proof-of-concept for radical, non-Euclidean
architectures. The concepts demonstrated here are strongly driven by geometric
intuition and may currently reside ahead of established academic literature.

Please understand that rigorous mathematical proofs or comprehensive citations
might be incomplete at this stage. If this geometric hypothesis proves structurally
sound, it is planned to be spin off into a dedicated, independent repository
for detailed research.

==============================================================================

Geometric Deterministic Optimizer (GDO): Topology-Aware Optimization.

HYPOTHESIS
==========
Standard optimizers (SGD, Adam) treat parameter space as flat Euclidean space,
ignoring the topological structure of the loss landscape. We hypothesize that:

  1. TOPOLOGY SEARCH (Morse Theory)
     The loss landscape L: R^n -> R is a Morse function almost everywhere.
     Its critical points (grad L = 0) partition into types by Morse index k:
       - k=0: local minimum (all Hessian eigenvalues > 0)
       - k=n: local maximum (all eigenvalues < 0)
       - k=j: saddle point with j downward directions
     The Morse complex (critical points + connecting gradient flow lines)
     encodes the global topology. Finding this map lets us reason about
     whether a local minimum is isolated or connected to something lower.

  2. PROBE DEPLOYMENT
     Rather than blindly following gradient, deploy lightweight probes that
     measure curvature k(theta) = Tr(H) / ||grad L|| and gradient alignment
     across a small sphere around the current point. This curvature map reveals:
       - Flat plateau regions (|k| ~= 0, ||grad L|| ~= 0)
       - Narrow valleys (k >> 0 in one direction, ~= 0 in others)
       - Saddle approach directions (k < 0)

  3. GEODESIC PATH
     Instead of following instantaneous gradient, compute the geodesic from
     current position theta to the nearest detected lower critical point.
     Geodesic equation: d^2 theta / dt^2 + Gamma^k_ij (d theta^i/dt)(d theta^j/dt) = 0
     where Gamma^k_ij are Christoffel symbols from the parameter-space metric G_ij(theta).
     For loss landscapes, a natural metric is G_ij = H_ij + lambda*I (regularized Hessian).
     Since exact computation of the Christoffel symbols Gamma^k_ij requires
     3rd-order derivatives, we bypass continuous integration entirely.
     Instead, we approximate the natural metric using a pseudo-Hessian diagonal
     (proportional to gradient magnitude) and interpolate this flow with a straight chord
     toward the known target. This 'Geodesic Blend' keeps computational complexity
     strictly O(N) while preserving topological directionality.
     The geodesic is a deterministic trajectory that avoids shortcuts through
     high-curvature barriers.

  4. WARP (Lorentz Boost + Commutator Groups)
     On plateaus (||grad L|| ~= 0), the Euclidean metric is degenerate.
     Lorentz Boost Analogy: In Minkowski spacetime Cl(p,1), a boost beta along
     direction e_hat contracts proper length by gamma = 1/sqrt(1 - beta^2), allowing
     faster traversal of flat regions. Applied to optimization:
       - Identify plateau direction u_hat = grad^2 L * e_min (smallest curvature dir)
       - Apply effective metric G'_ij = G_ij + (gamma-1) * u_hat_i * u_hat_j
       - This warps step size: large steps along plateau, small in sharp valleys
     Commutator Analysis: Gradient update groups [A, B] = d_A(d_B L) - d_B(d_A L)
       - If [A,B] ~= 0: updates to A and B are independent -- update simultaneously
       - If [A,B] large: sequential update required to avoid oscillation
       - This gives a natural layer coloring for parallel vs sequential passes
     Dynamic Signature (Cl(p,q) -> Cl(p+k, q-k)):
       - Converting positive-definite dimensions to indefinite at a local minimum
         can turn it into a saddle point (adding imaginary directions)
       - Hypothesis: anneal signature during training to escape local basins

  5. EXECUTION
     Once topology is mapped and geodesic is found:
       - Sprint along the geodesic using commutator-grouped parallel updates
       - Apply Lorentz warp on plateau segments
       - Stop probing, ignore curvature; just follow the path
"""

from __future__ import annotations

import sys
import os
import re
import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from layers import RotorLayer, MultiRotorLayer, CliffordLinear, CliffordLayerNorm, CliffordModule
from functional.activation import GeometricGELU
from optimizers.riemannian import (
    RiemannianAdam, ExponentialSGD, group_parameters_by_manifold,
    MANIFOLD_SPIN, MANIFOLD_SPHERE, MANIFOLD_EUCLIDEAN,
)
from models.blocks.gbn import GeometricBladeNetwork
from core.analysis import (
    StatisticalSampler, SamplingConfig,
    EffectiveDimensionAnalyzer,
    DimensionLifter,
    SpectralAnalyzer,
    SymmetryDetector,
    CommutatorAnalyzer as CoreCommutatorAnalyzer,
    GeodesicFlow,
)
from core.analysis._types import (
    DimensionResult, SpectralResult, SymmetryResult, CommutatorResult,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


# ======================================================================
# Core Data Structures
# ======================================================================

class CriticalPointType(Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SADDLE = "saddle"
    UNKNOWN = "unknown"


@dataclass
class CriticalPoint:
    """A detected critical point in the loss landscape."""
    params: torch.Tensor          # Parameter vector at critical point
    loss: float                   # Loss value
    morse_index: int              # Number of negative Hessian eigenvalues
    eigenvalues: torch.Tensor     # Hessian eigenvalues (sorted ascending)
    point_type: CriticalPointType
    step: int = 0                 # Training step when detected

    @classmethod
    def from_hessian(cls, params: torch.Tensor, loss: float,
                     eigenvalues: torch.Tensor, step: int = 0) -> "CriticalPoint":
        n_neg = (eigenvalues < 0).sum().item()
        n_pos = (eigenvalues > 0).sum().item()
        if n_neg == 0:
            ptype = CriticalPointType.MINIMUM
        elif n_pos == 0:
            ptype = CriticalPointType.MAXIMUM
        else:
            ptype = CriticalPointType.SADDLE
        return cls(
            params=params.clone(),
            loss=loss,
            morse_index=int(n_neg),
            eigenvalues=eigenvalues,
            point_type=ptype,
            step=step,
        )

    def __repr__(self) -> str:
        return (f"CriticalPoint({self.point_type.value}, "
                f"loss={self.loss:.4f}, index={self.morse_index}, "
                f"step={self.step})")


@dataclass
class LandscapeMap:
    """Accumulated topology map of the loss landscape."""
    critical_points: List[CriticalPoint] = field(default_factory=list)
    curvature_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    plateau_episodes: List[Tuple[int, int]] = field(default_factory=list)
    commutator_scores: Dict[str, float] = field(default_factory=dict)

    def add_critical(self, cp: CriticalPoint):
        self.critical_points.append(cp)

    def lower_minima(self, current_loss: float) -> List[CriticalPoint]:
        """Return known minima with loss lower than current."""
        return [cp for cp in self.critical_points
                if cp.point_type == CriticalPointType.MINIMUM
                and cp.loss < current_loss]

    def summary(self) -> str:
        n_min = sum(1 for c in self.critical_points if c.point_type == CriticalPointType.MINIMUM)
        n_sad = sum(1 for c in self.critical_points if c.point_type == CriticalPointType.SADDLE)
        return (f"LandscapeMap: {n_min} minima, {n_sad} saddles | "
                f"{len(self.plateau_episodes)} plateau episodes")


@dataclass
class GDOConfig:
    """Centralized configuration for the Geometric Deterministic Optimizer."""
    lr: float = 1e-3
    probe_interval: int = 50
    topology_interval: int = 200
    sprint_after: int = 500
    max_navigate_steps: int = 150
    lift_patience: int = 80
    lift_sigma: float = 0.05
    lift_k: int = 6
    lorentz_max_beta: float = 0.95
    commutator_threshold: float = 0.3
    # Geometric controller params
    fim_damping: float = 1e-4
    closure_trust_threshold: float = 0.1
    coherence_gate: float = 0.3
    entropy_exploration_threshold: float = 0.7
    # Parameter grouping
    grouping_strategy: str = "geometric"  # "geometric" or "module"
    min_group_size: int = 4
    max_groups: int = 16
    # Coloring algorithm
    dsatur_enabled: bool = True
    color_conflict_budget: float = 0.5
    manifold_compat_constraint: bool = True
    # Interaction estimation
    interaction_estimation: str = "efficient"  # "efficient", "fd", "gradient_only"
    grad_cosine_threshold: float = 0.1
    # Adaptive rescheduling
    adaptive_reschedule: bool = True
    reschedule_interval: int = 50
    reschedule_loss_delta: float = 0.2
    reschedule_grad_kl_threshold: float = 0.5


# ======================================================================
# Experiment Infrastructure
# ======================================================================

@dataclass
class ExperimentResult:
    """Collected results from one optimizer run."""
    name: str
    optimizer_name: str
    losses: List[float]
    wall_times: List[float]
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    final_loss: float = 0.0
    total_wall_time: float = 0.0
    gdo_diagnostics: Optional[Dict] = None
    bivector_norms: Optional[List[float]] = None
    mode_history: Optional[List[str]] = None


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    category: str
    steps: int
    lr: float
    seed: int = 42
    device: str = 'cpu'
    algebra_sig: Optional[Tuple[int, int]] = None
    gdo_config: Optional[GDOConfig] = None


EXPERIMENT_REGISTRY: Dict[str, Tuple[Callable, str]] = {}


def register_experiment(name: str, category: str):
    """Decorator for registering experiment functions."""
    def decorator(fn):
        EXPERIMENT_REGISTRY[name] = (fn, category)
        return fn
    return decorator


# ======================================================================
# GDO Sub-Components (unchanged from original)
# ======================================================================

class LandscapeTopologySearch:
    """Hessian-based critical point detector using Lanczos iteration.

    Uses Lanczos iteration for large parameter spaces to approximate
    leading eigenvalues without forming the full Hessian.

    A point theta is critical if ||grad L(theta)|| < eps_grad.
    Classification by Morse index = number of negative eigenvalues of H = d^2 L / d theta^2.
    """

    def __init__(
        self,
        loss_fn: Callable,
        grad_tol: float = 1e-3,
        fd_step: float = 1e-4,
        lanczos_steps: int = 20,
        detect_every: int = 100,
    ):
        self.loss_fn = loss_fn
        self.grad_tol = grad_tol
        self.fd_step = fd_step
        self.lanczos_steps = lanczos_steps
        self.detect_every = detect_every
        self._step = 0
        self._last_eigenvalues: Optional[torch.Tensor] = None
        self._last_eigenvecs: Optional[torch.Tensor] = None

    def _flat_grad(self, params: List[torch.Tensor]) -> torch.Tensor:
        grads = []
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.detach().reshape(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    def _flat_params(self, params: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in params])

    def _hessian_vector_product(
        self, loss: torch.Tensor, params: List[torch.Tensor], v: torch.Tensor
    ) -> torch.Tensor:
        """Compute H*v via double backprop."""
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        flat_grad = torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
            for g, p in zip(grads, params)
        ])
        gv = (flat_grad * v.detach()).sum()
        hvp = torch.autograd.grad(gv, params, allow_unused=True)
        return torch.cat([
            h.reshape(-1) if h is not None else torch.zeros(p.numel(), device=p.device)
            for h, p in zip(hvp, params)
        ])

    def _lanczos_eigenvalues(
        self, loss: torch.Tensor, params: List[torch.Tensor], k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate bottom-k eigenvalues and eigenvectors via Lanczos iteration."""
        n = sum(p.numel() for p in params)
        k = min(k, n)
        device = params[0].device

        alpha = []
        beta = [0.0]
        q_prev = torch.zeros(n, device=device)
        q_curr = torch.randn(n, device=device)
        q_curr = q_curr / q_curr.norm()
        Q = [q_curr]

        for j in range(k):
            Hq = self._hessian_vector_product(loss, params, q_curr)
            a = (q_curr * Hq).sum().item()
            alpha.append(a)

            if j == k - 1:
                break

            r = Hq - a * q_curr - beta[-1] * q_prev
            b = r.norm().item()
            beta.append(b)

            if b < 1e-10:
                break

            q_prev = q_curr
            q_curr = r / b
            for qv in Q:
                q_curr = q_curr - (q_curr * qv).sum() * qv
            q_curr = q_curr / (q_curr.norm() + 1e-12)
            Q.append(q_curr)

        m = len(alpha)
        T = torch.zeros(m, m, device=device)
        for i, a in enumerate(alpha):
            T[i, i] = a
        for i, b in enumerate(beta[1:min(len(beta), m)]):
            T[i, i+1] = b
            T[i+1, i] = b

        eigvals, eigvecs_T = torch.linalg.eigh(T)
        Q_mat = torch.stack(Q[:m])
        H_vecs = F.normalize(Q_mat.T @ eigvecs_T, dim=0)
        return eigvals.detach(), H_vecs.detach()

    def check(
        self, loss: torch.Tensor, params: List[torch.Tensor], step: int
    ) -> Optional[CriticalPoint]:
        """Check if current point is a critical point. Returns CriticalPoint or None."""
        self._step = step
        if step % self.detect_every != 0:
            return None

        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat_g = torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
            for g, p in zip(grads, params)
        ])
        grad_norm = flat_g.norm().item()

        if grad_norm > self.grad_tol * 10:
            return None

        flat_p = self._flat_params(params)
        try:
            eigenvalues, eigenvecs = self._lanczos_eigenvalues(
                loss, params, k=min(20, flat_p.numel())
            )
            self._last_eigenvalues = eigenvalues
            self._last_eigenvecs = eigenvecs
            cp = CriticalPoint.from_hessian(flat_p, loss.item(), eigenvalues, step)
            return cp
        except Exception:
            return None


class CurvatureProbe:
    """Deploys lightweight probes to measure local geometry."""

    def __init__(
        self,
        probe_radius: float = 1e-3,
        n_directions: int = 8,
        plateau_threshold: float = 1e-4,
    ):
        self.probe_radius = probe_radius
        self.n_directions = n_directions
        self.plateau_threshold = plateau_threshold

    @dataclass
    class ProbeResult:
        mean_curvature: float
        anisotropy: float
        plateau_score: float
        min_curvature_dir: torch.Tensor
        max_curvature_dir: torch.Tensor
        grad_norm: float

    def probe(
        self,
        loss_fn: Callable[[], torch.Tensor],
        params: List[torch.Tensor],
    ) -> "CurvatureProbe.ProbeResult":
        """Sample curvature via finite differences on the parameter sphere."""
        n = sum(p.numel() for p in params)
        device = params[0].device

        dirs = torch.randn(self.n_directions, n, device=device)
        dirs = F.normalize(dirs, dim=-1)
        for i in range(1, min(self.n_directions, 4)):
            for j in range(i):
                dirs[i] -= (dirs[i] * dirs[j]).sum() * dirs[j]
            dirs[i] = F.normalize(dirs[i], dim=0)

        orig = [p.data.clone() for p in params]

        curvatures = []
        with torch.no_grad():
            loss_0 = loss_fn().item()

            for d in dirs:
                offset = d * self.probe_radius
                idx = 0
                for p in params:
                    sz = p.numel()
                    p.data += offset[idx:idx+sz].reshape(p.shape)
                    idx += sz
                loss_plus = loss_fn().item()

                for p, o in zip(params, orig):
                    p.data.copy_(o)

                idx = 0
                for p in params:
                    sz = p.numel()
                    p.data -= offset[idx:idx+sz].reshape(p.shape)
                    idx += sz
                loss_minus = loss_fn().item()

                k = (loss_plus - 2 * loss_0 + loss_minus) / (self.probe_radius ** 2)
                curvatures.append(k)

                for p, o in zip(params, orig):
                    p.data.copy_(o)

        curvatures = torch.tensor(curvatures, device=device)
        mean_k = curvatures.mean().item()
        anisotropy = (curvatures.std() / (curvatures.abs().mean() + 1e-8)).item()
        plateau_score = (curvatures.abs() < self.plateau_threshold).float().mean().item()

        min_idx = curvatures.argmin()
        max_idx = curvatures.argmax()
        min_dir = dirs[min_idx]
        max_dir = dirs[max_idx]

        loss_val = loss_fn()
        try:
            grads = torch.autograd.grad(loss_val, params, allow_unused=True)
            g_flat = torch.cat([
                g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
                for g, p in zip(grads, params)
            ])
            grad_norm = g_flat.norm().item()
        except Exception:
            grad_norm = 0.0

        return CurvatureProbe.ProbeResult(
            mean_curvature=mean_k,
            anisotropy=anisotropy,
            plateau_score=plateau_score,
            min_curvature_dir=min_dir,
            max_curvature_dir=max_dir,
            grad_norm=grad_norm,
        )


class GeodesicIntegrator:
    """Approximates geodesic trajectories in parameter space."""

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        n_integration_steps: int = 5,
        geodesic_weight: float = 0.3,
    ):
        self.lambda_reg = lambda_reg
        self.n_integration_steps = n_integration_steps
        self.geodesic_weight = geodesic_weight
        self._target: Optional[torch.Tensor] = None
        self._target_loss: Optional[float] = None

    def set_target(self, target_params: torch.Tensor, target_loss: float):
        self._target = target_params.clone()
        self._target_loss = target_loss

    def natural_gradient_step(
        self,
        flat_grad: torch.Tensor,
        hessian_diag: torch.Tensor,
    ) -> torch.Tensor:
        metric_inv = 1.0 / (hessian_diag.abs() + self.lambda_reg)
        return metric_inv * flat_grad

    def geodesic_blend(
        self,
        flat_params: torch.Tensor,
        natural_step: torch.Tensor,
        lr: float,
    ) -> torch.Tensor:
        step = -lr * natural_step

        if self._target is not None:
            chord = self._target - flat_params
            chord_norm = chord.norm()
            if chord_norm > 1e-8:
                chord_dir = chord / chord_norm
                step_along = (step * chord_dir).sum() * chord_dir
                step_perp = step - step_along
                step = step_perp + step_along + self.geodesic_weight * lr * chord_dir

        return step


class LorentzWarpOptimizer:
    """Applies Lorentz-boost-inspired metric warping for plateau escape."""

    def __init__(
        self,
        plateau_grad_thresh: float = 1e-3,
        plateau_curvature_thresh: float = 0.7,
        max_beta: float = 0.95,
        beta_increment: float = 0.05,
        beta_decay: float = 0.5,
    ):
        self.plateau_grad_thresh = plateau_grad_thresh
        self.plateau_curvature_thresh = plateau_curvature_thresh
        self.max_beta = max_beta
        self.beta_increment = beta_increment
        self.beta_decay = beta_decay
        self._beta = 0.0
        self._on_plateau = False
        self._plateau_steps = 0
        self._boost_dir: Optional[torch.Tensor] = None

    @property
    def gamma(self) -> float:
        return 1.0 / math.sqrt(max(1.0 - self._beta ** 2, 1e-6))

    def update(
        self,
        grad_norm: float,
        plateau_score: float,
        min_curvature_dir: torch.Tensor,
    ) -> bool:
        on_plateau = (
            grad_norm < self.plateau_grad_thresh
            and plateau_score > self.plateau_curvature_thresh
        )

        if on_plateau:
            self._on_plateau = True
            self._plateau_steps += 1
            self._beta = min(self._beta + self.beta_increment, self.max_beta)
            self._boost_dir = min_curvature_dir
        else:
            if self._on_plateau:
                self._on_plateau = False
                self._plateau_steps = 0
            self._beta = max(self._beta - self.beta_decay * self._beta, 0.0)

        return on_plateau

    def warped_lr(self, lr: float, flat_grad_shape: int, device: torch.device) -> torch.Tensor:
        lr_vec = torch.ones(flat_grad_shape, device=device) * lr

        if self._on_plateau and self._boost_dir is not None and self._beta > 0.01:
            g = self.gamma
            d = self._boost_dir
            if d.shape[0] == flat_grad_shape:
                boost_factor = 1.0 + (g - 1.0) * d ** 2
                lr_vec = lr_vec * boost_factor

        return lr_vec


class GeometricParameterController:
    """Geometrically-verified partial parameter updates via commutator coloring."""

    def __init__(
        self,
        algebra: Optional[CliffordAlgebra] = None,
        commutator_threshold: float = 0.3,
        fim_damping: float = 1e-4,
        closure_trust_threshold: float = 0.1,
        coherence_gate: float = 0.3,
        entropy_exploration_threshold: float = 0.7,
        fd_step: float = 1e-3,
        config: Optional[GDOConfig] = None,
    ):
        self.algebra = algebra
        self.commutator_threshold = commutator_threshold
        self.fim_damping = fim_damping
        self.closure_trust_threshold = closure_trust_threshold
        self.coherence_gate = coherence_gate
        self.entropy_exploration_threshold = entropy_exploration_threshold
        self.fd_step = fd_step
        self.config = config or GDOConfig()

        self.core_comm: Optional[CoreCommutatorAnalyzer] = None
        self.spectral: Optional[SpectralAnalyzer] = None
        self.geodesic: Optional[GeodesicFlow] = None
        if algebra is not None and algebra.n >= 2:
            self.core_comm = CoreCommutatorAnalyzer(algebra)
            self.spectral = SpectralAnalyzer(algebra)
            self.geodesic = GeodesicFlow(algebra, k=8)

    @staticmethod
    def _flat_group_grad(
        loss_fn: Callable[[], torch.Tensor],
        group: List[torch.nn.Parameter],
        device: torch.device,
    ) -> torch.Tensor:
        loss = loss_fn()
        grads = torch.autograd.grad(loss, group, allow_unused=True)
        return torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
            for g, p in zip(grads, group)
        ])

    def compute_fim_diagonal(
        self,
        loss_fn: Callable[[], torch.Tensor],
        model: nn.Module,
        param_groups: List[List[nn.Parameter]],
        n_samples: int = 10,
    ) -> Dict[int, torch.Tensor]:
        device = next(model.parameters()).device
        fim: Dict[int, torch.Tensor] = {}
        for g_idx, group in enumerate(param_groups):
            n_params = sum(p.numel() for p in group)
            accum = torch.zeros(n_params, device=device)
            for _ in range(n_samples):
                try:
                    g = self._flat_group_grad(loss_fn, group, device)
                    accum += g * g
                except Exception:
                    pass
            fim[g_idx] = accum / max(n_samples, 1)
        return fim

    @staticmethod
    def _extract_mv_params(model: nn.Module) -> Optional[torch.Tensor]:
        mv_list: List[torch.Tensor] = []
        for m in model.modules():
            if isinstance(m, RotorLayer):
                bv = m.grade_weights.detach()
                full = torch.zeros(
                    bv.shape[0], m.algebra.dim,
                    device=bv.device, dtype=bv.dtype,
                )
                full[:, m.grade_indices] = bv
                mv_list.append(full)
            elif isinstance(m, MultiRotorLayer):
                bv = m.rotor_grade_weights.detach()
                full = torch.zeros(
                    bv.shape[0], m.algebra.dim,
                    device=bv.device, dtype=bv.dtype,
                )
                full[:, m.grade_indices] = bv
                mv_list.append(full)
        if not mv_list:
            return None
        return torch.cat(mv_list, dim=0)

    def compute_geometric_scores(self, model: nn.Module) -> Dict:
        if self.core_comm is None:
            return {}

        mv_params = self._extract_mv_params(model)
        if mv_params is None or mv_params.shape[0] < 2:
            return {}

        result: Dict = {}

        try:
            comm_result = self.core_comm.analyze(mv_params)
            result["comm_result"] = comm_result
            result["closure_error"] = comm_result.lie_bracket_structure.get(
                "closure_error", 1.0
            )
            result["mean_commutator_norm"] = comm_result.mean_commutator_norm
        except Exception:
            pass

        if self.geodesic is not None and mv_params.shape[0] >= 3:
            try:
                k_actual = min(self.geodesic.k, mv_params.shape[0] - 1)
                gf = GeodesicFlow(self.algebra, k=k_actual)
                result["coherence"] = gf.coherence(mv_params)
                result["per_point_coherence"] = gf.per_point_coherence(mv_params)
            except Exception:
                pass

        if self.spectral is not None:
            try:
                grade_energy = self.spectral.grade_energy_spectrum(
                    mv_params.unsqueeze(1)
                )
                result["grade_energy"] = grade_energy
                ge = grade_energy.clamp(min=0)
                total = ge.sum()
                if total > 1e-8:
                    probs = ge / total
                    entropy = -(probs * (probs + 1e-12).log()).sum().item()
                    max_ent = math.log(len(probs))
                    result["grade_entropy"] = entropy / max_ent if max_ent > 0 else 0.0
                else:
                    result["grade_entropy"] = 0.0
            except Exception:
                pass

        return result

    def _fd_cross_hessian(
        self,
        loss_fn: Callable[[], torch.Tensor],
        param_groups: List[List[nn.Parameter]],
    ) -> Dict[Tuple[int, int], float]:
        if not param_groups:
            return {}

        n = len(param_groups)
        device = param_groups[0][0].device

        baseline: List[torch.Tensor] = []
        for g in param_groups:
            try:
                baseline.append(self._flat_group_grad(loss_fn, g, device))
            except Exception:
                baseline.append(torch.zeros(
                    sum(p.numel() for p in g), device=device
                ))

        orig = {id(p): p.data.clone() for g in param_groups for p in g}
        scores: Dict[Tuple[int, int], float] = {
            (i, j): 0.0 for i in range(n) for j in range(i + 1, n)
        }

        for i in range(n):
            g_i_norm = baseline[i].norm().item()
            if g_i_norm < 1e-10:
                continue

            step_i = baseline[i] / g_i_norm * self.fd_step
            ptr = 0
            for p in param_groups[i]:
                sz = p.numel()
                p.data -= step_i[ptr:ptr + sz].reshape(p.shape)
                ptr += sz

            for j in range(n):
                if j == i:
                    continue
                key = (min(i, j), max(i, j))
                try:
                    g_j_new = self._flat_group_grad(loss_fn, param_groups[j], device)
                    delta = (g_j_new - baseline[j]).norm().item()
                    g_j_norm = baseline[j].norm().item()
                    scores[key] = max(scores[key], delta / (g_j_norm + 1e-8))
                except Exception:
                    pass

            for p in param_groups[i]:
                p.data.copy_(orig[id(p)])

        return scores

    def build_hybrid_scores(
        self,
        loss_fn: Callable[[], torch.Tensor],
        param_groups: List[List[nn.Parameter]],
        geometric_scores: Dict,
    ) -> Dict[Tuple[int, int], float]:
        fd_scores = self._fd_cross_hessian(loss_fn, param_groups)

        if "comm_result" not in geometric_scores:
            return fd_scores

        alg_matrix = geometric_scores["comm_result"].commutativity_matrix
        alg_n = alg_matrix.shape[0]

        for (i, j), fd_score in fd_scores.items():
            ai, aj = min(i, alg_n - 1), min(j, alg_n - 1)
            alg_score = alg_matrix[ai, aj].item()
            alg_max = alg_matrix.max().item()
            if alg_max > 1e-8:
                alg_score /= alg_max
            fd_scores[(i, j)] = 0.6 * fd_score + 0.4 * alg_score

        return fd_scores

    def build_hybrid_scores_efficient(
        self,
        loss_fn: Callable[[], torch.Tensor],
        param_groups: List[List[nn.Parameter]],
        geometric_scores: Dict,
    ) -> Dict[Tuple[int, int], float]:
        """Cheaper interaction estimation using gradient cosine + selective HVP."""
        n = len(param_groups)
        if n < 2:
            return {}

        device = param_groups[0][0].device
        scores: Dict[Tuple[int, int], float] = {
            (i, j): 0.0 for i in range(n) for j in range(i + 1, n)
        }

        # Tier 0: algebraic scores from commutativity matrix (free)
        alg_scores: Dict[Tuple[int, int], float] = {}
        if "comm_result" in geometric_scores:
            alg_matrix = geometric_scores["comm_result"].commutativity_matrix
            alg_n = alg_matrix.shape[0]
            alg_max = alg_matrix.max().item()
            for i in range(n):
                for j in range(i + 1, n):
                    ai, aj = min(i, alg_n - 1), min(j, alg_n - 1)
                    val = alg_matrix[ai, aj].item()
                    if alg_max > 1e-8:
                        val /= alg_max
                    alg_scores[(i, j)] = val

        # Tier 1: gradient-based interaction via norm sensitivity
        # Compute per-group gradient vectors
        group_grads: List[torch.Tensor] = []
        for g in param_groups:
            try:
                group_grads.append(self._flat_group_grad(loss_fn, g, device))
            except Exception:
                group_grads.append(torch.zeros(
                    sum(p.numel() for p in g), device=device
                ))

        # Measure interaction via gradient norm correlation:
        # For each pair (i,j), compare how similar the gradient norms
        # are across components — high correlation = potential coupling
        group_norms = torch.tensor(
            [g.norm().item() for g in group_grads], device=device,
        )

        cosine_scores: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            gi_norm = group_norms[i].item()
            if gi_norm < 1e-10:
                continue
            # Perturb group i's params slightly, measure grad change in j
            gi_dir = group_grads[i] / (gi_norm + 1e-8)
            orig_data = {id(p): p.data.clone() for p in param_groups[i]}
            step = gi_dir * self.fd_step
            ptr = 0
            for p in param_groups[i]:
                sz = p.numel()
                p.data -= step[ptr:ptr + sz].reshape(p.shape)
                ptr += sz

            for j in range(i + 1, n):
                try:
                    gj_new = self._flat_group_grad(loss_fn, param_groups[j], device)
                    delta = (gj_new - group_grads[j]).norm().item()
                    gj_norm = group_norms[j].item()
                    cosine_scores[(i, j)] = max(
                        cosine_scores.get((i, j), 0.0),
                        delta / (gj_norm + 1e-8),
                    )
                except Exception:
                    pass

            for p in param_groups[i]:
                p.data.copy_(orig_data[id(p)])

        # Blend Tier 0 (algebraic) + Tier 1 (FD cross-sensitivity)
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                alg = alg_scores.get(key, 0.0)
                fd = cosine_scores.get(key, 0.0)
                scores[key] = 0.4 * alg + 0.6 * fd

        return scores

    def parallel_groups(
        self, scores: Dict[Tuple[int, int], float], n_groups: int
    ) -> List[List[int]]:
        conflicts = {i: set() for i in range(n_groups)}
        for (i, j), s in scores.items():
            if s > self.commutator_threshold:
                conflicts[i].add(j)
                conflicts[j].add(i)

        colors = [-1] * n_groups
        for i in range(n_groups):
            used = {colors[c] for c in conflicts[i] if colors[c] >= 0}
            color = 0
            while color in used:
                color += 1
            colors[i] = color

        n_colors = max(colors) + 1 if colors else 1
        schedule = [[] for _ in range(n_colors)]
        for i, c in enumerate(colors):
            schedule[c].append(i)
        return schedule

    def parallel_groups_dsatur(
        self,
        scores: Dict[Tuple[int, int], float],
        n_groups: int,
        group_meta: Optional[List[Dict]] = None,
    ) -> List[List[int]]:
        """DSatur coloring with soft conflict budget and manifold constraints."""
        if n_groups <= 1:
            return [[i] for i in range(n_groups)]

        budget = self.config.color_conflict_budget
        use_manifold = (self.config.manifold_compat_constraint
                        and group_meta is not None)

        # Build weighted adjacency
        adj: Dict[int, Dict[int, float]] = {i: {} for i in range(n_groups)}
        for (i, j), w in scores.items():
            if w > 0:
                adj[i][j] = w
                adj[j][i] = w

        # DSatur state
        colors = [-1] * n_groups
        # Per-node: set of distinct colors in neighborhood
        neighbor_colors: List[set] = [set() for _ in range(n_groups)]
        # Per-color: list of node indices and total weight per node
        color_members: Dict[int, List[int]] = {}

        for _ in range(n_groups):
            # Pick uncolored node with highest saturation, break ties by
            # weighted degree to colored neighbors
            best_node = -1
            best_sat = -1
            best_wdeg = -1.0
            for node in range(n_groups):
                if colors[node] >= 0:
                    continue
                sat = len(neighbor_colors[node])
                wdeg = sum(
                    adj[node].get(nb, 0.0)
                    for nb in range(n_groups)
                    if colors[nb] >= 0
                )
                if (sat > best_sat
                        or (sat == best_sat and wdeg > best_wdeg)):
                    best_node = node
                    best_sat = sat
                    best_wdeg = wdeg
            if best_node < 0:
                break

            # Find valid color for best_node
            node_manifold = (group_meta[best_node].get("manifold")
                             if use_manifold else None)
            assigned_color = -1
            # Try existing colors first
            for c in sorted(color_members.keys()):
                # Hard conflict: no edge above threshold
                has_hard_conflict = any(
                    adj[best_node].get(m, 0.0) > self.commutator_threshold
                    for m in color_members[c]
                )
                if has_hard_conflict:
                    continue
                # Soft budget: total weight within color
                total_weight = sum(
                    adj[best_node].get(m, 0.0) for m in color_members[c]
                )
                if total_weight > budget:
                    continue
                # Manifold compatibility
                if use_manifold:
                    compat = all(
                        group_meta[m].get("manifold") == node_manifold
                        or adj[best_node].get(m, 0.0) == 0.0
                        for m in color_members[c]
                    )
                    if not compat:
                        continue
                assigned_color = c
                break

            if assigned_color < 0:
                # New color needed
                assigned_color = len(color_members)

            colors[best_node] = assigned_color
            color_members.setdefault(assigned_color, []).append(best_node)

            # Update saturation of uncolored neighbors
            for nb in adj[best_node]:
                if colors[nb] < 0:
                    neighbor_colors[nb].add(assigned_color)

        n_colors = max(colors) + 1 if any(c >= 0 for c in colors) else 1
        schedule: List[List[int]] = [[] for _ in range(n_colors)]
        for i, c in enumerate(colors):
            if c >= 0:
                schedule[c].append(i)
        return schedule

    def compute_group_scales(
        self,
        param_groups: List[List[nn.Parameter]],
        fim_diag: Dict[int, torch.Tensor],
        geometric_scores: Dict,
    ) -> List[float]:
        scales = []
        for g_idx in range(len(param_groups)):
            fim_g = fim_diag.get(g_idx)
            if fim_g is not None and fim_g.numel() > 0:
                fim_sensitivity = fim_g.mean().item()
                fim_scale = 1.0 / (1.0 + fim_sensitivity / self.fim_damping)
            else:
                fim_scale = 1.0

            closure_err = geometric_scores.get("closure_error", 0.5)
            if closure_err < self.closure_trust_threshold:
                closure_scale = 1.5
            elif closure_err > 0.5:
                closure_scale = 0.5
            else:
                closure_scale = 1.0

            coherence = geometric_scores.get("coherence", 0.5)
            coherence_scale = max(0.3, min(1.0, coherence / self.coherence_gate))

            entropy = geometric_scores.get("grade_entropy", 0.5)
            if entropy > self.entropy_exploration_threshold:
                entropy_scale = 1.2
            else:
                entropy_scale = 0.8

            scale = fim_scale * closure_scale * coherence_scale * entropy_scale
            scale = max(0.1, min(2.0, scale))
            scales.append(scale)

        return scales

    def analyze_and_schedule(
        self,
        model: nn.Module,
        loss_fn: Callable[[], torch.Tensor],
        param_groups: List[List[nn.Parameter]],
        group_meta: Optional[List[Dict]] = None,
    ) -> Tuple[List[List[int]], List[float], Dict]:
        fim_diag = self.compute_fim_diagonal(loss_fn, model, param_groups)
        geo_scores = self.compute_geometric_scores(model)

        # Select interaction estimation strategy
        if self.config.interaction_estimation == "efficient":
            hybrid_scores = self.build_hybrid_scores_efficient(
                loss_fn, param_groups, geo_scores,
            )
        elif self.config.interaction_estimation == "gradient_only":
            hybrid_scores = self.build_hybrid_scores_efficient(
                loss_fn, param_groups, {},
            )
        else:
            hybrid_scores = self.build_hybrid_scores(
                loss_fn, param_groups, geo_scores,
            )

        # Select coloring algorithm
        if self.config.dsatur_enabled:
            schedule = self.parallel_groups_dsatur(
                hybrid_scores, len(param_groups), group_meta,
            )
        else:
            schedule = self.parallel_groups(hybrid_scores, len(param_groups))

        scales = self.compute_group_scales(param_groups, fim_diag, geo_scores)

        diagnostics = {
            "fim_diag": fim_diag,
            "geometric_scores": geo_scores,
            "hybrid_scores": hybrid_scores,
            "schedule": schedule,
            "scales": scales,
        }
        return schedule, scales, diagnostics


class DimensionalLiftOracle:
    """Escape local minima via lift -> oracle search -> pull-down."""

    def __init__(
        self,
        k: int = 6,
        oracle_steps: int = 50,
        oracle_lr: float = 2e-3,
        lift_sigma: float = 0.05,
        max_sigma: float = 3.0,
        sigma_scale: float = 2.0,
        patience: int = 80,
        check_every: int = 10,
        min_improvement: float = 1e-4,
        accept_blend: float = 1.0,
    ):
        self.k = k
        self.oracle_steps = oracle_steps
        self.oracle_lr = oracle_lr
        self.lift_sigma = lift_sigma
        self.max_sigma = max_sigma
        self.sigma_scale = sigma_scale
        self.patience = patience
        self.check_every = check_every
        self.min_improvement = min_improvement
        self.accept_blend = accept_blend

        self._best_loss = float('inf')
        self._steps_no_improve = 0
        self._lift_count = 0
        self._consecutive_fails = 0
        self._current_sigma = lift_sigma

    def should_lift(self, current_loss: float, step: int) -> bool:
        if step % self.check_every != 0:
            return False
        if current_loss < self._best_loss - self.min_improvement:
            self._best_loss = current_loss
            self._steps_no_improve = 0
            return False
        self._steps_no_improve += self.check_every
        return self._steps_no_improve >= self.patience

    @staticmethod
    def _get_flat(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in model.parameters()])

    @staticmethod
    def _set_flat(model: nn.Module, flat: torch.Tensor):
        idx = 0
        for p in model.parameters():
            sz = p.numel()
            p.data.copy_(flat[idx:idx + sz].reshape(p.shape))
            idx += sz

    @staticmethod
    def _flat_grad(model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        return torch.cat([
            p.grad.reshape(-1) if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in model.parameters()
        ])

    @staticmethod
    def _compute_bottom_eigvecs(
        loss_fn: Callable[[], torch.Tensor],
        model: nn.Module,
        k: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = list(model.parameters())
        n = sum(p.numel() for p in params)
        device = next(model.parameters()).device
        k = min(k, n)

        loss = loss_fn()
        alpha, beta = [], [0.0]
        q_prev = torch.zeros(n, device=device)
        q_curr = F.normalize(torch.randn(n, device=device), dim=0)
        Q = [q_curr.clone()]

        for j in range(k):
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            flat_g = torch.cat([
                g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
                for g, p in zip(grads, params)
            ])
            gv = (flat_g * q_curr.detach()).sum()
            hvp = torch.autograd.grad(gv, params, allow_unused=True)
            Hq = torch.cat([
                h.reshape(-1) if h is not None else torch.zeros(p.numel(), device=device)
                for h, p in zip(hvp, params)
            ])

            a = (q_curr * Hq).sum().item()
            alpha.append(a)

            if j == k - 1:
                break

            r = Hq - a * q_curr - beta[-1] * q_prev
            b = r.norm().item()
            beta.append(b)
            if b < 1e-10:
                break

            q_prev = q_curr
            q_curr = r / b
            for qv in Q:
                q_curr = q_curr - (q_curr * qv).sum() * qv
            q_curr = q_curr / (q_curr.norm() + 1e-12)
            Q.append(q_curr.clone())

        m = len(alpha)
        T = torch.zeros(m, m, device=device)
        for i, a in enumerate(alpha):
            T[i, i] = a
        for i, b in enumerate(beta[1:min(len(beta), m)]):
            T[i, i+1] = b
            T[i+1, i] = b

        eigvals, eigvecs_T = torch.linalg.eigh(T)
        Q_mat = torch.stack(Q[:m])
        H_vecs = F.normalize(Q_mat.T @ eigvecs_T, dim=0)
        return eigvals.detach(), H_vecs.detach()

    def lift_and_search(
        self,
        model: nn.Module,
        loss_fn: Callable[[], torch.Tensor],
        current_loss: float,
        probe_result=None,
        hessian_vecs: Optional[torch.Tensor] = None,
        hessian_vals: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], float]:
        self._lift_count += 1
        self._steps_no_improve = 0

        flat_orig = self._get_flat(model)
        n = flat_orig.shape[0]
        device = flat_orig.device

        sigma = self._current_sigma
        scaled_lr = self.oracle_lr * max(1.0, sigma / self.lift_sigma)

        print(f"  [LiftOracle] #{self._lift_count}: sigma={sigma:.4f}  "
              f"oracle_lr={scaled_lr:.5f}  loss={current_loss:.5f}")

        directions: List[torch.Tensor] = []

        if hessian_vecs is None:
            try:
                hessian_vals, hessian_vecs = self._compute_bottom_eigvecs(
                    loss_fn, model, k=min(4, n)
                )
            except Exception:
                hessian_vecs = None
                hessian_vals = None

        if hessian_vecs is not None and hessian_vals is not None:
            ev = hessian_vecs.to(device)
            vals = hessian_vals.to(device)
            neg_idx = (vals < 0).nonzero(as_tuple=False).view(-1)
            for i in neg_idx[:2].tolist():
                v = F.normalize(ev[:, i], dim=0)
                directions.append(v)
                if len(directions) < self.k:
                    directions.append(-v)
            if len(directions) < self.k and ev.shape[1] > 0:
                directions.append(F.normalize(ev[:, 0], dim=0))

        if probe_result is not None and probe_result.min_curvature_dir.shape[0] == n:
            d_probe = F.normalize(probe_result.min_curvature_dir.to(device), dim=0)
            if len(directions) < self.k:
                directions.append(d_probe)
            if len(directions) < self.k:
                directions.append(-d_probe)

        n_anti = min(2, self.k - len(directions))
        probe_dirs = F.normalize(torch.randn(n_anti, n, device=device), dim=-1)
        for pd in probe_dirs:
            perturbed = flat_orig + sigma * pd
            self._set_flat(model, perturbed)
            model.zero_grad()
            with torch.enable_grad():
                loss_p = loss_fn()
                loss_p.backward()
            g_p = self._flat_grad(model)
            g_norm = g_p.norm()
            if g_norm > 1e-8:
                anti_g = F.normalize(-g_p, dim=0)
                directions.append(anti_g)
        self._set_flat(model, flat_orig)

        n_rand = self.k - len(directions)
        if n_rand > 0:
            R = torch.randn(n_rand, n, device=device)
            for i in range(n_rand):
                for d in directions:
                    R[i] = R[i] - (R[i] * d).sum() * d
                for j in range(i):
                    R[i] = R[i] - (R[i] * R[j]).sum() * R[j]
                norm = R[i].norm()
                if norm > 1e-8:
                    R[i] = R[i] / norm
            directions.extend([R[i] for i in range(n_rand)])

        candidates = [flat_orig + sigma * d for d in directions[:self.k]]

        best_params = None
        best_loss = current_loss
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for j, psi_init in enumerate(candidates):
            psi = psi_init.clone()
            m = torch.zeros_like(psi)
            v = torch.zeros_like(psi)

            for t in range(1, self.oracle_steps + 1):
                self._set_flat(model, psi)
                model.zero_grad()
                with torch.enable_grad():
                    loss_o = loss_fn()
                    loss_o.backward()
                g = self._flat_grad(model)

                with torch.no_grad():
                    m = beta1 * m + (1 - beta1) * g
                    v = beta2 * v + (1 - beta2) * g * g
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    psi = psi - scaled_lr * m_hat / (v_hat.sqrt() + eps)

            self._set_flat(model, psi)
            with torch.no_grad():
                final_loss = loss_fn().item()

            tag = "*" if final_loss < best_loss else " "
            print(f"  [LiftOracle]  {tag}cand {j}: loss={final_loss:.5f}")

            if final_loss < best_loss:
                best_loss = final_loss
                best_params = psi.clone()

        self._set_flat(model, flat_orig)

        if best_params is not None:
            if self.accept_blend < 1.0:
                best_params = flat_orig + self.accept_blend * (best_params - flat_orig)
            improvement = current_loss - best_loss
            print(f"  [LiftOracle] [ok] improvement={improvement:.5f} -> {best_loss:.5f}")
            self._consecutive_fails = 0
            self._current_sigma = self.lift_sigma
            return best_params, best_loss

        self._consecutive_fails += 1
        self._current_sigma = min(
            self.lift_sigma * (self.sigma_scale ** self._consecutive_fails),
            self.max_sigma,
        )
        print(f"  [LiftOracle] [fail] no improvement. "
              f"Escalating sigma: {sigma:.4f} -> {self._current_sigma:.4f}")
        return None, current_loss


@dataclass
class PreExplorationResult:
    """Output of PreExplorationAnalyzer."""
    dim_result: Optional[DimensionResult] = None
    spectral_result: Optional[SpectralResult] = None
    symmetry_result: Optional[SymmetryResult] = None
    commutator_result: Optional[CommutatorResult] = None
    landscape_coherence: float = 0.0
    landscape_curvature: float = 0.0
    loss_statistics: Dict = field(default_factory=dict)
    geometric_scores: Dict = field(default_factory=dict)
    recommended_config: GDOConfig = field(default_factory=GDOConfig)
    strategy_label: str = "EXPLORE-heavy"
    causal_report: Optional[Dict] = None
    lifting_report: Optional[Dict] = None
    landscape_losses: Optional[torch.Tensor] = None
    landscape_positions: Optional[torch.Tensor] = None
    flow_bivectors: Optional[torch.Tensor] = None
    per_point_coherence: Optional[torch.Tensor] = None


class PreExplorationAnalyzer:
    """Pre-optimization landscape analysis pipeline."""

    def __init__(
        self,
        algebra: Optional[CliffordAlgebra] = None,
        n_samples: int = 200,
        sample_radius: float = 0.5,
        device: str = 'cpu',
    ):
        self.algebra = algebra
        self.n_samples = n_samples
        self.sample_radius = sample_radius
        self.device = device

    @staticmethod
    def _get_flat(model: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in model.parameters()])

    @staticmethod
    def _set_flat(model: nn.Module, flat: torch.Tensor):
        idx = 0
        for p in model.parameters():
            sz = p.numel()
            p.data.copy_(flat[idx:idx + sz].reshape(p.shape))
            idx += sz

    def _sample_landscape(
        self, model: nn.Module, loss_fn: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        theta0 = self._get_flat(model).clone()
        n_params = theta0.shape[0]
        device = theta0.device

        positions = [theta0]
        losses = []

        with torch.no_grad():
            losses.append(loss_fn().item())

        for _ in range(self.n_samples - 1):
            direction = torch.randn(n_params, device=device)
            direction = F.normalize(direction, dim=0)
            perturbed = theta0 + self.sample_radius * direction
            self._set_flat(model, perturbed)
            with torch.no_grad():
                losses.append(loss_fn().item())
            positions.append(perturbed.clone())

        self._set_flat(model, theta0)

        return torch.stack(positions), torch.tensor(losses, device=device)

    def analyze(
        self, model: nn.Module, loss_fn: Callable
    ) -> PreExplorationResult:
        result = PreExplorationResult()

        positions, losses = self._sample_landscape(model, loss_fn)
        result.landscape_losses = losses
        result.landscape_positions = positions
        result.loss_statistics = {
            "mean": losses.mean().item(),
            "std": losses.std().item(),
            "min": losses.min().item(),
            "max": losses.max().item(),
            "median": losses.median().item(),
            "q25": losses.quantile(0.25).item(),
            "q75": losses.quantile(0.75).item(),
        }

        config = SamplingConfig(strategy="random", max_samples=min(200, len(positions)))
        sampled, _ = StatisticalSampler.sample(positions, config)

        eda = None
        try:
            eda = EffectiveDimensionAnalyzer(device=self.device)
            dim_result = eda.analyze(sampled)
            result.dim_result = dim_result
        except Exception:
            dim_result = None

        if self.algebra is not None and self.algebra.n >= 2:
            mv_params = GeometricParameterController._extract_mv_params(model)
            if mv_params is not None and mv_params.shape[0] >= 1:
                try:
                    sa = SpectralAnalyzer(self.algebra)
                    result.spectral_result = sa.analyze(mv_params)
                except Exception:
                    pass

                try:
                    sd = SymmetryDetector(self.algebra)
                    result.symmetry_result = sd.analyze(mv_params)
                except Exception:
                    pass

                try:
                    ca = CoreCommutatorAnalyzer(self.algebra)
                    result.commutator_result = ca.analyze(mv_params)
                except Exception:
                    pass

                gpc = GeometricParameterController(algebra=self.algebra)
                result.geometric_scores = gpc.compute_geometric_scores(model)

                try:
                    k_flow = min(8, mv_params.shape[0] - 1)
                    if k_flow >= 2:
                        gf_params = GeodesicFlow(self.algebra, k=k_flow)
                        result.flow_bivectors = gf_params.flow_bivectors(mv_params)
                        result.per_point_coherence = gf_params.per_point_coherence(mv_params)
                except Exception:
                    pass

        if dim_result is not None and dim_result.intrinsic_dim >= 2:
            try:
                land_dim = min(dim_result.intrinsic_dim, 6)
                temp_algebra = CliffordAlgebra(land_dim, 0, device=self.device)
                reduced = eda.reduce(sampled, land_dim)
                mv_land = temp_algebra.embed_vector(reduced)
                k = min(8, mv_land.shape[0] - 1)
                gf = GeodesicFlow(temp_algebra, k=k)
                result.landscape_coherence = gf.coherence(mv_land)
                result.landscape_curvature = gf.curvature(mv_land)
                result.causal_report = {
                    'coherence': result.landscape_coherence,
                    'curvature': result.landscape_curvature,
                    'causal': (result.landscape_coherence > 0.5
                               and result.landscape_curvature < 0.5),
                    'label': (
                        'Causal - smooth, aligned flow'
                        if (result.landscape_coherence > 0.5
                            and result.landscape_curvature < 0.5)
                        else 'Noisy - fragmented flow'
                    ),
                }
            except Exception:
                pass

        if self.algebra is not None and dim_result is not None:
            try:
                p, q = self.algebra.p, self.algebra.q
                n = p + q
                lift_dim = min(n, dim_result.intrinsic_dim) if eda else n
                if lift_dim >= 2 and eda is not None:
                    reduced_lift = eda.reduce(sampled, lift_dim)
                    lifter = DimensionLifter(device=self.device)
                    result.lifting_report = lifter.test(
                        reduced_lift, p=lift_dim, q=0, k=min(8, reduced_lift.shape[0] - 1))
            except Exception:
                pass

        result.recommended_config = self._recommend_config(result)
        result.strategy_label = self._classify_strategy(result)

        return result

    def _recommend_config(self, result: PreExplorationResult) -> GDOConfig:
        cfg = GDOConfig()

        if result.dim_result is not None:
            pr = result.dim_result.participation_ratio
            if pr < 5:
                cfg.probe_interval = 30
                cfg.lift_k = 4
            elif pr > 20:
                cfg.probe_interval = 100
                cfg.lift_k = 8
                cfg.lift_sigma = 0.1

            ev = result.dim_result.eigenvalues
            if len(ev) >= 2 and ev[-1].item() > 1e-10:
                cond = ev[0].item() / ev[-1].item()
                if cond > 100:
                    cfg.lr = 5e-4

        coh = result.landscape_coherence
        curv = result.landscape_curvature
        if coh > 0.5:
            cfg.sprint_after = 300
            cfg.topology_interval = 100
        elif coh < 0.3:
            cfg.topology_interval = 400
            cfg.sprint_after = 800
            cfg.lift_patience = 50

        if curv > 0.5:
            cfg.lorentz_max_beta = 0.98
            cfg.max_navigate_steps = 100

        ls = result.loss_statistics
        if ls.get("mean", 0) > 1e-8:
            cv = ls.get("std", 0) / ls["mean"]
            if cv > 1.0:
                cfg.lift_patience = 50
                cfg.lift_sigma = 0.1

        gs = result.geometric_scores
        if gs:
            ce = gs.get("closure_error", None)
            if ce is not None and ce < 0.1:
                cfg.closure_trust_threshold = ce
                cfg.commutator_threshold = 0.2

            co = gs.get("coherence", None)
            if co is not None and co > 0.6:
                cfg.coherence_gate = 0.2

            ge = gs.get("grade_entropy", None)
            if ge is not None:
                if ge > 0.8:
                    cfg.entropy_exploration_threshold = 0.8
                elif ge < 0.3:
                    cfg.sprint_after = min(cfg.sprint_after, 200)

        return cfg

    @staticmethod
    def _classify_strategy(result: PreExplorationResult) -> str:
        coh = result.landscape_coherence
        curv = result.landscape_curvature
        ls = result.loss_statistics
        cv = ls.get("std", 0) / max(ls.get("mean", 1e-8), 1e-8)

        if coh < 0.3 or curv > 0.5 or cv > 1.0:
            return "EXPLORE-heavy"
        elif coh > 0.5 and curv < 0.3:
            return "SPRINT-viable"
        else:
            return "NAVIGATE-ready"


# ======================================================================
# GDOOptimizer: torch.optim.Optimizer interface
# ======================================================================

class GDOOptimizer(Optimizer):
    """Geometric Deterministic Optimizer -- torch.optim.Optimizer interface.

    Performs Adam-like updates with:
    - Per-parameter Lorentz warp scaling
    - Geodesic blend toward known targets
    - Per-group scaling from geometric controller
    - Per-manifold retraction (spin, sphere, euclidean) via Versor tags
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        algebra: Optional[CliffordAlgebra] = None,
        max_bivector_norm: Optional[float] = 10.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.algebra = algebra
        self.max_bivector_norm = max_bivector_norm

        # External state that controller can inject
        self._warp_lr: Optional[torch.Tensor] = None
        self._geodesic_target: Optional[torch.Tensor] = None
        self._geodesic_weight: float = 0.0
        self._group_scales: Optional[List[float]] = None

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        algebra: Optional[CliffordAlgebra] = None,
        max_bivector_norm: Optional[float] = 10.0,
    ) -> "GDOOptimizer":
        """Create optimizer with auto-detected manifold parameter groups."""
        grouped = group_parameters_by_manifold(model)
        param_groups = []
        for manifold in (MANIFOLD_SPIN, MANIFOLD_SPHERE, MANIFOLD_EUCLIDEAN):
            params = grouped[manifold]
            if params:
                param_groups.append({'params': params, 'manifold': manifold})
        if not param_groups:
            param_groups = [{'params': list(model.parameters()), 'manifold': MANIFOLD_EUCLIDEAN}]
        return cls(param_groups, lr=lr, betas=betas, eps=eps, algebra=algebra,
                   max_bivector_norm=max_bivector_norm)

    def set_warp_state(self, warp_lr: Optional[torch.Tensor]):
        """Set per-parameter Lorentz warp LR (from LorentzWarpOptimizer)."""
        self._warp_lr = warp_lr

    def set_geodesic_blend(self, target: Optional[torch.Tensor], weight: float = 0.3):
        """Set geodesic target for blended update."""
        self._geodesic_target = target
        self._geodesic_weight = weight

    def set_group_scales(self, scales: Optional[List[float]]):
        """Set per-group update scales from geometric controller."""
        self._group_scales = scales

    @torch.no_grad()
    def step(self, closure=None) -> Optional[torch.Tensor]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            betas = group['betas']
            beta1, beta2 = betas
            eps = group['eps']
            lr = group['lr']
            manifold = group.get('manifold', MANIFOLD_EUCLIDEAN)

            # Apply group scale
            if self._group_scales is not None and group_idx < len(self._group_scales):
                lr = lr * self._group_scales[group_idx]

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2 ** 0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Manifold retraction (matches RiemannianAdam)
                if manifold == MANIFOLD_SPHERE:
                    p_norm = p.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    p.div_(p_norm)
                elif manifold == MANIFOLD_SPIN and self.max_bivector_norm is not None:
                    p_norm = p.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(p_norm / self.max_bivector_norm, min=1.0)
                    p.div_(scale)

        return loss

    def get_state_snapshot(self) -> Dict:
        """Expose internal state for external analysis."""
        snap = {
            "warp_lr_set": self._warp_lr is not None,
            "geodesic_target_set": self._geodesic_target is not None,
            "geodesic_weight": self._geodesic_weight,
            "group_scales": self._group_scales,
            "param_group_count": len(self.param_groups),
        }
        return snap


# ======================================================================
# GDOController: Full Morse-Geometric Optimization Pipeline
# ======================================================================

class GDOController:
    """Full Morse-Geometric optimization pipeline orchestrator.

    Owns model, loss_fn, and a GDOOptimizer. Manages mode transitions,
    probes, topology search, lift oracle, and commutator scheduling.
    """

    class Mode(Enum):
        EXPLORE = "explore"
        NAVIGATE = "navigate"
        SPRINT = "sprint"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: Optional[GDOOptimizer] = None,
        lr: float = 1e-3,
        probe_interval: int = 50,
        topology_interval: int = 200,
        sprint_after: int = 500,
        max_navigate_steps: int = 150,
        lift_patience: int = 80,
        algebra: Optional[CliffordAlgebra] = None,
        device: str = 'cpu',
        config: Optional[GDOConfig] = None,
    ):
        if config is not None:
            lr = config.lr
            probe_interval = config.probe_interval
            topology_interval = config.topology_interval
            sprint_after = config.sprint_after
            max_navigate_steps = config.max_navigate_steps
            lift_patience = config.lift_patience
        self.config = config or GDOConfig(
            lr=lr, probe_interval=probe_interval,
            topology_interval=topology_interval, sprint_after=sprint_after,
            max_navigate_steps=max_navigate_steps, lift_patience=lift_patience,
        )

        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.device = device
        self.algebra = algebra

        # Create GDOOptimizer if not provided
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if algebra is not None:
                self.optimizer = GDOOptimizer.from_model(model, lr=lr, algebra=algebra)
            else:
                self.optimizer = GDOOptimizer(model.parameters(), lr=lr)

        # Sub-components
        self.topology = LandscapeTopologySearch(
            loss_fn=loss_fn, detect_every=topology_interval
        )
        self.probe = CurvatureProbe()
        self.geodesic = GeodesicIntegrator()
        self.warp = LorentzWarpOptimizer()
        self.controller = GeometricParameterController(
            algebra=algebra,
            commutator_threshold=self.config.commutator_threshold,
            fim_damping=self.config.fim_damping,
            closure_trust_threshold=self.config.closure_trust_threshold,
            coherence_gate=self.config.coherence_gate,
            entropy_exploration_threshold=self.config.entropy_exploration_threshold,
            config=self.config,
        )
        self.lift_oracle = DimensionalLiftOracle(
            patience=lift_patience,
            oracle_lr=lr,
        )

        # State
        self.landscape = LandscapeMap()
        self.mode = GDOController.Mode.EXPLORE
        self.step = 0
        self.probe_interval = probe_interval
        self.sprint_after = sprint_after
        self.max_navigate_steps = max_navigate_steps
        self._probe_result: Optional[CurvatureProbe.ProbeResult] = None
        self._commutator_schedule: Optional[List[List[int]]] = None
        self._group_scales: Optional[List[float]] = None
        self._controller_diagnostics: Optional[Dict] = None
        self._mode_history: List[str] = []

        # NAVIGATE phase tracking
        self._navigate_steps: int = 0
        self._navigate_best_loss: float = float('inf')
        self._navigate_no_improve: int = 0

        # SPRINT phase tracking (adaptive rescheduling)
        self._sprint_step: int = 0
        self._last_schedule_loss: float = float('inf')
        self._last_schedule_grad_norms: Optional[torch.Tensor] = None

        # Auto-group parameters
        self._param_group_meta: List[Dict] = []
        self._param_groups: List[List[nn.Parameter]] = self._build_param_groups()
        self._group_ranges: List[List[Tuple[int, int]]] = self._compute_group_ranges()

        # Cached Hessian eigenvectors
        self._hessian_vecs: Optional[torch.Tensor] = None
        self._hessian_vals: Optional[torch.Tensor] = None

        # Flat Adam state for EXPLORE/NAVIGATE (global) and SPRINT (per-group)
        self._adam_m: Optional[torch.Tensor] = None
        self._adam_v: Optional[torch.Tensor] = None
        self._adam_t: int = 0
        n_groups = len(self._param_groups)
        self._grp_m: List[Optional[torch.Tensor]] = [None] * n_groups
        self._grp_v: List[Optional[torch.Tensor]] = [None] * n_groups
        self._grp_t: List[int] = [0] * n_groups

    def _build_param_groups(self) -> List[List[nn.Parameter]]:
        if self.config.grouping_strategy == "geometric":
            groups, meta = self._build_geometric_param_groups()
            self._param_group_meta = meta
            return groups
        # Fallback: module-based grouping
        groups = []
        for _, module in self.model.named_children():
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                groups.append(params)
        if not groups:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                groups.append(params)
        self._param_group_meta = [
            {"manifold": "euclidean", "role": "mixed", "depth_range": (0, 0),
             "total_numel": sum(p.numel() for p in g)}
            for g in groups
        ]
        return groups

    @staticmethod
    def _classify_param(name: str, param: nn.Parameter) -> Tuple[str, str, int]:
        """Classify a parameter into (manifold, role, depth)."""
        manifold = getattr(param, '_manifold', MANIFOLD_EUCLIDEAN)

        # Determine role from name
        lower = name.lower()
        if 'grade_weights' in lower or 'bivector' in lower:
            role = "bivector"
        elif 'bias' in lower:
            role = "bias"
        elif ('weight' in lower and 'grade' not in lower
              and 'bivector' not in lower):
            role = "linear"
        else:
            role = "other"

        # Extract layer depth from name (e.g. "layer_2_rotor" -> 2)
        depth_match = re.search(r'layer[_.]?(\d+)', lower)
        depth = int(depth_match.group(1)) if depth_match else 0

        return manifold, role, depth

    def _build_geometric_param_groups(
        self,
    ) -> Tuple[List[List[nn.Parameter]], List[Dict]]:
        """Group parameters by (manifold, role) with depth-based splitting."""
        classified: Dict[Tuple[str, str], List[Tuple[int, nn.Parameter]]] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            manifold, role, depth = self._classify_param(name, param)
            key = (manifold, role)
            classified.setdefault(key, []).append((depth, param))

        if not classified:
            all_p = [p for p in self.model.parameters() if p.requires_grad]
            if all_p:
                return [all_p], [{"manifold": "euclidean", "role": "mixed",
                                  "depth_range": (0, 0),
                                  "total_numel": sum(p.numel() for p in all_p)}]
            return [], []

        raw_groups: List[Tuple[Dict, List[nn.Parameter]]] = []
        for (manifold, role), items in classified.items():
            items.sort(key=lambda x: x[0])
            # Split by depth gaps > 1
            current: List[nn.Parameter] = [items[0][1]]
            current_depths = [items[0][0]]
            for depth, param in items[1:]:
                if depth - current_depths[-1] > 1:
                    meta = {"manifold": manifold, "role": role,
                            "depth_range": (current_depths[0], current_depths[-1]),
                            "total_numel": sum(p.numel() for p in current)}
                    raw_groups.append((meta, current))
                    current = [param]
                    current_depths = [depth]
                else:
                    current.append(param)
                    current_depths.append(depth)
            meta = {"manifold": manifold, "role": role,
                    "depth_range": (current_depths[0], current_depths[-1]),
                    "total_numel": sum(p.numel() for p in current)}
            raw_groups.append((meta, current))

        # Merge undersized groups into nearest same-manifold neighbor
        min_size = self.config.min_group_size
        merged_groups: List[Tuple[Dict, List[nn.Parameter]]] = []
        undersized: List[Tuple[Dict, List[nn.Parameter]]] = []
        for meta, params in raw_groups:
            if len(params) >= min_size:
                merged_groups.append((meta, params))
            else:
                undersized.append((meta, params))

        for u_meta, u_params in undersized:
            best_idx = -1
            best_dist = float('inf')
            for i, (m, _) in enumerate(merged_groups):
                if m["manifold"] == u_meta["manifold"]:
                    dist = abs(m["depth_range"][0] - u_meta["depth_range"][0])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            if best_idx >= 0:
                m, p = merged_groups[best_idx]
                p.extend(u_params)
                m["total_numel"] += u_meta["total_numel"]
                m["depth_range"] = (
                    min(m["depth_range"][0], u_meta["depth_range"][0]),
                    max(m["depth_range"][1], u_meta["depth_range"][1]),
                )
            else:
                merged_groups.append((u_meta, u_params))

        # Cap at max_groups by merging smallest same-manifold groups
        max_g = self.config.max_groups
        while len(merged_groups) > max_g:
            # Find smallest group
            smallest_idx = min(
                range(len(merged_groups)),
                key=lambda i: merged_groups[i][0]["total_numel"],
            )
            s_meta, s_params = merged_groups.pop(smallest_idx)
            best_idx = -1
            best_dist = float('inf')
            for i, (m, _) in enumerate(merged_groups):
                if m["manifold"] == s_meta["manifold"]:
                    dist = abs(m["depth_range"][0] - s_meta["depth_range"][0])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
            if best_idx >= 0:
                m, p = merged_groups[best_idx]
                p.extend(s_params)
                m["total_numel"] += s_meta["total_numel"]
                m["depth_range"] = (
                    min(m["depth_range"][0], s_meta["depth_range"][0]),
                    max(m["depth_range"][1], s_meta["depth_range"][1]),
                )
            else:
                # No same-manifold neighbor, just append back
                merged_groups.append((s_meta, s_params))
                break

        groups = [p for _, p in merged_groups]
        metas = [m for m, _ in merged_groups]
        return groups, metas

    def _compute_group_ranges(self) -> List[List[Tuple[int, int]]]:
        offsets: Dict[int, Tuple[int, int]] = {}
        ptr = 0
        for p in self.model.parameters():
            offsets[id(p)] = (ptr, ptr + p.numel())
            ptr += p.numel()
        return [
            [offsets[id(p)] for p in grp if id(p) in offsets]
            for grp in self._param_groups
        ]

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])

    def _get_flat_grad(self) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return torch.cat([
            p.grad.detach().reshape(-1) if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in self.model.parameters()
        ])

    def _set_flat_params(self, flat: torch.Tensor):
        idx = 0
        for p in self.model.parameters():
            sz = p.numel()
            p.data.copy_(flat[idx:idx + sz].reshape(p.shape))
            idx += sz

    def _adam_warp_step(self, flat_grad: torch.Tensor):
        """Global Adam + Lorentz warp step (EXPLORE and NAVIGATE)."""
        device = flat_grad.device
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        if self._adam_m is None:
            self._adam_m = torch.zeros_like(flat_grad)
            self._adam_v = torch.zeros_like(flat_grad)
        self._adam_t += 1
        t = self._adam_t

        self._adam_m = beta1 * self._adam_m + (1 - beta1) * flat_grad
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * flat_grad * flat_grad

        m_hat = self._adam_m / (1 - beta1 ** t)
        v_hat = self._adam_v / (1 - beta2 ** t)
        adam_dir = m_hat / (v_hat.sqrt() + eps)

        lr_vec = self.warp.warped_lr(self.lr, flat_grad.shape[0], device)

        self._set_flat_params(self._get_flat_params() - lr_vec * adam_dir)

    def _group_adam_warp_step(self, group_idx: int, group_grad: torch.Tensor):
        """Per-group Adam + warp step for SPRINT."""
        device = group_grad.device
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        if self._grp_m[group_idx] is None:
            self._grp_m[group_idx] = torch.zeros_like(group_grad)
            self._grp_v[group_idx] = torch.zeros_like(group_grad)
        self._grp_t[group_idx] += 1
        t = self._grp_t[group_idx]

        m = beta1 * self._grp_m[group_idx] + (1 - beta1) * group_grad
        v = beta2 * self._grp_v[group_idx] + (1 - beta2) * group_grad * group_grad
        self._grp_m[group_idx] = m
        self._grp_v[group_idx] = v

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        adam_dir = m_hat / (v_hat.sqrt() + eps)

        lr_vec = self.warp.warped_lr(self.lr, group_grad.shape[0], device)
        if self._group_scales is not None and group_idx < len(self._group_scales):
            lr_vec = lr_vec * self._group_scales[group_idx]

        flat_p = self._get_flat_params()
        ptr = 0
        for start, end in self._group_ranges[group_idx]:
            sz = end - start
            flat_p[start:end] -= lr_vec[ptr:ptr + sz] * adam_dir[ptr:ptr + sz]
            ptr += sz
        self._set_flat_params(flat_p)

    def _run_scheduling(self):
        """Compute (or recompute) the commutator-based update schedule."""
        if len(self._param_groups) > 1:
            print(f"  [GPC] Analyzing parameter geometry...")
            schedule, scales, diagnostics = (
                self.controller.analyze_and_schedule(
                    self.model, self.loss_fn, self._param_groups,
                    group_meta=self._param_group_meta,
                )
            )
            self._commutator_schedule = schedule
            self._group_scales = scales
            self._controller_diagnostics = diagnostics
            self.landscape.commutator_scores = {
                f"({i},{j})": v
                for (i, j), v in diagnostics["hybrid_scores"].items()
            }
            print(f"  [GPC] Schedule: {schedule}")
            print(f"  [GPC] Group scales: "
                  f"{[f'{s:.2f}' for s in scales]}")
            gs = diagnostics.get("geometric_scores", {})
            if gs:
                ce = gs.get("closure_error", None)
                co = gs.get("coherence", None)
                ge = gs.get("grade_entropy", None)
                parts = []
                if ce is not None:
                    parts.append(f"closure={ce:.4f}")
                if co is not None:
                    parts.append(f"coherence={co:.4f}")
                if ge is not None:
                    parts.append(f"entropy={ge:.4f}")
                if parts:
                    print(f"  [GPC] {' | '.join(parts)}")
        else:
            self._commutator_schedule = (
                [[0]] if self._param_groups else [[]]
            )
            self._group_scales = [1.0]

        self._last_schedule_loss = float('inf')
        self._sprint_step = 0

    def _maybe_reschedule(self, current_loss: float) -> bool:
        """Check if rescheduling is needed and perform it."""
        # Condition 1: periodic interval
        interval_trigger = (
            self._sprint_step > 0
            and self._sprint_step % self.config.reschedule_interval == 0
        )

        # Condition 2: significant loss improvement
        loss_trigger = False
        if self._last_schedule_loss < float('inf'):
            rel_improve = (
                (self._last_schedule_loss - current_loss)
                / (abs(self._last_schedule_loss) + 1e-8)
            )
            loss_trigger = rel_improve > self.config.reschedule_loss_delta

        # Condition 3: gradient norm distribution shift
        grad_trigger = False
        if self._last_schedule_grad_norms is not None:
            self.model.zero_grad()
            loss_val = self.loss_fn()
            loss_val.backward()
            full_grad = self._get_flat_grad()
            current_norms = torch.tensor([
                torch.cat([full_grad[s:e] for s, e in ranges]).norm().item()
                for ranges in self._group_ranges
            ])
            # Normalize to distributions
            old_p = self._last_schedule_grad_norms / (
                self._last_schedule_grad_norms.sum() + 1e-8
            )
            new_p = current_norms / (current_norms.sum() + 1e-8)
            # KL divergence (with smoothing)
            eps = 1e-6
            old_p = old_p.clamp(min=eps)
            new_p = new_p.clamp(min=eps)
            kl = (new_p * (new_p / old_p).log()).sum().item()
            grad_trigger = kl > self.config.reschedule_grad_kl_threshold
            self.model.zero_grad()

        if not (interval_trigger or loss_trigger or grad_trigger):
            return False

        # Capture gradient norms before rescheduling
        self.model.zero_grad()
        loss_val = self.loss_fn()
        loss_val.backward()
        full_grad = self._get_flat_grad()
        self._last_schedule_grad_norms = torch.tensor([
            torch.cat([full_grad[s:e] for s, e in ranges]).norm().item()
            for ranges in self._group_ranges
        ])
        self.model.zero_grad()

        old_n_colors = (len(self._commutator_schedule)
                        if self._commutator_schedule else 0)

        # Recompute schedule
        self._run_scheduling()
        self._last_schedule_loss = current_loss

        new_n_colors = (len(self._commutator_schedule)
                        if self._commutator_schedule else 0)

        # Reinitialize Adam state for affected groups (safe warm restart)
        if new_n_colors != old_n_colors:
            n_groups = len(self._param_groups)
            self._grp_m = [None] * n_groups
            self._grp_v = [None] * n_groups
            self._grp_t = [0] * n_groups

        reason = ("interval" if interval_trigger
                  else "loss_delta" if loss_trigger else "grad_shift")
        print(f"  [GPC] Rescheduled ({reason}): "
              f"{old_n_colors} -> {new_n_colors} colors")
        return True

    def _apply_color_updates(
        self, color: List[int], full_grad: torch.Tensor,
    ):
        """Batched per-color update: one flat-param read/write per color."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        flat_p = self._get_flat_params()

        for group_idx in color:
            if group_idx >= len(self._param_groups):
                continue
            ranges = self._group_ranges[group_idx]
            group_grad = torch.cat([full_grad[s:e] for s, e in ranges])
            device = group_grad.device

            if self._grp_m[group_idx] is None:
                self._grp_m[group_idx] = torch.zeros_like(group_grad)
                self._grp_v[group_idx] = torch.zeros_like(group_grad)
            self._grp_t[group_idx] += 1
            t = self._grp_t[group_idx]

            m = beta1 * self._grp_m[group_idx] + (1 - beta1) * group_grad
            v = (beta2 * self._grp_v[group_idx]
                 + (1 - beta2) * group_grad * group_grad)
            self._grp_m[group_idx] = m
            self._grp_v[group_idx] = v

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            adam_dir = m_hat / (v_hat.sqrt() + eps)

            lr_vec = self.warp.warped_lr(self.lr, group_grad.shape[0], device)
            if (self._group_scales is not None
                    and group_idx < len(self._group_scales)):
                lr_vec = lr_vec * self._group_scales[group_idx]

            ptr = 0
            for start, end in ranges:
                sz = end - start
                flat_p[start:end] -= lr_vec[ptr:ptr + sz] * adam_dir[ptr:ptr + sz]
                ptr += sz

        self._set_flat_params(flat_p)

    def optimize_step(self, loss: torch.Tensor) -> Dict:
        """Execute one step of Morse-geometric optimization."""
        current_loss = loss.item()
        info = {"step": self.step, "mode": self.mode.value, "loss": current_loss}
        self._mode_history.append(self.mode.value)
        params = list(self.model.parameters())

        # ---- EXPLORE ----
        if self.mode == GDOController.Mode.EXPLORE:
            if self.step % self.probe_interval == 0:
                self._probe_result = self.probe.probe(self.loss_fn, params)
                self.landscape.curvature_history.append(self._probe_result.mean_curvature)
                self.landscape.gradient_norm_history.append(self._probe_result.grad_norm)
                self.warp.update(
                    self._probe_result.grad_norm,
                    self._probe_result.plateau_score,
                    self._probe_result.min_curvature_dir,
                )
                info["probe"] = {
                    "mean_curvature": self._probe_result.mean_curvature,
                    "plateau_score": self._probe_result.plateau_score,
                    "grad_norm": self._probe_result.grad_norm,
                    "beta": self.warp._beta,
                }

            if self.lift_oracle.should_lift(current_loss, self.step):
                new_flat, new_loss = self.lift_oracle.lift_and_search(
                    self.model, self.loss_fn, current_loss,
                    probe_result=self._probe_result,
                    hessian_vecs=self._hessian_vecs,
                    hessian_vals=self._hessian_vals,
                )
                if new_flat is not None:
                    self._set_flat_params(new_flat)
                    self._adam_m = None
                    self._adam_v = None
                    self._adam_t = 0
                    self.step += 1
                    info["lift_oracle"] = f"improved to {new_loss:.5f}"
                    return info

            cp = self.topology.check(self.loss_fn(), params, self.step)
            if self.topology._last_eigenvecs is not None:
                self._hessian_vecs = self.topology._last_eigenvecs
                self._hessian_vals = self.topology._last_eigenvalues
            if cp is not None:
                self.landscape.add_critical(cp)
                info["critical_point"] = str(cp)
                print(f"  [Morse] Detected {cp}")
                if cp.point_type == CriticalPointType.MINIMUM:
                    lower = self.landscape.lower_minima(cp.loss)
                    if lower:
                        target = min(lower, key=lambda x: x.loss)
                        self.geodesic.set_target(target.params, target.loss)
                        self._navigate_steps = 0
                        self._navigate_best_loss = current_loss
                        self._navigate_no_improve = 0
                        self.mode = GDOController.Mode.NAVIGATE
                        print(f"  [Morse] -> NAVIGATE toward {target}")

            if self.step >= self.sprint_after:
                self.mode = GDOController.Mode.SPRINT
                print(f"  [Morse] Step {self.step}: -> SPRINT")

            loss.backward()
            flat_g = self._get_flat_grad()
            self._adam_warp_step(flat_g)
            self.model.zero_grad()

        # ---- NAVIGATE ----
        elif self.mode == GDOController.Mode.NAVIGATE:
            loss.backward()
            flat_g = self._get_flat_grad()
            hess_diag = flat_g.abs() + 1e-6
            nat_step = self.geodesic.natural_gradient_step(flat_g, hess_diag)
            flat_p = self._get_flat_params()
            delta = self.geodesic.geodesic_blend(flat_p, nat_step, self.lr)
            self._set_flat_params(flat_p + delta)
            self.model.zero_grad()

            self._navigate_steps += 1

            if current_loss < self._navigate_best_loss - 1e-4:
                self._navigate_best_loss = current_loss
                self._navigate_no_improve = 0
            else:
                self._navigate_no_improve += 1

            stuck = self._navigate_no_improve >= 30
            timed_out = self._navigate_steps >= self.max_navigate_steps
            if stuck or timed_out or self.step >= self.sprint_after:
                reason = "stuck" if stuck else ("timeout" if timed_out else "sprint")
                next_mode = (GDOController.Mode.SPRINT
                             if self.step >= self.sprint_after
                             else GDOController.Mode.EXPLORE)
                print(f"  [Morse] NAVIGATE exit ({reason}) -> {next_mode.value}")
                self.mode = next_mode
                self.geodesic._target = None

        # ---- SPRINT ----
        elif self.mode == GDOController.Mode.SPRINT:
            if self._commutator_schedule is None:
                self._run_scheduling()

            # Adaptive rescheduling
            if (self.config.adaptive_reschedule
                    and self._commutator_schedule is not None):
                if self._maybe_reschedule(current_loss):
                    info["rescheduled"] = True

            for color in self._commutator_schedule:
                self.model.zero_grad()
                loss_c = self.loss_fn()
                loss_c.backward()
                full_grad = self._get_flat_grad()
                self._apply_color_updates(color, full_grad)

            self._sprint_step += 1
            self.model.zero_grad()

        self.step += 1
        return info

    # === State Inspection API ===
    def get_topology_map(self) -> LandscapeMap:
        return self.landscape

    def get_mode_history(self) -> List[str]:
        return self._mode_history

    def get_full_diagnostics(self) -> Dict:
        return {
            "topology_map": {
                "critical_points": len(self.landscape.critical_points),
                "curvature_history": self.landscape.curvature_history,
                "gradient_norm_history": self.landscape.gradient_norm_history,
                "plateau_episodes": self.landscape.plateau_episodes,
                "commutator_scores": self.landscape.commutator_scores,
            },
            "mode_history": self._mode_history,
            "commutator_schedule": self._commutator_schedule,
            "group_scales": self._group_scales,
            "controller_diagnostics": self._controller_diagnostics,
            "lift_oracle": {
                "lift_count": self.lift_oracle._lift_count,
                "consecutive_fails": self.lift_oracle._consecutive_fails,
                "current_sigma": self.lift_oracle._current_sigma,
                "best_loss": self.lift_oracle._best_loss,
            },
            "warp": {
                "beta": self.warp._beta,
                "gamma": self.warp.gamma,
                "on_plateau": self.warp._on_plateau,
                "plateau_steps": self.warp._plateau_steps,
            },
            "optimizer_state": self.optimizer.get_state_snapshot(),
        }


# Backward compatibility alias
GeometricDeterministicOptimizer = GDOController


# ======================================================================
# Benchmark Models
# ======================================================================

# --- Category: Analytic Functions ---

class RosenbrockModel(nn.Module):
    """2D Rosenbrock function. Famous narrow curved valley."""
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b
        self.x = nn.Parameter(torch.tensor([-1.5]))
        self.y = nn.Parameter(torch.tensor([1.5]))

    def forward(self) -> torch.Tensor:
        return (self.a - self.x) ** 2 + self.b * (self.y - self.x ** 2) ** 2


class RastriginModel(nn.Module):
    """N-dimensional Rastrigin function. Many local minima; global at x=0."""
    def __init__(self, n_dims: int = 4, A: float = 10.0):
        super().__init__()
        self.A = A
        self.n = n_dims
        self.x = nn.Parameter(torch.randn(n_dims) * 3.0)

    def forward(self) -> torch.Tensor:
        x = self.x
        return self.A * self.n + (x ** 2 - self.A * torch.cos(2 * math.pi * x)).sum()


class AckleyModel(nn.Module):
    """N-dimensional Ackley function. Nearly flat plateau with narrow central well.
    Tests Lorentz warp effectiveness."""
    def __init__(self, n_dims: int = 10, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi):
        super().__init__()
        self._a = a
        self._b = b
        self._c = c
        self.x = nn.Parameter(torch.randn(n_dims) * 2.0)

    def forward(self) -> torch.Tensor:
        x = self.x
        sum_sq = (x ** 2).mean()
        sum_cos = (torch.cos(self._c * x)).mean()
        return -self._a * torch.exp(-self._b * sum_sq.sqrt()) - torch.exp(sum_cos) + self._a + math.e


class StyblinskiTangModel(nn.Module):
    """N-dimensional Styblinski-Tang. Multiple asymmetric wells.
    Tests topology search for finding global basin."""
    def __init__(self, n_dims: int = 6):
        super().__init__()
        self.x = nn.Parameter(torch.randn(n_dims) * 3.0)

    def forward(self) -> torch.Tensor:
        x = self.x
        return 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x).sum()


# --- Category: Geometric Primitives ---

class SmallGBNModel(CliffordModule):
    """Small Geometric Blade Network for testing optimizer on actual GA model."""

    def __init__(self, p: int = 3, q: int = 0, channels: int = 4, device: str = 'cpu'):
        algebra = CliffordAlgebra(p, q, device=device)
        super().__init__(algebra)
        dim = 2 ** (p + q)
        self.norm = CliffordLayerNorm(self.algebra, channels)
        self.rotor = RotorLayer(self.algebra, channels)
        self.linear = CliffordLinear(self.algebra, channels, channels)
        self.act = GeometricGELU(self.algebra)
        self._channels = channels
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.rotor(x)
        x = self.act(x)
        x = self.linear(x)
        return x


class RotorRegistrationModel(CliffordModule):
    """Fit a rotor in Cl(3,0) to align a source point cloud to a rotated+noised target."""

    def __init__(
        self,
        n_points: int = 50,
        noise_std: float = 0.05,
        rotation_angle: float = 2.5,
        device: str = 'cpu',
    ):
        algebra = CliffordAlgebra(3, 0, device=device)
        super().__init__(algebra)

        torch.manual_seed(42)
        raw = torch.randn(n_points, 3, device=device)
        raw = F.normalize(raw, dim=-1)
        self.register_buffer('source', raw)

        axis = torch.tensor([1.0, 1.0, 1.0], device=device)
        axis = axis / axis.norm()
        gt_bv = self._axis_angle_to_bivector(axis, rotation_angle)
        self.register_buffer('gt_bivector', gt_bv)

        gt_rotor = self.algebra.exp(-0.5 * gt_bv.unsqueeze(0))
        source_mv = self.algebra.embed_vector(raw)
        rotated = self.algebra.sandwich_product(
            gt_rotor.expand(n_points, -1),
            source_mv.unsqueeze(1),
        ).squeeze(1)
        target_pts = self._extract_vector(rotated)
        target_pts = target_pts + noise_std * torch.randn_like(target_pts)
        self.register_buffer('target', target_pts)

        self.rotor = RotorLayer(self.algebra, channels=1)

    def _axis_angle_to_bivector(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        bv = torch.zeros(self.algebra.dim, device=axis.device)
        bv[3] = angle * axis[2]
        bv[5] = -angle * axis[1]
        bv[6] = angle * axis[0]
        return bv

    def _extract_vector(self, mv: torch.Tensor) -> torch.Tensor:
        return torch.stack([mv[..., 1], mv[..., 2], mv[..., 4]], dim=-1)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source)
        source_mv = source_mv.unsqueeze(1)
        rotated_mv = self.rotor(source_mv)
        pred_pts = self._extract_vector(rotated_mv.squeeze(1))
        return F.mse_loss(pred_pts, self.target)

    def angular_error(self) -> float:
        with torch.no_grad():
            learned_bv = torch.zeros(
                self.algebra.dim, device=self.gt_bivector.device
            )
            learned_bv[self.rotor.grade_indices] = self.rotor.grade_weights[0]
            r_learned = self.algebra.exp(-0.5 * learned_bv.unsqueeze(0))
            r_gt = self.algebra.exp(-0.5 * self.gt_bivector.unsqueeze(0))
            r_gt_rev = self.algebra.reverse(r_gt)
            product = self.algebra.geometric_product(r_learned, r_gt_rev)
            cos_half = product[0, 0].abs().clamp(max=1.0).item()
            return 2.0 * math.acos(cos_half)


class MinkowskiRotorModel(CliffordModule):
    """Fit a Lorentz boost in Cl(2,1) to align spacetime events.
    Tests optimizer on indefinite signature (mixed exp map regime)."""
    def __init__(self, n_events: int = 30, boost_rapidity: float = 0.8, device: str = 'cpu'):
        algebra = CliffordAlgebra(2, 1, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim  # 8

        torch.manual_seed(42)
        # Source events: spatial (e1, e2) random, temporal (e3 where e3^2=-1) = 1
        raw = torch.randn(n_events, 2, device=device)
        spatial = F.normalize(raw, dim=-1) * 0.5
        events_3d = torch.cat([spatial, torch.ones(n_events, 1, device=device)], dim=-1)
        self.register_buffer('source', events_3d)

        # Ground-truth: boost along e1-e3 plane (bivector e13)
        # In Cl(2,1): e1^2=+1, e3^2=-1, so e13 is a boost plane
        gt_bv = torch.zeros(dim, device=device)
        # e13 index in Cl(2,1): basis e1=idx1, e3=idx4 -> e13=idx5
        gt_bv[5] = boost_rapidity  # e13 component
        self.register_buffer('gt_bivector', gt_bv)

        gt_rotor = self.algebra.exp(-0.5 * gt_bv.unsqueeze(0))
        source_mv = self.algebra.embed_vector(events_3d)
        boosted = self.algebra.sandwich_product(
            gt_rotor.expand(n_events, -1),
            source_mv.unsqueeze(1),
        ).squeeze(1)
        # Extract vector part
        target_3d = torch.stack([boosted[..., 1], boosted[..., 2], boosted[..., 4]], dim=-1)
        target_3d = target_3d + 0.02 * torch.randn_like(target_3d)
        self.register_buffer('target', target_3d)

        self.rotor = RotorLayer(self.algebra, channels=1)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source).unsqueeze(1)
        boosted_mv = self.rotor(source_mv).squeeze(1)
        pred = torch.stack([boosted_mv[..., 1], boosted_mv[..., 2], boosted_mv[..., 4]], dim=-1)
        return F.mse_loss(pred, self.target)

    def rapidity_error(self) -> float:
        with torch.no_grad():
            learned_bv = torch.zeros(self.algebra.dim, device=self.gt_bivector.device)
            learned_bv[self.rotor.grade_indices] = self.rotor.grade_weights[0]
            return (learned_bv - self.gt_bivector).norm().item()


class ConformalRegistrationModel(CliffordModule):
    """Fit a conformal rotor in Cl(4,1) for rotation+translation.
    Tests optimizer on 32-dimensional multivectors."""
    def __init__(self, n_points: int = 40, device: str = 'cpu'):
        algebra = CliffordAlgebra(4, 1, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim  # 32

        torch.manual_seed(42)
        raw = torch.randn(n_points, 3, device=device) * 0.5
        self.register_buffer('source_pts', raw)

        # Ground-truth: rotation + translation
        # Rotation around z-axis by 0.8 rad
        gt_bv = torch.zeros(dim, device=device)
        # In Cl(4,1) the first 3 spatial bivectors handle rotations
        # e12 component for z-rotation
        bv_indices = [i for i in range(dim) if bin(i).count('1') == 2]
        if len(bv_indices) > 0:
            gt_bv[bv_indices[0]] = 0.4  # small rotation bivector

        self.register_buffer('gt_bivector', gt_bv)

        # Apply via rotor
        gt_rotor = self.algebra.exp(-0.5 * gt_bv.unsqueeze(0))
        # Embed source as 5D vectors (pad with zeros for CGA extra dims)
        src_5d = torch.zeros(n_points, 5, device=device)
        src_5d[:, :3] = raw
        source_mv = self.algebra.embed_vector(src_5d)
        rotated = self.algebra.sandwich_product(
            gt_rotor.expand(n_points, -1),
            source_mv.unsqueeze(1),
        ).squeeze(1)
        # Also apply a small translation shift in Euclidean coords
        target_mv = rotated.clone()
        target_mv[:, 1] += 0.3  # shift e1 component
        target_mv += 0.01 * torch.randn_like(target_mv)
        self.register_buffer('target_mv', target_mv)
        self.register_buffer('source_mv', source_mv)

        self.rotor = RotorLayer(self.algebra, channels=1)

    def forward(self) -> torch.Tensor:
        src = self.source_mv.unsqueeze(1)
        pred = self.rotor(src).squeeze(1)
        return F.mse_loss(pred, self.target_mv)


class MultiRotorRegistrationModel(CliffordModule):
    """Fit a MultiRotorLayer to align multi-cluster point clouds.
    Tests commutator scheduling and multi-modal optimization."""
    def __init__(self, n_clusters: int = 3, points_per_cluster: int = 20, device: str = 'cpu'):
        algebra = CliffordAlgebra(3, 0, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim

        torch.manual_seed(42)
        sources = []
        targets = []

        for c in range(n_clusters):
            center = torch.randn(3, device=device)
            pts = center + 0.2 * torch.randn(points_per_cluster, 3, device=device)
            sources.append(pts)

            # Each cluster gets a different rotation
            angle = 0.5 + c * 1.0
            axis = F.normalize(torch.randn(3, device=device), dim=0)
            bv = torch.zeros(dim, device=device)
            bv[3] = angle * axis[2]
            bv[5] = -angle * axis[1]
            bv[6] = angle * axis[0]
            rotor = self.algebra.exp(-0.5 * bv.unsqueeze(0))
            pts_mv = self.algebra.embed_vector(pts)
            rotated = self.algebra.sandwich_product(
                rotor.expand(points_per_cluster, -1),
                pts_mv.unsqueeze(1),
            ).squeeze(1)
            tgt_pts = torch.stack([rotated[..., 1], rotated[..., 2], rotated[..., 4]], dim=-1)
            targets.append(tgt_pts + 0.03 * torch.randn_like(tgt_pts))

        self.register_buffer('source', torch.cat(sources))
        self.register_buffer('target', torch.cat(targets))

        self.multi_rotor = MultiRotorLayer(self.algebra, channels=1, num_rotors=n_clusters)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source).unsqueeze(1)
        rotated_mv = self.multi_rotor(source_mv).squeeze(1)
        pred = torch.stack([rotated_mv[..., 1], rotated_mv[..., 2], rotated_mv[..., 4]], dim=-1)
        return F.mse_loss(pred, self.target)


# --- Category: GA Neural Networks ---

class MediumGBNModel(CliffordModule):
    """Medium GBN using GeometricBladeNetwork. 3 layers, 16ch.
    Task: learn regression on multivector inputs."""
    def __init__(self, p=3, q=0, channels=16, layers=3, n_samples=64, device='cpu'):
        algebra = CliffordAlgebra(p, q, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim
        self.gbn = GeometricBladeNetwork(
            self.algebra, in_channels=channels,
            hidden_channels=channels, out_channels=channels,
            layers=layers,
        )
        torch.manual_seed(42)
        self.register_buffer('X', torch.randn(n_samples, channels, dim, device=device) * 0.3)
        self.register_buffer('y', self.X[:, :, 0].mean(dim=1, keepdim=True))

    def forward(self) -> torch.Tensor:
        out = self.gbn(self.X)
        pred = out[:, :, 0].mean(dim=1, keepdim=True)
        return F.mse_loss(pred, self.y)


class MultiSigGBNModel(CliffordModule):
    """GBN in Minkowski signature Cl(2,1). 2 layers, 8ch.
    Tests optimizer with mixed exp map regime."""
    def __init__(self, channels=8, layers=2, n_samples=48, device='cpu'):
        algebra = CliffordAlgebra(2, 1, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim
        self.gbn = GeometricBladeNetwork(
            self.algebra, in_channels=channels,
            hidden_channels=channels, out_channels=channels,
            layers=layers,
        )
        torch.manual_seed(42)
        self.register_buffer('X', torch.randn(n_samples, channels, dim, device=device) * 0.3)
        self.register_buffer('y', self.X[:, :, 0].mean(dim=1, keepdim=True))

    def forward(self) -> torch.Tensor:
        out = self.gbn(self.X)
        pred = out[:, :, 0].mean(dim=1, keepdim=True)
        return F.mse_loss(pred, self.y)


class DeepGBNModel(CliffordModule):
    """Deep GBN (5 layers, 16 channels) for scalability testing.
    Task: predict grade-2 energy from input multivectors."""
    def __init__(self, p=3, q=0, channels=16, layers=5, n_samples=48, device='cpu'):
        algebra = CliffordAlgebra(p, q, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim
        self.gbn = GeometricBladeNetwork(
            self.algebra, in_channels=channels,
            hidden_channels=channels, out_channels=channels,
            layers=layers,
        )
        torch.manual_seed(42)
        self.register_buffer('X', torch.randn(n_samples, channels, dim, device=device) * 0.3)
        # Target: grade-2 energy of input
        grade2_mask = self.algebra.grade_masks_float[2]
        g2_energy = (self.X * grade2_mask).pow(2).sum(dim=-1).mean(dim=1, keepdim=True)
        self.register_buffer('y', g2_energy)

    def forward(self) -> torch.Tensor:
        out = self.gbn(self.X)
        pred = out[:, :, 0].mean(dim=1, keepdim=True)
        return F.mse_loss(pred, self.y)


# --- Category: Manifold Tasks ---

class SO3InterpolationModel(CliffordModule):
    """Learn a smooth rotor trajectory through waypoints on SO(3).
    Tests geodesic integrator on curved manifold."""
    def __init__(self, n_waypoints: int = 8, device: str = 'cpu'):
        algebra = CliffordAlgebra(3, 0, device=device)
        super().__init__(algebra)
        dim = self.algebra.dim
        self.n_waypoints = n_waypoints

        torch.manual_seed(42)
        # Generate random waypoint rotors
        waypoint_bivectors = []
        for i in range(n_waypoints):
            bv = torch.zeros(dim, device=device)
            angle = 0.3 + i * 0.5
            axis = F.normalize(torch.randn(3, device=device), dim=0)
            bv[3] = angle * axis[2]
            bv[5] = -angle * axis[1]
            bv[6] = angle * axis[0]
            waypoint_bivectors.append(bv)
        waypoint_bvs = torch.stack(waypoint_bivectors)
        waypoint_rotors = self.algebra.exp(-0.5 * waypoint_bvs)
        self.register_buffer('target_rotors', waypoint_rotors)

        # A test vector to rotate (unit e1)
        test_vec = torch.zeros(dim, device=device)
        test_vec[1] = 1.0  # e1
        self.register_buffer('test_vec', test_vec)

        # Compute target points
        targets = []
        for i in range(n_waypoints):
            R = waypoint_rotors[i:i+1]
            v = test_vec.unsqueeze(0)
            rotated = self.algebra.sandwich_product(R, v.unsqueeze(1)).squeeze(1)
            targets.append(rotated)
        self.register_buffer('target_points', torch.cat(targets))

        # Learnable rotor bank -- one per waypoint
        self.rotor_bank = RotorLayer(self.algebra, channels=n_waypoints)

    def forward(self) -> torch.Tensor:
        # Rotate test vector by each learned rotor
        test_expanded = self.test_vec.unsqueeze(0).unsqueeze(0).expand(1, self.n_waypoints, -1)
        rotated = self.rotor_bank(test_expanded).squeeze(0)  # [n_waypoints, dim]
        return F.mse_loss(rotated, self.target_points)

    def geodesic_deviation(self) -> float:
        """Measure how far learned rotors are from ground-truth on SO(3)."""
        with torch.no_grad():
            learned_bv = torch.zeros(self.n_waypoints, self.algebra.dim,
                                     device=self.target_rotors.device)
            learned_bv[:, self.rotor_bank.grade_indices] = self.rotor_bank.grade_weights
            r_learned = self.algebra.exp(-0.5 * learned_bv)
            r_gt_rev = self.algebra.reverse(self.target_rotors)
            product = self.algebra.geometric_product(r_learned, r_gt_rev)
            cos_half = product[:, 0].abs().clamp(max=1.0)
            angles = 2.0 * torch.acos(cos_half)
            return angles.mean().item()


# ======================================================================
# Optimizer Factory & Training Loops
# ======================================================================

def _collect_bivector_norms(model: nn.Module) -> float:
    """Sum of bivector parameter norms across all rotor layers."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, (RotorLayer, MultiRotorLayer)):
            w = m.grade_weights if isinstance(m, RotorLayer) else m.rotor_grade_weights
            total += w.detach().norm().item()
    return total


def create_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    algebra: Optional[CliffordAlgebra] = None,
    loss_fn: Optional[Callable] = None,
    config: Optional[GDOConfig] = None,
    device: str = 'cpu',
) -> Tuple[Union[Optimizer, GDOController], str]:
    """Factory for all optimizer variants."""
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr), "Adam"
    elif name == 'riemannian_adam':
        if algebra is None:
            return torch.optim.Adam(model.parameters(), lr=lr), "Adam (no algebra)"
        return RiemannianAdam.from_model(model, lr=lr, algebra=algebra), "RiemannianAdam"
    elif name == 'exponential_sgd':
        if algebra is None:
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9), "SGD"
        return ExponentialSGD.from_model(model, lr=lr, algebra=algebra), "ExponentialSGD"
    elif name == 'gdo':
        assert loss_fn is not None, "GDO requires loss_fn"
        gdo_opt = GDOOptimizer.from_model(model, lr=lr, algebra=algebra) if algebra else \
            GDOOptimizer(model.parameters(), lr=lr)
        controller = GDOController(
            model, loss_fn, optimizer=gdo_opt,
            config=config, algebra=algebra, device=device, lr=lr,
        )
        return controller, "GDO"
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_loop_standard(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    steps: int,
    metric_fn: Optional[Callable] = None,
    log_interval: int = 500,
    label: str = "",
) -> ExperimentResult:
    """Standard training loop for torch.optim.Optimizer (Adam, RiemannianAdam)."""
    losses = []
    wall_times = []
    metrics: Dict[str, List[float]] = {}
    bv_norms = []

    for s in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        wt = time.perf_counter() - t0

        losses.append(loss.item())
        wall_times.append(wt)
        bv_norms.append(_collect_bivector_norms(model))

        if metric_fn is not None:
            for k, v in metric_fn().items():
                metrics.setdefault(k, []).append(v)

        if s % log_interval == 0 or s == steps - 1:
            print(f"  [{label}] Step {s:5d}: loss={loss.item():.6f}")

    return ExperimentResult(
        name="", optimizer_name=label,
        losses=losses, wall_times=wall_times,
        metrics=metrics,
        final_loss=losses[-1] if losses else float('inf'),
        total_wall_time=sum(wall_times),
        bivector_norms=bv_norms,
    )


def train_loop_gdo(
    model: nn.Module,
    controller: GDOController,
    steps: int,
    metric_fn: Optional[Callable] = None,
    log_interval: int = 500,
) -> ExperimentResult:
    """GDO controller loop with full diagnostic collection."""
    losses = []
    wall_times = []
    metrics: Dict[str, List[float]] = {}
    bv_norms = []

    for s in range(steps):
        t0 = time.perf_counter()
        loss = controller.loss_fn()
        info = controller.optimize_step(loss)
        wt = time.perf_counter() - t0

        losses.append(info['loss'])
        wall_times.append(wt)
        bv_norms.append(_collect_bivector_norms(model))

        if metric_fn is not None:
            for k, v in metric_fn().items():
                metrics.setdefault(k, []).append(v)

        if s % log_interval == 0 or s == steps - 1:
            print(f"  [GDO] Step {s:5d}: loss={info['loss']:.6f}  mode={info['mode']}")

    return ExperimentResult(
        name="", optimizer_name="GDO",
        losses=losses, wall_times=wall_times,
        metrics=metrics,
        final_loss=losses[-1] if losses else float('inf'),
        total_wall_time=sum(wall_times),
        gdo_diagnostics=controller.get_full_diagnostics(),
        bivector_norms=bv_norms,
        mode_history=controller.get_mode_history(),
    )


def run_comparison(
    task_name: str,
    model_factory: Callable,
    loss_factory: Callable,
    config: ExperimentConfig,
    optimizers: Tuple[str, ...] = ('gdo', 'riemannian_adam', 'adam'),
    metric_factory: Optional[Callable] = None,
    pre_explore: bool = True,
    output_dir: str = "gdo_plots",
) -> Dict[str, ExperimentResult]:
    """Run all optimizers on same task, same init, collect results."""
    results: Dict[str, ExperimentResult] = {}

    # Get reference init state
    torch.manual_seed(config.seed)
    ref_model = model_factory()
    init_state = {k: v.clone() for k, v in ref_model.state_dict().items()}
    algebra = getattr(ref_model, 'algebra', None)
    del ref_model

    for opt_name in optimizers:
        print(f"\n  --- {opt_name.upper()} ---")
        torch.manual_seed(config.seed)
        model = model_factory()
        model.load_state_dict(init_state)
        loss_fn = loss_factory(model)
        metric_fn = metric_factory(model) if metric_factory else None

        if opt_name == 'gdo':
            # Pre-exploration for GDO
            if pre_explore and algebra is not None:
                try:
                    pre_analyzer = PreExplorationAnalyzer(
                        algebra=algebra, n_samples=100, device=config.device)
                    pre_result = pre_analyzer.analyze(model, loss_fn)
                    print(f"  Strategy: {pre_result.strategy_label}")
                    gdo_config = pre_result.recommended_config
                except Exception:
                    gdo_config = GDOConfig(lr=config.lr)
            else:
                gdo_config = config.gdo_config or GDOConfig(lr=config.lr)

            controller_or_opt, label = create_optimizer(
                'gdo', model, config.lr, algebra=algebra,
                loss_fn=loss_fn, config=gdo_config, device=config.device)
            result = train_loop_gdo(
                model, controller_or_opt, config.steps,
                metric_fn=metric_fn, log_interval=max(config.steps // 5, 1))
        else:
            opt, label = create_optimizer(
                opt_name, model, config.lr, algebra=algebra, device=config.device)
            result = train_loop_standard(
                model, opt, loss_fn, config.steps,
                metric_fn=metric_fn, log_interval=max(config.steps // 5, 1), label=label)

        result.name = task_name
        result.optimizer_name = label
        results[label] = result

    return results


# ======================================================================
# Visualization (existing + new)
# ======================================================================

def _ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def plot_pre_exploration(
    pre_result: PreExplorationResult,
    title: str = "Pre-Exploration Analysis",
    output_dir: str = "gdo_plots",
):
    """2x3 dashboard: eigenvalues, local dims, grade energy, coherence, geometry, config."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    if pre_result.dim_result is not None:
        ev = pre_result.dim_result.eigenvalues.cpu().numpy()
        ax.semilogy(range(1, len(ev) + 1), ev, 'b.-')
        ax.axhline(y=ev[pre_result.dim_result.broken_stick_threshold - 1]
                    if pre_result.dim_result.broken_stick_threshold > 0
                    else ev[-1],
                    color='r', linestyle='--', alpha=0.7, label='broken-stick')
        ax.set_title(f"Eigenvalue Spectrum\n"
                     f"intrinsic_dim={pre_result.dim_result.intrinsic_dim}, "
                     f"PR={pre_result.dim_result.participation_ratio:.1f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No dimension\nanalysis", ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if pre_result.dim_result is not None and pre_result.dim_result.local_dims is not None:
        ld = pre_result.dim_result.local_dims.cpu().numpy()
        ax.hist(ld, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(pre_result.dim_result.participation_ratio, color='red',
                   linestyle='--', label=f'PR={pre_result.dim_result.participation_ratio:.1f}')
        ax.set_title("Local Dimension Distribution")
        ax.legend()
    else:
        ls = pre_result.loss_statistics
        if ls:
            vals = [ls["min"], ls["mean"], ls["max"]]
            labels = ["min", "mean", "max"]
            ax.barh(labels, vals, color=['green', 'steelblue', 'red'], alpha=0.7)
            ax.set_title("Loss Statistics")
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    gs = pre_result.geometric_scores
    if pre_result.spectral_result is not None:
        ge = pre_result.spectral_result.grade_energy.cpu().numpy()
        grades = list(range(len(ge)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ge)))
        ax.bar(grades, ge, color=colors, edgecolor='white')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Energy")
        ax.set_title("Grade Energy Spectrum")
    elif "grade_energy" in gs:
        ge = gs["grade_energy"].cpu().numpy()
        grades = list(range(len(ge)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ge)))
        ax.bar(grades, ge, color=colors, edgecolor='white')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Energy")
        ax.set_title("Grade Energy Spectrum")
    else:
        ax.text(0.5, 0.5, "No spectral\nanalysis", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    metrics_names = ["Coherence", "Curvature"]
    values = [pre_result.landscape_coherence, pre_result.landscape_curvature]
    bar_colors = []
    for v, name in zip(values, metrics_names):
        if name == "Coherence":
            bar_colors.append('green' if v > 0.5 else ('orange' if v > 0.3 else 'red'))
        else:
            bar_colors.append('green' if v < 0.3 else ('orange' if v < 0.5 else 'red'))
    ax.barh(metrics_names, values, color=bar_colors, alpha=0.7, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_title(f"Landscape Geometry\nStrategy: {pre_result.strategy_label}")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels_gs, vals_gs, colors_gs = [], [], []
    if "closure_error" in gs:
        labels_gs.append("Lie Closure\nError")
        vals_gs.append(gs["closure_error"])
        ce = gs["closure_error"]
        colors_gs.append('green' if ce < 0.1 else ('orange' if ce < 0.5 else 'red'))
    if "grade_entropy" in gs:
        labels_gs.append("Grade\nEntropy")
        vals_gs.append(gs["grade_entropy"])
        colors_gs.append('purple')
    if "coherence" in gs:
        labels_gs.append("Bivector\nCoherence")
        vals_gs.append(gs["coherence"])
        colors_gs.append('steelblue')
    if labels_gs:
        ax.barh(labels_gs, vals_gs, color=colors_gs, alpha=0.7, edgecolor='white')
        ax.set_xlim(0, max(1.0, max(vals_gs) * 1.1))
        ax.set_title("Geometric Signals")
    else:
        ax.text(0.5, 0.5, "No geometric\nscores", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis('off')
    cfg = pre_result.recommended_config
    lines = [
        f"lr: {cfg.lr}",
        f"probe_interval: {cfg.probe_interval}",
        f"topology_interval: {cfg.topology_interval}",
        f"sprint_after: {cfg.sprint_after}",
        f"lift_patience: {cfg.lift_patience}",
        f"lift_sigma: {cfg.lift_sigma}",
        f"lorentz_max_beta: {cfg.lorentz_max_beta}",
        f"commutator_threshold: {cfg.commutator_threshold}",
    ]
    ax.text(0.05, 0.95, "Recommended Config\n" + "-" * 25 + "\n" + "\n".join(lines),
            transform=ax.transAxes, va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, f"pre_exploration_{title.replace(' ', '_').lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_optimization_trajectory(
    history: Dict,
    title: str = "Optimization Trajectory",
    output_dir: str = "gdo_plots",
):
    """Loss curve, probe results, landscape map summary."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    losses = history.get("losses", [])
    modes = history.get("modes", [])
    if losses:
        ax.semilogy(losses, 'b-', linewidth=0.8)
        mode_colors = {"explore": "#cce5ff", "navigate": "#d4edda", "sprint": "#fff3cd"}
        if modes:
            prev = modes[0]
            start = 0
            for i, m in enumerate(modes + [None]):
                if m != prev or i == len(modes):
                    ax.axvspan(start, i, alpha=0.15,
                               color=mode_colors.get(prev, '#ffffff'))
                    prev = m
                    start = i
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    probe_steps = history.get("probe_steps", [])
    curvatures = history.get("curvatures", [])
    grad_norms = history.get("grad_norms", [])
    if probe_steps and curvatures:
        ax.plot(probe_steps, curvatures, 'b.-', label='Mean curvature')
        ax2 = ax.twinx()
        ax2.plot(probe_steps, grad_norms, 'r.-', alpha=0.7, label='Grad norm')
        ax2.set_ylabel("Grad Norm", color='red')
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Curvature", color='blue')
        ax.set_title("Probe Results")
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, "No probe data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    betas = history.get("betas", [])
    if probe_steps and betas:
        ax.plot(probe_steps, betas, 'g.-')
        ax.set_xlabel("Step")
        ax.set_ylabel("Lorentz beta")
        ax.set_title("Lorentz Warp Factor")
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No warp data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    lifts = history.get("lifts", [])
    if lifts:
        lift_steps = [l["step"] for l in lifts]
        lift_losses = [l["loss"] for l in lifts]
        lift_colors = ['green' if l["success"] else 'red' for l in lifts]
        ax.scatter(lift_steps, lift_losses, c=lift_colors, s=50, zorder=5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss at Lift")
        ax.set_title(f"Lift Oracle Events ({len(lifts)} total)")
        patches = [Patch(color='green', label='Success'),
                   Patch(color='red', label='Fail')]
        ax.legend(handles=patches, fontsize=8)
    else:
        if modes:
            mode_counts = {}
            for m in modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            ax.bar(mode_counts.keys(), mode_counts.values(),
                   color=['steelblue', 'seagreen', 'orange'][:len(mode_counts)])
            ax.set_title("Mode Distribution")
        else:
            ax.text(0.5, 0.5, "No lift/mode data", ha='center', va='center',
                    transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"trajectory_{title.replace(' ', '_').lower()}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_geometric_controller(
    diagnostics: Dict,
    title: str = "Geometric Parameter Controller",
    output_dir: str = "gdo_plots",
):
    """2x2 dashboard: FIM, commutativity heatmap, group scales, grade energy."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    fim = diagnostics.get("fim_diag", {})
    if fim:
        groups = sorted(fim.keys())
        means = [fim[g].mean().item() for g in groups]
        ax.bar(groups, means, color='steelblue', edgecolor='white')
        ax.set_xlabel("Param Group")
        ax.set_ylabel("Mean FIM")
        ax.set_title("Fisher Information (per group)")
    else:
        ax.text(0.5, 0.5, "No FIM data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    gs = diagnostics.get("geometric_scores", {})
    if "comm_result" in gs:
        mat = gs["comm_result"].commutativity_matrix.cpu().numpy()
        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Commutativity Matrix")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Dimension")
        schedule = diagnostics.get("schedule", [])
        sched_colors = plt.cm.Set2(np.linspace(0, 1, max(len(schedule), 1)))
        for ci, color_group in enumerate(schedule):
            for g in color_group:
                if g < mat.shape[0]:
                    ax.axhline(y=g, color=sched_colors[ci], linewidth=2, alpha=0.5)
    else:
        hybrid = diagnostics.get("hybrid_scores", {})
        if hybrid:
            n = max(max(k) for k in hybrid.keys()) + 1 if hybrid else 1
            mat = np.zeros((n, n))
            for (i, j), v in hybrid.items():
                mat[i, j] = v
                mat[j, i] = v
            im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title("Hybrid Commutativity Scores")
        else:
            ax.text(0.5, 0.5, "No commutativity\ndata", ha='center', va='center',
                    transform=ax.transAxes)
    ax.grid(False)

    ax = axes[1, 0]
    scales = diagnostics.get("scales", [])
    if scales:
        x = list(range(len(scales)))
        bar_colors = []
        for s in scales:
            if s > 1.3:
                bar_colors.append('green')
            elif s < 0.5:
                bar_colors.append('red')
            elif s < 0.8:
                bar_colors.append('orange')
            else:
                bar_colors.append('steelblue')
        ax.bar(x, scales, color=bar_colors, edgecolor='white')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Param Group")
        ax.set_ylabel("Update Scale")
        ax.set_title("Group Update Scales\n(green=trust, red=caution)")
    else:
        ax.text(0.5, 0.5, "No scale data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "grade_energy" in gs:
        ge = gs["grade_energy"].cpu().numpy()
        grades = list(range(len(ge)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ge)))
        ax.bar(grades, ge, color=colors, edgecolor='white')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Energy")
        title_parts = ["Grade Energy"]
        if "grade_entropy" in gs:
            title_parts.append(f"H={gs['grade_entropy']:.3f}")
        if "closure_error" in gs:
            title_parts.append(f"closure={gs['closure_error']:.3f}")
        ax.set_title(" | ".join(title_parts))
    else:
        ax.text(0.5, 0.5, "No grade energy\ndata", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "geometric_controller.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_topology_map(
    model: nn.Module,
    landscape: LandscapeMap,
    trajectory: List[Tuple[float, float]],
    modes: List[str],
    output_dir: str = "gdo_plots",
):
    """Contour plot of 2D loss surface + critical points + trajectory."""
    _ensure_output_dir(output_dir)
    if not hasattr(model, 'a'):
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    xr = np.linspace(-2.5, 2.5, 100)
    yr = np.linspace(-1.5, 3.5, 100)
    X, Y = np.meshgrid(xr, yr)
    Z = (model.a - X) ** 2 + model.b * (Y - X ** 2) ** 2
    ax.contourf(X, Y, np.log10(Z + 1e-10), levels=30, cmap='terrain', alpha=0.7)
    ax.contour(X, Y, np.log10(Z + 1e-10), levels=15, colors='gray',
               linewidths=0.3, alpha=0.5)

    cp_markers = {
        CriticalPointType.MINIMUM: ('o', 'blue', 'Minimum'),
        CriticalPointType.SADDLE: ('^', 'red', 'Saddle'),
        CriticalPointType.MAXIMUM: ('s', 'gray', 'Maximum'),
    }
    for cp in landscape.critical_points:
        if cp.params.shape[0] >= 2:
            marker, color, cp_label = cp_markers.get(
                cp.point_type, ('x', 'black', 'Unknown'))
            ax.scatter(cp.params[0].item(), cp.params[1].item(),
                       marker=marker, c=color, s=80, zorder=5,
                       edgecolors='white', linewidths=1, label=cp_label)

    if trajectory:
        mode_colors_traj = {"explore": "blue", "navigate": "green", "sprint": "orange"}
        for i in range(1, len(trajectory)):
            m = modes[i] if i < len(modes) else "explore"
            ax.plot([trajectory[i - 1][0], trajectory[i][0]],
                    [trajectory[i - 1][1], trajectory[i][1]],
                    color=mode_colors_traj.get(m, "blue"), linewidth=0.5, alpha=0.7)
        ax.scatter(*trajectory[0], marker='*', c='lime', s=150, zorder=6,
                   edgecolors='black', label='Start')
        ax.scatter(*trajectory[-1], marker='*', c='red', s=150, zorder=6,
                   edgecolors='black', label='End')
        ax.scatter(1.0, 1.0, marker='D', c='gold', s=100, zorder=6,
                   edgecolors='black', label='Optimum')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Loss Landscape & Optimization Trajectory")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "topology_map.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# --- New Enhanced Visualizations ---

def plot_three_way_comparison(
    results: Dict[str, ExperimentResult],
    title: str = "Optimizer Comparison",
    output_dir: str = "gdo_plots",
):
    """4-panel: loss curves, final loss bars, wall time bars, convergence rate."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {
        "GDO": ("b-", 2.0), "RiemannianAdam": ("r--", 1.5),
        "Adam": ("g:", 1.2), "Adam (no algebra)": ("g:", 1.2),
        "ExponentialSGD": ("m-.", 1.2), "SGD": ("m-.", 1.2),
    }
    color_map = {
        "GDO": "steelblue", "RiemannianAdam": "salmon",
        "Adam": "seagreen", "Adam (no algebra)": "seagreen",
        "ExponentialSGD": "plum", "SGD": "plum",
    }

    # (0,0) Loss curves
    ax = axes[0, 0]
    for name, res in results.items():
        style, lw = styles.get(name, ("k-", 1.0))
        ax.semilogy(res.losses, style, label=name, linewidth=lw)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Final loss bars
    ax = axes[0, 1]
    names = list(results.keys())
    finals = [results[n].final_loss for n in names]
    bar_colors = [color_map.get(n, 'gray') for n in names]
    ax.bar(names, finals, color=bar_colors, edgecolor='white')
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(finals):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    # (1,0) Wall time bars
    ax = axes[1, 0]
    wall_times = [results[n].total_wall_time for n in names]
    ax.bar(names, wall_times, color=bar_colors, edgecolor='white')
    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title("Wall Time")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(wall_times):
        ax.text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontsize=8)

    # (1,1) Loss vs wall time
    ax = axes[1, 1]
    for name, res in results.items():
        cum_time = np.cumsum(res.wall_times)
        style, lw = styles.get(name, ("k-", 1.0))
        ax.semilogy(cum_time, res.losses, style, label=name, linewidth=lw)
    ax.set_xlabel("Cumulative Wall Time (s)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Wall Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace('/', '_').lower()
    path = os.path.join(output_dir, f"comparison_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_convergence_rate(
    results: Dict[str, ExperimentResult],
    title: str = "Convergence Rate",
    output_dir: str = "gdo_plots",
):
    """3-panel: loss vs step, loss vs wall-time, convergence rate."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {"GDO": "b-", "RiemannianAdam": "r--", "Adam": "g:", "Adam (no algebra)": "g:"}

    ax = axes[0]
    for name, res in results.items():
        ax.semilogy(res.losses, styles.get(name, "k-"), label=name, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, res in results.items():
        cum_time = np.cumsum(res.wall_times)
        ax.semilogy(cum_time, res.losses, styles.get(name, "k-"), label=name, linewidth=1.2)
    ax.set_xlabel("Wall Time (s)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Wall Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    window = 50
    for name, res in results.items():
        if len(res.losses) > window:
            losses_arr = np.array(res.losses)
            rate = -(losses_arr[window:] - losses_arr[:-window]) / window
            smoothed = np.convolve(rate, np.ones(20)/20, mode='valid')
            ax.plot(smoothed, styles.get(name, "k-"), label=name, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Convergence Rate (loss drop/step)")
    ax.set_title("Smoothed Convergence Rate")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"convergence_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_timing_breakdown(
    results: Dict[str, ExperimentResult],
    title: str = "Timing Breakdown",
    output_dir: str = "gdo_plots",
):
    """2-panel: per-step wall time, cumulative time vs loss."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {"GDO": "b", "RiemannianAdam": "r", "Adam": "g", "Adam (no algebra)": "g"}

    ax = axes[0]
    for name, res in results.items():
        wt = np.array(res.wall_times) * 1000  # ms
        # Smooth for readability
        window = max(1, len(wt) // 100)
        if window > 1:
            wt_smooth = np.convolve(wt, np.ones(window)/window, mode='valid')
        else:
            wt_smooth = wt
        ax.plot(wt_smooth, color=styles.get(name, 'k'), label=name, linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Wall Time (ms)")
    ax.set_title("Per-Step Wall Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, res in results.items():
        cum_time = np.cumsum(res.wall_times)
        ax.plot(cum_time, res.losses, color=styles.get(name, 'k'), label=name, linewidth=1.2)
    ax.set_xlabel("Cumulative Time (s)")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.set_title("Cumulative Time vs Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"timing_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_bivector_trajectory(
    results: Dict[str, ExperimentResult],
    title: str = "Bivector Trajectory",
    output_dir: str = "gdo_plots",
):
    """Bivector param norm evolution across optimizers."""
    _ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    styles = {"GDO": "b-", "RiemannianAdam": "r--", "Adam": "g:", "Adam (no algebra)": "g:"}
    for name, res in results.items():
        if res.bivector_norms:
            ax.plot(res.bivector_norms, styles.get(name, "k-"), label=name, linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Bivector Param Norm")
    ax.set_title("Bivector Parameter Evolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"bivector_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_optimizer_state_dashboard(
    gdo_result: ExperimentResult,
    title: str = "GDO State Dashboard",
    output_dir: str = "gdo_plots",
):
    """4-panel: mode timeline, topology summary, warp beta/gamma, lift events."""
    _ensure_output_dir(output_dir)
    diag = gdo_result.gdo_diagnostics
    if diag is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # (0,0) Mode timeline
    ax = axes[0, 0]
    mode_hist = gdo_result.mode_history or diag.get("mode_history", [])
    if mode_hist:
        mode_to_int = {"explore": 0, "navigate": 1, "sprint": 2}
        mode_ints = [mode_to_int.get(m, 0) for m in mode_hist]
        mode_colors = {0: '#4a90d9', 1: '#66bb6a', 2: '#ffa726'}
        for i in range(len(mode_ints)):
            ax.bar(i, 1, color=mode_colors.get(mode_ints[i], 'gray'), width=1.0)
        ax.set_xlabel("Step")
        ax.set_yticks([])
        ax.set_title("Mode Timeline (blue=explore, green=navigate, orange=sprint)")
    else:
        ax.text(0.5, 0.5, "No mode data", ha='center', va='center', transform=ax.transAxes)

    # (0,1) Topology summary
    ax = axes[0, 1]
    topo = diag.get("topology_map", {})
    ax.axis('off')
    lines = [
        f"Critical points detected: {topo.get('critical_points', 0)}",
        f"Plateau episodes: {len(topo.get('plateau_episodes', []))}",
        f"Curvature samples: {len(topo.get('curvature_history', []))}",
    ]
    warp = diag.get("warp", {})
    lines.append(f"\nWarp beta: {warp.get('beta', 0):.4f}")
    lines.append(f"Warp gamma: {warp.get('gamma', 1):.4f}")
    lines.append(f"On plateau: {warp.get('on_plateau', False)}")

    lift = diag.get("lift_oracle", {})
    lines.append(f"\nLift count: {lift.get('lift_count', 0)}")
    lines.append(f"Consecutive fails: {lift.get('consecutive_fails', 0)}")
    lines.append(f"Current sigma: {lift.get('current_sigma', 0):.4f}")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    ax.set_title("GDO State Summary")

    # (1,0) Curvature history
    ax = axes[1, 0]
    curv_hist = topo.get("curvature_history", [])
    if curv_hist:
        ax.plot(curv_hist, 'b.-', linewidth=0.8)
        ax.set_xlabel("Probe Index")
        ax.set_ylabel("Mean Curvature")
        ax.set_title("Curvature History")
    else:
        ax.text(0.5, 0.5, "No curvature data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # (1,1) Gradient norm history
    ax = axes[1, 1]
    grad_hist = topo.get("gradient_norm_history", [])
    if grad_hist:
        ax.semilogy(grad_hist, 'r.-', linewidth=0.8)
        ax.set_xlabel("Probe Index")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm History")
    else:
        ax.text(0.5, 0.5, "No gradient data", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"gdo_state_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_loss_landscape_2d_slice(
    model: nn.Module,
    loss_fn: Callable,
    n_grid: int = 40,
    radius: float = 1.0,
    title: str = "Loss Landscape Slice",
    output_dir: str = "gdo_plots",
):
    """Contour plot along 2 random orthogonal directions in param space."""
    _ensure_output_dir(output_dir)
    params = list(model.parameters())
    flat_center = torch.cat([p.detach().reshape(-1) for p in params])
    n = flat_center.shape[0]
    device = flat_center.device

    # Two random orthogonal directions
    d1 = F.normalize(torch.randn(n, device=device), dim=0)
    d2 = torch.randn(n, device=device)
    d2 = d2 - (d2 @ d1) * d1
    d2 = F.normalize(d2, dim=0)

    alphas = np.linspace(-radius, radius, n_grid)
    betas = np.linspace(-radius, radius, n_grid)
    Z = np.zeros((n_grid, n_grid))

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                flat_p = flat_center + a * d1 + b * d2
                idx = 0
                for p in params:
                    sz = p.numel()
                    p.data.copy_(flat_p[idx:idx+sz].reshape(p.shape))
                    idx += sz
                Z[j, i] = loss_fn().item()

    # Restore
    idx = 0
    for p in params:
        sz = p.numel()
        p.data.copy_(flat_center[idx:idx+sz].reshape(p.shape))
        idx += sz

    fig, ax = plt.subplots(figsize=(8, 7))
    A, B = np.meshgrid(alphas, betas)
    cs = ax.contourf(A, B, np.log10(Z + 1e-10), levels=30, cmap='viridis')
    fig.colorbar(cs, ax=ax, label='log10(loss)')
    ax.scatter([0], [0], c='red', s=100, marker='*', zorder=5, label='Current')
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    safe_title = title.replace(' ', '_').lower()
    path = os.path.join(output_dir, f"landscape_{safe_title}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


# ======================================================================
# Analysis Infrastructure
# ======================================================================

def compute_convergence_metrics(result: ExperimentResult) -> Dict:
    """Steps to 50%/90%/99% of total improvement, AUC."""
    losses = np.array(result.losses)
    if len(losses) < 2:
        return {}
    initial = losses[0]
    final = losses[-1]
    total_improvement = initial - final
    if total_improvement <= 0:
        return {"auc": float(losses.sum()), "improvement": 0.0}

    metrics = {"auc": float(losses.sum()), "improvement": float(total_improvement)}
    for pct, label in [(0.5, "steps_to_50pct"), (0.9, "steps_to_90pct"), (0.99, "steps_to_99pct")]:
        threshold = initial - pct * total_improvement
        reached = np.where(losses <= threshold)[0]
        metrics[label] = int(reached[0]) if len(reached) > 0 else len(losses)

    return metrics


def compute_overhead_ratio(
    gdo_result: ExperimentResult,
    baseline_result: ExperimentResult,
) -> float:
    """Wall-time ratio GDO / baseline."""
    if baseline_result.total_wall_time < 1e-6:
        return float('inf')
    return gdo_result.total_wall_time / baseline_result.total_wall_time


def analyze_experiment_results(
    all_results: Dict[str, Dict[str, ExperimentResult]],
    output_dir: str = "gdo_plots",
) -> str:
    """Cross-experiment analysis. Returns formatted report text."""
    lines = []
    lines.append("=" * 70)
    lines.append("GDO ANALYSIS REPORT")
    lines.append("=" * 70)

    # 1. Win/loss matrix
    lines.append("\n1. WIN/LOSS MATRIX (lowest final loss)")
    lines.append("-" * 50)
    wins = {}
    for task_name, task_results in all_results.items():
        if not task_results:
            continue
        best_name = min(task_results.keys(), key=lambda k: task_results[k].final_loss)
        wins.setdefault(best_name, []).append(task_name)
        final_strs = [f"  {k}: {v.final_loss:.6f}" for k, v in task_results.items()]
        lines.append(f"\n  {task_name}:")
        lines.extend(final_strs)
        lines.append(f"  Winner: {best_name}")

    lines.append("\n  Summary:")
    for opt_name, tasks in sorted(wins.items()):
        lines.append(f"    {opt_name}: {len(tasks)} wins ({', '.join(tasks)})")

    # 2. Convergence speed
    lines.append("\n2. CONVERGENCE SPEED")
    lines.append("-" * 50)
    for task_name, task_results in all_results.items():
        lines.append(f"\n  {task_name}:")
        for opt_name, res in task_results.items():
            cm = compute_convergence_metrics(res)
            if cm:
                lines.append(f"    {opt_name}: 50%@{cm.get('steps_to_50pct','?')}, "
                             f"90%@{cm.get('steps_to_90pct','?')}, "
                             f"99%@{cm.get('steps_to_99pct','?')}")

    # 3. Wall-time efficiency
    lines.append("\n3. WALL-TIME EFFICIENCY")
    lines.append("-" * 50)
    for task_name, task_results in all_results.items():
        lines.append(f"\n  {task_name}:")
        for opt_name, res in task_results.items():
            ms_per_step = (res.total_wall_time / max(len(res.losses), 1)) * 1000
            lines.append(f"    {opt_name}: {res.total_wall_time:.1f}s total, "
                         f"{ms_per_step:.1f}ms/step")

    # 4. Overhead ratio
    lines.append("\n4. GDO OVERHEAD RATIO (vs RiemannianAdam)")
    lines.append("-" * 50)
    for task_name, task_results in all_results.items():
        gdo_res = task_results.get("GDO")
        riem_res = task_results.get("RiemannianAdam")
        if gdo_res and riem_res:
            ratio = compute_overhead_ratio(gdo_res, riem_res)
            lines.append(f"  {task_name}: {ratio:.2f}x")

    # 5. When GDO wins/loses
    lines.append("\n5. GDO ADVANTAGE ANALYSIS")
    lines.append("-" * 50)
    gdo_better = []
    gdo_worse = []
    for task_name, task_results in all_results.items():
        gdo_res = task_results.get("GDO")
        riem_res = task_results.get("RiemannianAdam")
        if gdo_res and riem_res:
            if gdo_res.final_loss < riem_res.final_loss * 0.95:
                gdo_better.append(task_name)
            elif gdo_res.final_loss > riem_res.final_loss * 1.05:
                gdo_worse.append(task_name)
    lines.append(f"  GDO better (>5% lower loss): {gdo_better or 'none'}")
    lines.append(f"  GDO worse (>5% higher loss): {gdo_worse or 'none'}")

    report = "\n".join(lines)

    # Save report
    _ensure_output_dir(output_dir)
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    return report


# ======================================================================
# History helpers (for legacy GDO-only mode)
# ======================================================================

def _new_history() -> Dict:
    return {
        "losses": [], "modes": [], "probe_steps": [],
        "curvatures": [], "grad_norms": [], "betas": [],
        "lifts": [], "plateaus": [], "trajectory": [],
        "angle_errors": [],
    }


def _collect_history(controller, info, history):
    history["losses"].append(info["loss"])
    history["modes"].append(info["mode"])
    if "probe" in info:
        history["probe_steps"].append(info["step"])
        history["curvatures"].append(info["probe"]["mean_curvature"])
        history["grad_norms"].append(info["probe"]["grad_norm"])
        history["betas"].append(info["probe"]["beta"])
    if "lift_oracle" in info:
        history["lifts"].append({
            "step": info["step"],
            "loss": info["loss"],
            "success": "improved" in str(info["lift_oracle"]),
            "sigma": getattr(controller.lift_oracle, '_current_sigma', 0),
        })


# ======================================================================
# Experiment Runners
# ======================================================================

@register_experiment("rosenbrock", "analytic")
def run_rosenbrock(steps: int = 2000, optimizers=('gdo', 'riemannian_adam', 'adam'),
                   seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Rosenbrock Function (a=1, b=100)")
    print("=" * 60)

    config = ExperimentConfig(name="rosenbrock", category="analytic",
                              steps=steps, lr=1e-3, seed=seed, device=device)
    results = run_comparison(
        "rosenbrock",
        model_factory=RosenbrockModel,
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Rosenbrock", output_dir=output_dir)
    plot_convergence_rate(results, title="Rosenbrock", output_dir=output_dir)
    plot_timing_breakdown(results, title="Rosenbrock", output_dir=output_dir)
    return results


@register_experiment("rastrigin", "analytic")
def run_rastrigin(n_dims: int = 8, steps: int = 3000,
                  optimizers=('gdo', 'riemannian_adam', 'adam'),
                  seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Rastrigin Function ({n_dims}D)")
    print("=" * 60)

    config = ExperimentConfig(name="rastrigin", category="analytic",
                              steps=steps, lr=1e-2, seed=seed, device=device)
    results = run_comparison(
        "rastrigin",
        model_factory=lambda: RastriginModel(n_dims=n_dims),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title=f"Rastrigin {n_dims}D", output_dir=output_dir)
    plot_convergence_rate(results, title=f"Rastrigin {n_dims}D", output_dir=output_dir)
    return results


@register_experiment("ackley", "analytic")
def run_ackley(n_dims: int = 10, steps: int = 3000,
               optimizers=('gdo', 'riemannian_adam', 'adam'),
               seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Ackley Function ({n_dims}D)")
    print("=" * 60)

    config = ExperimentConfig(name="ackley", category="analytic",
                              steps=steps, lr=1e-2, seed=seed, device=device)
    results = run_comparison(
        "ackley",
        model_factory=lambda: AckleyModel(n_dims=n_dims),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title=f"Ackley {n_dims}D", output_dir=output_dir)
    plot_convergence_rate(results, title=f"Ackley {n_dims}D", output_dir=output_dir)
    return results


@register_experiment("styblinski_tang", "analytic")
def run_styblinski_tang(n_dims: int = 6, steps: int = 2000,
                        optimizers=('gdo', 'riemannian_adam', 'adam'),
                        seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Styblinski-Tang Function ({n_dims}D)")
    print("=" * 60)

    config = ExperimentConfig(name="styblinski_tang", category="analytic",
                              steps=steps, lr=5e-3, seed=seed, device=device)
    results = run_comparison(
        "styblinski_tang",
        model_factory=lambda: StyblinskiTangModel(n_dims=n_dims),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title=f"Styblinski-Tang {n_dims}D", output_dir=output_dir)
    return results


@register_experiment("registration", "geometric")
def run_registration(steps: int = 1500, noise_std: float = 0.05, rotation_angle: float = 2.5,
                     optimizers=('gdo', 'riemannian_adam', 'adam'),
                     seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Rotor Registration (Cl(3,0), angle={rotation_angle:.2f} rad)")
    print("=" * 60)

    config = ExperimentConfig(name="registration", category="geometric",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "registration",
        model_factory=lambda: RotorRegistrationModel(
            noise_std=noise_std, rotation_angle=rotation_angle, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers,
        metric_factory=lambda m: lambda: {"angle_error": m.angular_error()},
        output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Rotor Registration", output_dir=output_dir)
    plot_convergence_rate(results, title="Rotor Registration", output_dir=output_dir)
    plot_timing_breakdown(results, title="Rotor Registration", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Rotor Registration", output_dir=output_dir)
    return results


@register_experiment("minkowski_rotor", "geometric")
def run_minkowski_rotor(steps: int = 1500,
                        optimizers=('gdo', 'riemannian_adam', 'adam'),
                        seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Minkowski Rotor Registration (Cl(2,1))")
    print("=" * 60)

    config = ExperimentConfig(name="minkowski_rotor", category="geometric",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(2, 1))
    results = run_comparison(
        "minkowski_rotor",
        model_factory=lambda: MinkowskiRotorModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers,
        metric_factory=lambda m: lambda: {"rapidity_error": m.rapidity_error()},
        output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Minkowski Rotor Cl(2,1)", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Minkowski Rotor", output_dir=output_dir)
    return results


@register_experiment("conformal_registration", "geometric")
def run_conformal_registration(steps: int = 2000,
                               optimizers=('gdo', 'riemannian_adam', 'adam'),
                               seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Conformal Registration (Cl(4,1), 32D)")
    print("=" * 60)

    config = ExperimentConfig(name="conformal_registration", category="geometric",
                              steps=steps, lr=5e-4, seed=seed, device=device,
                              algebra_sig=(4, 1))
    results = run_comparison(
        "conformal_registration",
        model_factory=lambda: ConformalRegistrationModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Conformal Cl(4,1)", output_dir=output_dir)
    return results


@register_experiment("multi_rotor", "geometric")
def run_multi_rotor(steps: int = 2000,
                    optimizers=('gdo', 'riemannian_adam', 'adam'),
                    seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Multi-Rotor Registration (Cl(3,0), 3 clusters)")
    print("=" * 60)

    config = ExperimentConfig(name="multi_rotor", category="geometric",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "multi_rotor",
        model_factory=lambda: MultiRotorRegistrationModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Multi-Rotor", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Multi-Rotor", output_dir=output_dir)
    return results


@register_experiment("gbn_small", "ga_neural")
def run_gbn_small(steps: int = 200,
                  optimizers=('gdo', 'riemannian_adam', 'adam'),
                  seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Small GBN (Cl(3,0), 4ch)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_small", category="ga_neural",
                              steps=steps, lr=5e-4, seed=seed, device=device,
                              algebra_sig=(3, 0))

    def model_factory():
        return SmallGBNModel(p=3, q=0, channels=4, device=device)

    def loss_factory(model):
        X = torch.randn(32, 4, model._dim, device=device) * 0.3
        y_target = X[:, :, 0].mean(dim=1, keepdim=True)
        def loss_fn():
            out = model(X)
            pred = out[:, :, 0].mean(dim=1, keepdim=True)
            return F.mse_loss(pred, y_target)
        return loss_fn

    results = run_comparison(
        "gbn_small", model_factory=model_factory, loss_factory=loss_factory,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Small GBN", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Small GBN", output_dir=output_dir)

    # GDO state dashboard
    gdo_res = results.get("GDO")
    if gdo_res:
        plot_optimizer_state_dashboard(gdo_res, title="Small GBN GDO", output_dir=output_dir)
    return results


@register_experiment("gbn_medium", "ga_neural")
def run_gbn_medium(steps: int = 300,
                   optimizers=('gdo', 'riemannian_adam', 'adam'),
                   seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Medium GBN (Cl(3,0), 16ch, 3 layers)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_medium", category="ga_neural",
                              steps=steps, lr=3e-4, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "gbn_medium",
        model_factory=lambda: MediumGBNModel(channels=16, layers=3, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Medium GBN", output_dir=output_dir)
    plot_convergence_rate(results, title="Medium GBN", output_dir=output_dir)
    plot_timing_breakdown(results, title="Medium GBN", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Medium GBN", output_dir=output_dir)
    return results


@register_experiment("gbn_multisig", "ga_neural")
def run_gbn_multisig(steps: int = 250,
                     optimizers=('gdo', 'riemannian_adam', 'adam'),
                     seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Minkowski GBN (Cl(2,1), 8ch)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_multisig", category="ga_neural",
                              steps=steps, lr=3e-4, seed=seed, device=device,
                              algebra_sig=(2, 1))
    results = run_comparison(
        "gbn_multisig",
        model_factory=lambda: MultiSigGBNModel(channels=8, layers=2, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Minkowski GBN Cl(2,1)", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Minkowski GBN", output_dir=output_dir)
    return results


@register_experiment("gbn_deep", "ga_neural")
def run_gbn_deep(steps: int = 300,
                 optimizers=('gdo', 'riemannian_adam', 'adam'),
                 seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Deep GBN (Cl(3,0), 16ch, 5 layers)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_deep", category="ga_neural",
                              steps=steps, lr=2e-4, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "gbn_deep",
        model_factory=lambda: DeepGBNModel(channels=32, layers=5, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Deep GBN", output_dir=output_dir)
    plot_convergence_rate(results, title="Deep GBN", output_dir=output_dir)
    plot_timing_breakdown(results, title="Deep GBN", output_dir=output_dir)
    return results


@register_experiment("so3_interpolation", "manifold")
def run_so3_interpolation(steps: int = 1500,
                          optimizers=('gdo', 'riemannian_adam', 'adam'),
                          seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: SO(3) Rotor Interpolation")
    print("=" * 60)

    config = ExperimentConfig(name="so3_interpolation", category="manifold",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "so3_interpolation",
        model_factory=lambda: SO3InterpolationModel(n_waypoints=8, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers,
        metric_factory=lambda m: lambda: {"geodesic_deviation": m.geodesic_deviation()},
        output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="SO(3) Interpolation", output_dir=output_dir)
    plot_convergence_rate(results, title="SO(3) Interpolation", output_dir=output_dir)
    plot_bivector_trajectory(results, title="SO(3) Interpolation", output_dir=output_dir)
    return results


# ======================================================================
# Category / Full Runners
# ======================================================================

def run_category(category: str, **kwargs):
    """Run all experiments in a category."""
    results = {}
    for name, (fn, cat) in EXPERIMENT_REGISTRY.items():
        if cat == category:
            results[name] = fn(**kwargs)
    return results


def run_all_experiments(**kwargs):
    """Run all registered experiments and produce analysis report."""
    all_results = {}
    for name, (fn, _cat) in EXPERIMENT_REGISTRY.items():
        try:
            all_results[name] = fn(**kwargs)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    return all_results


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    registered = list(EXPERIMENT_REGISTRY.keys())
    all_choices = registered + [
        "all", "analytic", "geometric", "ga_neural", "manifold", "compare_all",
    ]
    p = argparse.ArgumentParser(
        description="Geometric Deterministic Optimizer (GDO) Experiment Suite")
    p.add_argument("--task", choices=all_choices, default="rosenbrock")
    p.add_argument("--optimizers", nargs="+",
                   default=["gdo", "riemannian_adam", "adam"],
                   help="Optimizers to compare")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--n-dims", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="gdo_plots")
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--rotation-angle", type=float, default=2.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Geometric Deterministic Optimizer (GDO) Experiment Suite")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Optimizers: {args.optimizers}")
    print(f"Steps: {args.steps}")
    print(f"Device: {args.device}")
    print("=" * 60)

    od = args.output_dir
    opts = tuple(args.optimizers)
    common = dict(optimizers=opts, seed=args.seed, output_dir=od, device=args.device)

    if args.task in EXPERIMENT_REGISTRY:
        fn, cat = EXPERIMENT_REGISTRY[args.task]
        # Pass relevant kwargs based on task
        kwargs = dict(steps=args.steps, **common)
        if args.task in ("rastrigin", "ackley", "styblinski_tang"):
            kwargs["n_dims"] = args.n_dims
        if args.task == "registration":
            kwargs["noise_std"] = args.noise_std
            kwargs["rotation_angle"] = args.rotation_angle
        fn(**kwargs)

    elif args.task == "analytic":
        run_category("analytic", steps=args.steps, n_dims=args.n_dims, **common)
    elif args.task == "geometric":
        run_category("geometric", steps=args.steps, **common)
    elif args.task == "ga_neural":
        run_category("ga_neural", steps=args.steps, **common)
    elif args.task == "manifold":
        run_category("manifold", steps=args.steps, **common)
    elif args.task in ("all", "compare_all"):
        all_results = run_all_experiments(steps=min(args.steps, 1000), **common)
        report = analyze_experiment_results(all_results, output_dir=od)
        print("\n" + report)
