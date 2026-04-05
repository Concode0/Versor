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
sound, it is planned to be spun off into a dedicated, independent repository 
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
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.algebra import CliffordAlgebra
from layers import RotorLayer, MultiRotorLayer, CliffordLinear, CliffordLayerNorm
from functional.activation import GeometricGELU
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
        # Hessian-vector product: grad(g*v) = H*v
        gv = (flat_grad * v.detach()).sum()
        hvp = torch.autograd.grad(gv, params, allow_unused=True)
        return torch.cat([
            h.reshape(-1) if h is not None else torch.zeros(p.numel(), device=p.device)
            for h, p in zip(hvp, params)
        ])

    def _lanczos_eigenvalues(
        self, loss: torch.Tensor, params: List[torch.Tensor], k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate bottom-k eigenvalues and eigenvectors via Lanczos iteration.

        Returns (eigenvalues [k], eigenvectors [n, k]) sorted ascending.
        Eigenvectors are approximate Hessian eigenvectors projected back from
        the Lanczos basis: H_vecs = Q * eigvecs_T where Q is the Lanczos basis.
        """
        n = sum(p.numel() for p in params)
        k = min(k, n)
        device = params[0].device

        # Lanczos: build tridiagonal T such that Q^T H Q = T
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
            # Re-orthogonalize (modified Gram-Schmidt)
            for qv in Q:
                q_curr = q_curr - (q_curr * qv).sum() * qv
            q_curr = q_curr / (q_curr.norm() + 1e-12)
            Q.append(q_curr)

        # Build tridiagonal matrix
        m = len(alpha)
        T = torch.zeros(m, m, device=device)
        for i, a in enumerate(alpha):
            T[i, i] = a
        for i, b in enumerate(beta[1:min(len(beta), m)]):
            T[i, i+1] = b
            T[i+1, i] = b

        # eigh returns ascending eigenvalues and eigenvectors as columns
        eigvals, eigvecs_T = torch.linalg.eigh(T)
        # Map T-eigenvectors back to parameter space: H_vecs = Q_mat^T @ eigvecs_T
        Q_mat = torch.stack(Q[:m])          # [m, n]
        H_vecs = F.normalize(Q_mat.T @ eigvecs_T, dim=0)   # [n, m]
        return eigvals.detach(), H_vecs.detach()

    def check(
        self, loss: torch.Tensor, params: List[torch.Tensor], step: int
    ) -> Optional[CriticalPoint]:
        """Check if current point is a critical point. Returns CriticalPoint or None."""
        self._step = step
        if step % self.detect_every != 0:
            return None

        # Compute gradient norm
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat_g = torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
            for g, p in zip(grads, params)
        ])
        grad_norm = flat_g.norm().item()

        if grad_norm > self.grad_tol * 10:
            return None  # Not near a critical point

        # Near critical point: compute Hessian spectrum and eigenvectors
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
    """Deploys lightweight probes to measure local geometry.

    Probes a small sphere around current parameter point to estimate:
      - Mean curvature: Tr(H) / n
      - Anisotropy: std(eigenvalues) / mean(|eigenvalues|)
      - Plateau score: ratio of near-zero curvature directions
      - Principal curvature directions (for Lorentz warp targeting)

    For f: R^n -> R, the curvature of the graph surface at theta is
    determined by the shape operator S = H / sqrt(1 + ||grad f||^2).
    The principal curvatures are eigenvalues of S.
    """

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
        plateau_score: float            # fraction of directions with |k| < threshold
        min_curvature_dir: torch.Tensor # direction of least curvature (for warp)
        max_curvature_dir: torch.Tensor # direction of most curvature (valley axis)
        grad_norm: float

    def probe(
        self,
        loss_fn: Callable[[], torch.Tensor],
        params: List[torch.Tensor],
    ) -> "CurvatureProbe.ProbeResult":
        """Sample curvature via finite differences on the parameter sphere."""
        n = sum(p.numel() for p in params)
        device = params[0].device

        # Random orthogonal directions on S^{n-1}
        dirs = torch.randn(self.n_directions, n, device=device)
        dirs = F.normalize(dirs, dim=-1)
        # Gram-Schmidt for first few directions
        for i in range(1, min(self.n_directions, 4)):
            for j in range(i):
                dirs[i] -= (dirs[i] * dirs[j]).sum() * dirs[j]
            dirs[i] = F.normalize(dirs[i], dim=0)

        # Save original params
        orig = [p.data.clone() for p in params]

        curvatures = []
        with torch.no_grad():
            loss_0 = loss_fn().item()

            for d in dirs:
                offset = d * self.probe_radius
                # Perturb +direction
                idx = 0
                for p in params:
                    sz = p.numel()
                    p.data += offset[idx:idx+sz].reshape(p.shape)
                    idx += sz
                loss_plus = loss_fn().item()

                # Restore and perturb -direction
                for p, o in zip(params, orig):
                    p.data.copy_(o)

                idx = 0
                for p in params:
                    sz = p.numel()
                    p.data -= offset[idx:idx+sz].reshape(p.shape)
                    idx += sz
                loss_minus = loss_fn().item()

                # Second directional derivative: (f(+) - 2*f(0) + f(-)) / h^2
                k = (loss_plus - 2 * loss_0 + loss_minus) / (self.probe_radius ** 2)
                curvatures.append(k)

                # Restore
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

        # Gradient norm at current point
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
    """Approximates geodesic trajectories in parameter space.

    Uses the Riemannian metric G_ij(theta) = H_ij(theta) + lambda*I
    to define natural distances. Geodesic equation:
        d^2 theta / dt^2 + Gamma^k_ij (d theta^i/dt)(d theta^j/dt) = 0

    For practical optimization, simplified to natural gradient:
        theta_{t+1} = theta_t - alpha * G^{-1}(theta_t) * grad L(theta_t)

    This is natural gradient / Fisher-preconditioned gradient with
    the Hessian as the metric (Amari, 1998).

    When a geodesic to a known lower minimum is available, interpolate
    toward it rather than following local gradient.
    """

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
        """Set a known lower critical point as geodesic target."""
        self._target = target_params.clone()
        self._target_loss = target_loss

    def natural_gradient_step(
        self,
        flat_grad: torch.Tensor,
        hessian_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Precondition gradient by diagonal Hessian (natural gradient approx).

        G^{-1} ~= diag(1 / (|H_ii| + lambda))
        """
        metric_inv = 1.0 / (hessian_diag.abs() + self.lambda_reg)
        return metric_inv * flat_grad

    def geodesic_blend(
        self,
        flat_params: torch.Tensor,
        natural_step: torch.Tensor,
        lr: float,
    ) -> torch.Tensor:
        """Blend natural gradient step with direction toward known target.

        If we know a lower minimum, partially steer toward it along the
        chord (approximation of geodesic for large steps).
        """
        step = -lr * natural_step

        if self._target is not None:
            chord = self._target - flat_params
            chord_norm = chord.norm()
            if chord_norm > 1e-8:
                chord_dir = chord / chord_norm
                # Project step onto chord direction and complement
                step_along = (step * chord_dir).sum() * chord_dir
                step_perp = step - step_along
                # Bias toward target
                step = step_perp + step_along + self.geodesic_weight * lr * chord_dir

        return step


class LorentzWarpOptimizer:
    """Applies Lorentz-boost-inspired metric warping for plateau escape.

    On a plateau (||grad L|| ~= 0, curvature ~= 0), standard optimizers stall.
    Inspiration from special relativity: a Lorentz boost along direction u_hat
    contracts lengths by gamma = 1/sqrt(1 - beta^2). We apply this to the
    effective learning rate metric.

    Concretely:
    - Detect plateau: grad_norm < eps AND plateau_score > threshold
    - Compute boost direction: min-curvature direction from probe
    - Apply boosted LR: lr_eff_i = lr * (1 + (gamma-1) * |u_hat_i|^2)
      This gives larger steps along the plateau direction, smaller in sharp valleys.
    - beta increases as we stay on the plateau (escalating boost)
    - beta decays when we escape (grad_norm recovers)
    """

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
        """Update plateau detection state. Returns True if on plateau.

        Must be called only on fresh probe steps (not every optimizer step).
        """
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
        """Compute per-parameter effective learning rate with Lorentz warp."""
        lr_vec = torch.ones(flat_grad_shape, device=device) * lr

        if self._on_plateau and self._boost_dir is not None and self._beta > 0.01:
            g = self.gamma
            d = self._boost_dir
            if d.shape[0] == flat_grad_shape:
                # Lorentz length contraction: larger step along the plateau direction
                boost_factor = 1.0 + (g - 1.0) * d ** 2
                lr_vec = lr_vec * boost_factor

        return lr_vec


class GeometricParameterController:
    """Geometrically-verified partial parameter updates via commutator coloring.

    Combines four geometric signals to control parameter group updates:

    1. **FIM** (Fisher Information Matrix) -- diagonal approximation from
       gradient outer products.  High FIM = sensitive → smaller steps.
    2. **Lie Bracket Closure Error** -- from CoreCommutatorAnalyzer.
       Low error → bivectors form a Lie subalgebra → trust larger updates.
    3. **Bivector Coherence** -- from GeodesicFlow.  High coherence →
       parameters are structurally aligned → update confidently.
    4. **Grade Entropy** -- grade energy viewed as a probability distribution.
       High entropy → still exploring grade structure → allow larger steps.

    Pipeline:
      a) Build hybrid commutativity scores (FD cross-Hessian + algebraic)
      b) Greedy graph coloring → parallel update schedule
      c) Per-group scale = f(FIM, closure, coherence, entropy)
    """

    def __init__(
        self,
        algebra: Optional[CliffordAlgebra] = None,
        commutator_threshold: float = 0.3,
        fim_damping: float = 1e-4,
        closure_trust_threshold: float = 0.1,
        coherence_gate: float = 0.3,
        entropy_exploration_threshold: float = 0.7,
        fd_step: float = 1e-3,
    ):
        self.algebra = algebra
        self.commutator_threshold = commutator_threshold
        self.fim_damping = fim_damping
        self.closure_trust_threshold = closure_trust_threshold
        self.coherence_gate = coherence_gate
        self.entropy_exploration_threshold = entropy_exploration_threshold
        self.fd_step = fd_step

        # Core analyzers (only when algebra is available)
        self.core_comm: Optional[CoreCommutatorAnalyzer] = None
        self.spectral: Optional[SpectralAnalyzer] = None
        self.geodesic: Optional[GeodesicFlow] = None
        if algebra is not None and algebra.n >= 2:
            self.core_comm = CoreCommutatorAnalyzer(algebra)
            self.spectral = SpectralAnalyzer(algebra)
            self.geodesic = GeodesicFlow(algebra, k=8)

    # ------------------------------------------------------------------
    # FIM diagonal (loss-based, works for any model)
    # ------------------------------------------------------------------

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
        """Per-group diagonal Fisher Information approximation.

        FIM_diag ≈ E[g^2] where g = grad(loss).  High FIM = sensitive param.
        """
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

    # ------------------------------------------------------------------
    # Geometric scores (algebra-based, requires RotorLayer params)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mv_params(model: nn.Module) -> Optional[torch.Tensor]:
        """Extract learned multivector parameters from RotorLayer / MultiRotorLayer."""
        mv_list: List[torch.Tensor] = []
        for m in model.modules():
            if isinstance(m, RotorLayer):
                bv = m.grade_weights.detach()  # [channels, num_grade_elems]
                # Embed into full multivector space
                full = torch.zeros(
                    bv.shape[0], m.algebra.dim,
                    device=bv.device, dtype=bv.dtype,
                )
                full[:, m.grade_indices] = bv
                mv_list.append(full)
            elif isinstance(m, MultiRotorLayer):
                bv = m.rotor_grade_weights.detach()  # [num_rotors, num_grade_elems]
                full = torch.zeros(
                    bv.shape[0], m.algebra.dim,
                    device=bv.device, dtype=bv.dtype,
                )
                full[:, m.grade_indices] = bv
                mv_list.append(full)
        if not mv_list:
            return None
        return torch.cat(mv_list, dim=0)  # [K, dim]

    def compute_geometric_scores(self, model: nn.Module) -> Dict:
        """Compute Lie closure, coherence, and grade entropy from learned bivectors."""
        if self.core_comm is None:
            return {}

        mv_params = self._extract_mv_params(model)
        if mv_params is None or mv_params.shape[0] < 2:
            return {}

        result: Dict = {}

        # 1. Commutator analysis → commutativity matrix + Lie bracket closure
        try:
            comm_result = self.core_comm.analyze(mv_params)
            result["comm_result"] = comm_result
            result["closure_error"] = comm_result.lie_bracket_structure.get(
                "closure_error", 1.0
            )
            result["mean_commutator_norm"] = comm_result.mean_commutator_norm
        except Exception:
            pass

        # 2. Bivector coherence via GeodesicFlow
        if self.geodesic is not None and mv_params.shape[0] >= 3:
            try:
                k_actual = min(self.geodesic.k, mv_params.shape[0] - 1)
                gf = GeodesicFlow(self.algebra, k=k_actual)
                result["coherence"] = gf.coherence(mv_params)
                result["per_point_coherence"] = gf.per_point_coherence(mv_params)
            except Exception:
                pass

        # 3. Grade entropy via SpectralAnalyzer
        if self.spectral is not None:
            try:
                grade_energy = self.spectral.grade_energy_spectrum(
                    mv_params.unsqueeze(1)
                )
                result["grade_energy"] = grade_energy
                # Normalize to probability distribution → entropy
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

    # ------------------------------------------------------------------
    # Hybrid scoring: FD cross-Hessian + algebraic commutativity
    # ------------------------------------------------------------------

    def _fd_cross_hessian(
        self,
        loss_fn: Callable[[], torch.Tensor],
        param_groups: List[List[nn.Parameter]],
    ) -> Dict[Tuple[int, int], float]:
        """FD cross-Hessian scores (original algorithm from CommutatorAnalyzer)."""
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
        """Blend FD cross-Hessian with algebraic commutativity matrix."""
        fd_scores = self._fd_cross_hessian(loss_fn, param_groups)

        if "comm_result" not in geometric_scores:
            return fd_scores

        alg_matrix = geometric_scores["comm_result"].commutativity_matrix
        n = len(param_groups)
        alg_n = alg_matrix.shape[0]

        for (i, j), fd_score in fd_scores.items():
            # Map group indices to algebra dimensions (clamped)
            ai, aj = min(i, alg_n - 1), min(j, alg_n - 1)
            alg_score = alg_matrix[ai, aj].item()
            # Normalize algebraic score to [0, 1] range
            alg_max = alg_matrix.max().item()
            if alg_max > 1e-8:
                alg_score /= alg_max
            fd_scores[(i, j)] = 0.6 * fd_score + 0.4 * alg_score

        return fd_scores

    # ------------------------------------------------------------------
    # Greedy graph coloring
    # ------------------------------------------------------------------

    def parallel_groups(
        self, scores: Dict[Tuple[int, int], float], n_groups: int
    ) -> List[List[int]]:
        """Greedy graph coloring on hybrid commutativity scores."""
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

    # ------------------------------------------------------------------
    # Per-group update scale from 4 geometric signals
    # ------------------------------------------------------------------

    def compute_group_scales(
        self,
        param_groups: List[List[nn.Parameter]],
        fim_diag: Dict[int, torch.Tensor],
        geometric_scores: Dict,
    ) -> List[float]:
        """Per-group update scale = f(FIM, closure, coherence, entropy)."""
        scales = []
        for g_idx in range(len(param_groups)):
            # 1. FIM sensitivity: high FIM → small scale
            fim_g = fim_diag.get(g_idx)
            if fim_g is not None and fim_g.numel() > 0:
                fim_sensitivity = fim_g.mean().item()
                fim_scale = 1.0 / (1.0 + fim_sensitivity / self.fim_damping)
            else:
                fim_scale = 1.0

            # 2. Lie bracket closure: low error → trust larger steps
            closure_err = geometric_scores.get("closure_error", 0.5)
            if closure_err < self.closure_trust_threshold:
                closure_scale = 1.5
            elif closure_err > 0.5:
                closure_scale = 0.5
            else:
                closure_scale = 1.0

            # 3. Coherence gate: low coherence → reduce update
            coherence = geometric_scores.get("coherence", 0.5)
            coherence_scale = max(0.3, min(1.0, coherence / self.coherence_gate))

            # 4. Grade entropy: high → exploring → allow larger steps
            entropy = geometric_scores.get("grade_entropy", 0.5)
            if entropy > self.entropy_exploration_threshold:
                entropy_scale = 1.2
            else:
                entropy_scale = 0.8

            scale = fim_scale * closure_scale * coherence_scale * entropy_scale
            scale = max(0.1, min(2.0, scale))
            scales.append(scale)

        return scales

    # ------------------------------------------------------------------
    # Top-level: analyze and produce schedule + scales
    # ------------------------------------------------------------------

    def analyze_and_schedule(
        self,
        model: nn.Module,
        loss_fn: Callable[[], torch.Tensor],
        param_groups: List[List[nn.Parameter]],
    ) -> Tuple[List[List[int]], List[float], Dict]:
        """Full pipeline: FIM → geometric scores → hybrid coloring → scales.

        Returns (schedule, scales, diagnostics).
        """
        # 1. FIM diagonal
        fim_diag = self.compute_fim_diagonal(loss_fn, model, param_groups)

        # 2. Geometric scores (if algebra available)
        geo_scores = self.compute_geometric_scores(model)

        # 3. Hybrid commutator scores → greedy coloring
        hybrid_scores = self.build_hybrid_scores(loss_fn, param_groups, geo_scores)
        schedule = self.parallel_groups(hybrid_scores, len(param_groups))

        # 4. Per-group update scales
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
    """Escape local minima via lift -> oracle search -> pull-down.

    Flow when stuck:

      1. DETECT
         Monitor loss improvement. If no progress for `patience` steps, trigger.

      2. LIFT (adaptive sigma + biased candidates)
         Generate k candidate starting points. Two strategies based on geometry:

         a) SADDLE DETECTION: if probe found negative-curvature directions,
            push along them -- those are true downhill escape directions.
            Also push along the anti-gradient of the loss at slightly perturbed
            positions: grad(theta + eps*d) points back toward current basin;
            -grad points away from it.

         b) TRUE LOCAL MINIMUM (all curvatures positive): random orthonormal
            directions, but with ADAPTIVE SIGMA that doubles on each fail.
            Eventually sigma is large enough to cross the basin boundary ridge.

         Candidates:
             psi_0 = theta + sigma * d_probe   (min-curvature direction from probe)
             psi_1 = theta - sigma * d_probe   (opposite sign)
             psi_2..k = theta + sigma * d_rand  (random orthonormal)
         Anti-gradient biasing: for each psi_j, compute g_j = grad L(psi_j), then
         also try theta + sigma * (-g_j / ||g_j||) as an additional candidate.

      3. ORACLE SEARCH
         For each candidate psi_j, run `oracle_steps` of Adam independently.
         Fresh optimizer state per candidate (no momentum bias from stuck region).

      4. PULL-DOWN
         Evaluate L(psi_j*) for all optimized candidates.
         Return theta_new = argmin_j L(psi_j*).
         If theta_new improves on current loss, accept. Otherwise try soft blend:
             theta_blend = theta + alpha * (theta_new - theta)   alpha in (0, 1]

      5. WARP TARGET
         The pull-down result becomes a geodesic target for the main warp,
         steering subsequent normal steps toward the oracle's finding.
    """

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
        """Extract flat gradient from model params after backward."""
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
        """Compute bottom-k Hessian eigenvectors via lightweight Lanczos.

        Returns (eigenvalues [k], eigenvectors [n, k]) sorted ascending.
        Uses a fresh loss computation isolated from the main training graph.
        """
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
        Q_mat = torch.stack(Q[:m])                          # [m, n]
        H_vecs = F.normalize(Q_mat.T @ eigvecs_T, dim=0)   # [n, m]
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
        """Execute lift -> oracle search -> pull-down with adaptive sigma.

        Returns (best_flat_params, best_loss), or (None, current_loss) if
        no improvement found. On failure, sigma doubles for the next call.

        Direction priority for candidate generation:
          1. Negative-eigenvalue eigenvectors (saddle escape -- true downhill)
          2. Smallest-eigenvalue eigenvectors (lowest Hessian resistance)
          3. Probe min-curvature direction (FD-based, both signs)
          4. Anti-gradient from perturbed positions (basin boundary push)
          5. Random orthonormal (fill remaining slots)
        If hessian_vecs/vals are not provided, a lightweight Lanczos is run on
        the fly to compute them.
        """
        self._lift_count += 1
        self._steps_no_improve = 0

        flat_orig = self._get_flat(model)
        n = flat_orig.shape[0]
        device = flat_orig.device

        # Adaptive sigma: each consecutive fail doubles it, capped at max_sigma
        sigma = self._current_sigma
        # Oracle lr scales with sigma: bigger jumps need bigger search steps
        scaled_lr = self.oracle_lr * max(1.0, sigma / self.lift_sigma)

        print(f"  [LiftOracle] #{self._lift_count}: sigma={sigma:.4f}  "
              f"oracle_lr={scaled_lr:.5f}  loss={current_loss:.5f}")

        # --- Build candidate starting directions ---
        directions: List[torch.Tensor] = []

        # If no eigenvectors provided, compute a lightweight Lanczos on the fly.
        if hessian_vecs is None:
            try:
                hessian_vals, hessian_vecs = self._compute_bottom_eigvecs(
                    loss_fn, model, k=min(4, n)
                )
            except Exception:
                hessian_vecs = None
                hessian_vals = None

        # 1. Eigenvector-directed kicks (highest priority).
        #    Negative-eigenvalue eigenvectors are exact saddle escape directions.
        #    Smallest-eigenvalue eigenvectors are lowest-resistance directions.
        if hessian_vecs is not None and hessian_vals is not None:
            ev = hessian_vecs.to(device)       # [n, m], columns sorted ascending
            vals = hessian_vals.to(device)     # [m]
            # Negative eigenvalue directions: both +/- (saddle escape)
            neg_idx = (vals < 0).nonzero(as_tuple=False).view(-1)
            for i in neg_idx[:2].tolist():
                v = F.normalize(ev[:, i], dim=0)
                directions.append(v)
                if len(directions) < self.k:
                    directions.append(-v)
            # Smallest-eigenvalue direction (even if positive -- least resistance)
            if len(directions) < self.k and ev.shape[1] > 0:
                directions.append(F.normalize(ev[:, 0], dim=0))

        # 2. Probe-guided: min-curvature dir (both signs)
        if probe_result is not None and probe_result.min_curvature_dir.shape[0] == n:
            d_probe = F.normalize(probe_result.min_curvature_dir.to(device), dim=0)
            if len(directions) < self.k:
                directions.append(d_probe)
            if len(directions) < self.k:
                directions.append(-d_probe)

        # 3. Anti-gradient from perturbed positions.
        #    The gradient at theta + sigma*d points back toward the current basin;
        #    its negation points away. Using 2 random probe directions for this.
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

        # 3. Fill remaining slots with random orthonormal directions
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

        # Candidate starting points
        candidates = [flat_orig + sigma * d for d in directions[:self.k]]

        # --- Oracle search: independent Adam per candidate ---
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

        # --- Pull-down ---
        self._set_flat(model, flat_orig)

        if best_params is not None:
            if self.accept_blend < 1.0:
                best_params = flat_orig + self.accept_blend * (best_params - flat_orig)
            improvement = current_loss - best_loss
            print(f"  [LiftOracle] [ok] improvement={improvement:.5f} -> {best_loss:.5f}")
            # Reset sigma on success
            self._consecutive_fails = 0
            self._current_sigma = self.lift_sigma
            return best_params, best_loss

        # No improvement: escalate sigma for next attempt
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
    # Extended fields for richer visualization
    causal_report: Optional[Dict] = None
    lifting_report: Optional[Dict] = None
    landscape_losses: Optional[torch.Tensor] = None
    landscape_positions: Optional[torch.Tensor] = None
    flow_bivectors: Optional[torch.Tensor] = None
    per_point_coherence: Optional[torch.Tensor] = None


class PreExplorationAnalyzer:
    """Pre-optimization landscape analysis pipeline.

    Samples the loss landscape around current parameters, analyzes
    dimensionality and geometric structure, then recommends GDO
    configuration parameters.

    Pipeline:
      1. Perturb params in random directions, evaluate loss
      2. Subsample via StatisticalSampler
      3. Dimension analysis (PCA eigenvalues, participation ratio)
      4. GA analysis on model's learned bivectors (if algebra available)
      5. Geodesic flow on landscape (coherence + curvature)
      6. Recommend GDOConfig from analysis signals
    """

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
        """Perturb params in random directions, evaluate loss at each."""
        theta0 = self._get_flat(model).clone()
        n_params = theta0.shape[0]
        device = theta0.device

        positions = [theta0]
        losses = []

        # Evaluate at origin
        with torch.no_grad():
            losses.append(loss_fn().item())

        # Random perturbations
        for _ in range(self.n_samples - 1):
            direction = torch.randn(n_params, device=device)
            direction = F.normalize(direction, dim=0)
            perturbed = theta0 + self.sample_radius * direction
            self._set_flat(model, perturbed)
            with torch.no_grad():
                losses.append(loss_fn().item())
            positions.append(perturbed.clone())

        # Restore original
        self._set_flat(model, theta0)

        return torch.stack(positions), torch.tensor(losses, device=device)

    def analyze(
        self, model: nn.Module, loss_fn: Callable
    ) -> PreExplorationResult:
        """Run the full pre-exploration pipeline."""
        result = PreExplorationResult()

        # 1. Sample landscape
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

        # 2. Subsample via StatisticalSampler
        config = SamplingConfig(strategy="random", max_samples=min(200, len(positions)))
        sampled, _ = StatisticalSampler.sample(positions, config)

        # 3. Dimension analysis
        eda = None
        try:
            eda = EffectiveDimensionAnalyzer(device=self.device)
            dim_result = eda.analyze(sampled)
            result.dim_result = dim_result
        except Exception:
            dim_result = None

        # 4. GA analysis on model's learned bivectors
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

                # Geometric scores for config recommendation
                gpc = GeometricParameterController(algebra=self.algebra)
                result.geometric_scores = gpc.compute_geometric_scores(model)

                # Flow bivectors and per-point coherence on learned params
                try:
                    k_flow = min(8, mv_params.shape[0] - 1)
                    if k_flow >= 2:
                        gf_params = GeodesicFlow(self.algebra, k=k_flow)
                        result.flow_bivectors = gf_params.flow_bivectors(mv_params)
                        result.per_point_coherence = gf_params.per_point_coherence(mv_params)
                except Exception:
                    pass

        # 5. Geodesic flow on landscape
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
                # Causal report on landscape
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

        # 5b. Dimension lifting test (when algebra available)
        if self.algebra is not None and dim_result is not None:
            try:
                p, q = self.algebra.p, self.algebra.q
                n = p + q
                # Use PCA-reduced landscape data for lifting test
                lift_dim = min(n, dim_result.intrinsic_dim) if eda else n
                if lift_dim >= 2 and eda is not None:
                    reduced_lift = eda.reduce(sampled, lift_dim)
                    lifter = DimensionLifter(device=self.device)
                    result.lifting_report = lifter.test(
                        reduced_lift, p=lift_dim, q=0, k=min(8, reduced_lift.shape[0] - 1))
            except Exception:
                pass

        # 6. Recommend config
        result.recommended_config = self._recommend_config(result)
        result.strategy_label = self._classify_strategy(result)

        return result

    def _recommend_config(self, result: PreExplorationResult) -> GDOConfig:
        """Map analysis signals → GDOConfig hyperparameters."""
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

            # Condition number from eigenvalues
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

        # Geometric controller thresholds from pre-analysis
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


class GeometricDeterministicOptimizer:
    """Full Morse-Geometric optimization pipeline (GDO).

    Combines all phases:
      A. Topology search: detect critical points during training
      B. Curvature probes: map local geometry
      C. Geodesic integration: natural gradient + target steering
      D. Lorentz warp: plateau escape via metric contraction
      E. Commutator schedule: parallel/sequential update groups
      F. Dimensional lift oracle: escape local minima via higher-dim search

    The optimizer operates in 3 modes:
      EXPLORE: running probes, building topology map
      NAVIGATE: following geodesic to detected lower minimum
      SPRINT: commutator-grouped fast descent, minimal probing
    """

    class Mode(Enum):
        EXPLORE = "explore"
        NAVIGATE = "navigate"
        SPRINT = "sprint"

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
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
        # Config overrides individual kwargs when provided
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
        )
        self.lift_oracle = DimensionalLiftOracle(
            patience=lift_patience,
            oracle_lr=lr,
        )

        # State
        self.landscape = LandscapeMap()
        self.mode = GeometricDeterministicOptimizer.Mode.EXPLORE
        self.step = 0
        self.probe_interval = probe_interval
        self.sprint_after = sprint_after
        self.max_navigate_steps = max_navigate_steps
        self._probe_result: Optional[CurvatureProbe.ProbeResult] = None
        self._commutator_schedule: Optional[List[List[int]]] = None
        self._group_scales: Optional[List[float]] = None
        self._controller_diagnostics: Optional[Dict] = None

        # NAVIGATE phase tracking
        self._navigate_steps: int = 0
        self._navigate_best_loss: float = float('inf')
        self._navigate_no_improve: int = 0

        # Auto-group parameters by top-level named child modules
        self._param_groups: List[List[nn.Parameter]] = self._build_param_groups()
        # Flat index ranges for each group (for per-group Adam in SPRINT)
        self._group_ranges: List[List[Tuple[int, int]]] = self._compute_group_ranges()

        # Cached Hessian eigenvectors from last topology check (passed to lift oracle)
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
        """Group parameters by top-level named child modules."""
        groups = []
        for _name, module in self.model.named_children():
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                groups.append(params)
        if not groups:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if params:
                groups.append(params)
        return groups

    def _compute_group_ranges(self) -> List[List[Tuple[int, int]]]:
        """Map each param group to list of (start, end) offsets in the flat vector."""
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
        """Global Adam + Lorentz warp step (EXPLORE and NAVIGATE).

        Adam provides the adaptive preconditioned direction.
        Warp scales the per-element LR based on curvature geometry.
        Step = lr_vec(warp) * adam_dir(m_hat / sqrt(v_hat)).
        """
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

        # Warp LR uses state set on probe steps only -- not updated here
        lr_vec = self.warp.warped_lr(self.lr, flat_grad.shape[0], device)

        self._set_flat_params(self._get_flat_params() - lr_vec * adam_dir)

    def _group_adam_warp_step(self, group_idx: int, group_grad: torch.Tensor):
        """Per-group Adam + warp step for SPRINT.

        Each group has its own Adam moment state so updates are independent.
        group_grad is the flat gradient slice for this group only.
        """
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

        # Use warp-scaled LR * geometric controller scale
        lr_vec = self.warp.warped_lr(self.lr, group_grad.shape[0], device)
        if self._group_scales is not None and group_idx < len(self._group_scales):
            lr_vec = lr_vec * self._group_scales[group_idx]

        # Write step back into the correct flat-param positions
        flat_p = self._get_flat_params()
        ptr = 0
        for start, end in self._group_ranges[group_idx]:
            sz = end - start
            flat_p[start:end] -= lr_vec[ptr:ptr + sz] * adam_dir[ptr:ptr + sz]
            ptr += sz
        self._set_flat_params(flat_p)

    def optimize_step(self, loss: torch.Tensor) -> Dict:
        """Execute one step of Morse-geometric optimization."""
        current_loss = loss.item()
        info = {"step": self.step, "mode": self.mode.value, "loss": current_loss}
        params = list(self.model.parameters())

        # ---- EXPLORE ----
        if self.mode == GeometricDeterministicOptimizer.Mode.EXPLORE:

            # Probe: runs before backward, modifies and restores param data.
            # warp.update() is called only here, on fresh probe data.
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

            # LiftOracle: escape when stuck
            if self.lift_oracle.should_lift(current_loss, self.step):
                new_flat, new_loss = self.lift_oracle.lift_and_search(
                    self.model, self.loss_fn, current_loss,
                    probe_result=self._probe_result,
                    hessian_vecs=self._hessian_vecs,
                    hessian_vals=self._hessian_vals,
                )
                if new_flat is not None:
                    self._set_flat_params(new_flat)
                    # Reset Adam state so momentum from stuck region does not carry over
                    self._adam_m = None
                    self._adam_v = None
                    self._adam_t = 0
                    self.step += 1
                    info["lift_oracle"] = f"improved to {new_loss:.5f}"
                    return info

            # Topology check -- use a FRESH loss, not the outer `loss` tensor.
            # _hessian_vector_product inside check() calls autograd.grad with
            # create_graph=True then a second grad without retain_graph, which
            # frees saved tensors. A separate forward pass isolates graph
            # consumption so the outer loss.backward() below remains valid.
            cp = self.topology.check(self.loss_fn(), params, self.step)
            # Cache eigenvectors whenever Lanczos ran (even if no critical point)
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
                        self.mode = GeometricDeterministicOptimizer.Mode.NAVIGATE
                        print(f"  [Morse] -> NAVIGATE toward {target}")

            if self.step >= self.sprint_after:
                self.mode = GeometricDeterministicOptimizer.Mode.SPRINT
                print(f"  [Morse] Step {self.step}: -> SPRINT")

            # Adam direction + warp LR (composed, not separate)
            loss.backward()
            flat_g = self._get_flat_grad()
            self._adam_warp_step(flat_g)
            self.model.zero_grad()

        # ---- NAVIGATE ----
        elif self.mode == GeometricDeterministicOptimizer.Mode.NAVIGATE:
            loss.backward()
            flat_g = self._get_flat_grad()
            hess_diag = flat_g.abs() + 1e-6
            nat_step = self.geodesic.natural_gradient_step(flat_g, hess_diag)
            flat_p = self._get_flat_params()
            delta = self.geodesic.geodesic_blend(flat_p, nat_step, self.lr)
            self._set_flat_params(flat_p + delta)
            self.model.zero_grad()

            self._navigate_steps += 1

            # Exit by loss-no-improve OR timeout
            if current_loss < self._navigate_best_loss - 1e-4:
                self._navigate_best_loss = current_loss
                self._navigate_no_improve = 0
            else:
                self._navigate_no_improve += 1

            stuck = self._navigate_no_improve >= 30
            timed_out = self._navigate_steps >= self.max_navigate_steps
            if stuck or timed_out or self.step >= self.sprint_after:
                reason = "stuck" if stuck else ("timeout" if timed_out else "sprint")
                next_mode = (GeometricDeterministicOptimizer.Mode.SPRINT
                             if self.step >= self.sprint_after
                             else GeometricDeterministicOptimizer.Mode.EXPLORE)
                print(f"  [Morse] NAVIGATE exit ({reason}) -> {next_mode.value}")
                self.mode = next_mode
                self.geodesic._target = None

        # ---- SPRINT ----
        elif self.mode == GeometricDeterministicOptimizer.Mode.SPRINT:
            # Build geometric controller schedule once
            if self._commutator_schedule is None:
                if len(self._param_groups) > 1:
                    print(f"  [GPC] Analyzing parameter geometry...")
                    schedule, scales, diagnostics = (
                        self.controller.analyze_and_schedule(
                            self.model, self.loss_fn, self._param_groups
                        )
                    )
                    self._commutator_schedule = schedule
                    self._group_scales = scales
                    self._controller_diagnostics = diagnostics
                    # Log hybrid scores
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
                    self._commutator_schedule = [[0]] if self._param_groups else [[]]
                    self._group_scales = [1.0]

            # Sequential between colors (high commutator), parallel within color.
            # Each color round uses per-group Adam + warp.
            for color in self._commutator_schedule:
                self.model.zero_grad()
                loss_c = self.loss_fn()
                loss_c.backward()
                full_grad = self._get_flat_grad()

                for group_idx in color:
                    if group_idx >= len(self._param_groups):
                        continue
                    ranges = self._group_ranges[group_idx]
                    group_grad = torch.cat([full_grad[s:e] for s, e in ranges])
                    self._group_adam_warp_step(group_idx, group_grad)

            self.model.zero_grad()

        self.step += 1
        return info


class RosenbrockModel(nn.Module):
    """2D Rosenbrock function as a learnable parameter pair.

    f(x,y) = (a - x)^2 + b*(y - x^2)^2,  minimum at (a, a^2).
    Standard: a=1, b=100. Famous for narrow curved valley.
    """
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = a
        self.b = b
        self.x = nn.Parameter(torch.tensor([-1.5]))
        self.y = nn.Parameter(torch.tensor([1.5]))

    def forward(self) -> torch.Tensor:
        return (self.a - self.x) ** 2 + self.b * (self.y - self.x ** 2) ** 2


class RastriginModel(nn.Module):
    """N-dimensional Rastrigin function.

    f(x) = A*n + sum_i [x_i^2 - A*cos(2*pi*x_i)]
    Many local minima; global minimum at x=0 with f=0.
    """
    def __init__(self, n_dims: int = 4, A: float = 10.0):
        super().__init__()
        self.A = A
        self.n = n_dims
        self.x = nn.Parameter(torch.randn(n_dims) * 3.0)

    def forward(self) -> torch.Tensor:
        x = self.x
        return self.A * self.n + (x ** 2 - self.A * torch.cos(2 * math.pi * x)).sum()


class SmallGBNModel(nn.Module):
    """Small Geometric Blade Network for testing optimizer on actual GA model."""

    def __init__(self, p: int = 3, q: int = 0, channels: int = 4, device: str = 'cpu'):
        super().__init__()
        self.algebra = CliffordAlgebra(p, q, device=device)
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


class RotorRegistrationModel(nn.Module):
    """Fit a rotor in Cl(3,0) to align a source point cloud to a rotated+noised target.

    This is a natural geometric problem: the solution lives on Spin(3), has
    180-degree ambiguity (R and -R give the same rotation), and the learned
    bivectors should close under Lie bracket (so(3) is a Lie algebra).
    """

    def __init__(
        self,
        n_points: int = 50,
        noise_std: float = 0.05,
        rotation_angle: float = 2.5,
        device: str = 'cpu',
    ):
        super().__init__()
        self.algebra = CliffordAlgebra(3, 0, device=device)
        dim = self.algebra.dim  # 8

        # Source: random points on unit sphere
        torch.manual_seed(42)
        raw = torch.randn(n_points, 3, device=device)
        raw = F.normalize(raw, dim=-1)
        self.register_buffer('source', raw)

        # Ground-truth rotation bivector
        axis = torch.tensor([1.0, 1.0, 1.0], device=device)
        axis = axis / axis.norm()
        gt_bv = self._axis_angle_to_bivector(axis, rotation_angle)
        self.register_buffer('gt_bivector', gt_bv)

        # Apply ground-truth rotor to source → target
        gt_rotor = self.algebra.exp(-0.5 * gt_bv.unsqueeze(0))  # [1, 8]
        source_mv = self.algebra.embed_vector(raw)  # [N, 8]
        rotated = self.algebra.sandwich_product(
            gt_rotor.expand(n_points, -1),
            source_mv.unsqueeze(1),
        ).squeeze(1)  # [N, 8]
        target_pts = self._extract_vector(rotated)
        target_pts = target_pts + noise_std * torch.randn_like(target_pts)
        self.register_buffer('target', target_pts)

        # Learnable rotor
        self.rotor = RotorLayer(self.algebra, channels=1)

    def _axis_angle_to_bivector(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        """Convert axis-angle to bivector in Cl(3,0).

        Basis: e1=1, e2=2, e3=4, e12=3, e13=5, e23=6, e123=7
        Rotation bivector B = angle * (a3*e12 - a2*e13 + a1*e23)
        """
        bv = torch.zeros(self.algebra.dim, device=axis.device)
        bv[3] = angle * axis[2]    # e12
        bv[5] = -angle * axis[1]   # e13
        bv[6] = angle * axis[0]    # e23
        return bv

    def _extract_vector(self, mv: torch.Tensor) -> torch.Tensor:
        """Extract grade-1 (x,y,z) from multivector. e1=1, e2=2, e3=4."""
        return torch.stack([mv[..., 1], mv[..., 2], mv[..., 4]], dim=-1)

    def forward(self) -> torch.Tensor:
        source_mv = self.algebra.embed_vector(self.source)  # [N, 8]
        source_mv = source_mv.unsqueeze(1)  # [N, 1, 8]
        rotated_mv = self.rotor(source_mv)  # [N, 1, 8]
        pred_pts = self._extract_vector(rotated_mv.squeeze(1))
        return F.mse_loss(pred_pts, self.target)

    def angular_error(self) -> float:
        """Angle error (radians) between learned and ground-truth rotors."""
        with torch.no_grad():
            learned_bv = torch.zeros(
                self.algebra.dim, device=self.gt_bivector.device
            )
            learned_bv[self.rotor.grade_indices] = self.rotor.grade_weights[0]
            # Angle ≈ 2 * ||B_learned - B_gt|| (for small errors)
            # More robust: cos(angle/2) = <R_learned, R_gt>_0
            r_learned = self.algebra.exp(-0.5 * learned_bv.unsqueeze(0))
            r_gt = self.algebra.exp(-0.5 * self.gt_bivector.unsqueeze(0))
            # Inner product of rotors = grade-0 of R_learned * ~R_gt
            r_gt_rev = self.algebra.reverse(r_gt)
            product = self.algebra.geometric_product(r_learned, r_gt_rev)
            cos_half = product[0, 0].abs().clamp(max=1.0).item()
            return 2.0 * math.acos(cos_half)


# ======================================================================
# Visualization
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

    # (0,0) Eigenvalue spectrum
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

    # (0,1) Local dimension histogram / loss distribution
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

    # (0,2) Grade energy bar chart
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

    # (1,0) Coherence / curvature
    ax = axes[1, 0]
    metrics = ["Coherence", "Curvature"]
    values = [pre_result.landscape_coherence, pre_result.landscape_curvature]
    bar_colors = []
    for v, name in zip(values, metrics):
        if name == "Coherence":
            bar_colors.append('green' if v > 0.5 else ('orange' if v > 0.3 else 'red'))
        else:
            bar_colors.append('green' if v < 0.3 else ('orange' if v < 0.5 else 'red'))
    bars = ax.barh(metrics, values, color=bar_colors, alpha=0.7, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_title(f"Landscape Geometry\nStrategy: {pre_result.strategy_label}")
    ax.grid(True, alpha=0.3)

    # (1,1) Lie bracket closure + grade entropy + coherence
    ax = axes[1, 1]
    labels, vals, colors_list = [], [], []
    if "closure_error" in gs:
        labels.append("Lie Closure\nError")
        vals.append(gs["closure_error"])
        ce = gs["closure_error"]
        colors_list.append('green' if ce < 0.1 else ('orange' if ce < 0.5 else 'red'))
    if "grade_entropy" in gs:
        labels.append("Grade\nEntropy")
        vals.append(gs["grade_entropy"])
        colors_list.append('purple')
    if "coherence" in gs:
        labels.append("Bivector\nCoherence")
        vals.append(gs["coherence"])
        colors_list.append('steelblue')
    if labels:
        ax.barh(labels, vals, color=colors_list, alpha=0.7, edgecolor='white')
        ax.set_xlim(0, max(1.0, max(vals) * 1.1))
        ax.set_title("Geometric Signals")
    else:
        ax.text(0.5, 0.5, "No geometric\nanalysis", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # (1,2) Recommended config
    ax = axes[1, 2]
    ax.axis('off')
    cfg = pre_result.recommended_config
    config_text = (
        f"lr: {cfg.lr}\n"
        f"probe_interval: {cfg.probe_interval}\n"
        f"topology_interval: {cfg.topology_interval}\n"
        f"sprint_after: {cfg.sprint_after}\n"
        f"lift_patience: {cfg.lift_patience}\n"
        f"lift_sigma: {cfg.lift_sigma}\n"
        f"lorentz_max_beta: {cfg.lorentz_max_beta}\n"
        f"commutator_threshold: {cfg.commutator_threshold}\n"
        f"fim_damping: {cfg.fim_damping}\n"
        f"closure_trust: {cfg.closure_trust_threshold}\n"
        f"coherence_gate: {cfg.coherence_gate}\n"
        f"entropy_thresh: {cfg.entropy_exploration_threshold}"
    )
    ax.text(0.1, 0.95, "Recommended Config", fontsize=11, fontweight='bold',
            va='top', transform=ax.transAxes)
    ax.text(0.1, 0.85, config_text, fontsize=9, va='top', family='monospace',
            transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(output_dir, "pre_exploration.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_optimization_trajectory(
    history: Dict,
    title: str = "Optimization Trajectory",
    output_dir: str = "gdo_plots",
):
    """2x2 dashboard: loss curve with modes, curvature+grad, lorentz, lifts."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    losses = history.get("losses", [])
    modes = history.get("modes", [])
    steps = list(range(len(losses)))

    mode_colors = {"explore": "#cce5ff", "navigate": "#d4edda", "sprint": "#fff3cd"}

    # (0,0) Loss curve with mode bands
    ax = axes[0, 0]
    if losses:
        ax.semilogy(steps, losses, 'b-', linewidth=0.8)
        # Mode background bands
        if modes:
            prev_mode = modes[0]
            start = 0
            for i, m in enumerate(modes + [None]):
                if m != prev_mode or i == len(modes):
                    ax.axvspan(start, i, alpha=0.15,
                               color=mode_colors.get(prev_mode, '#ffffff'))
                    prev_mode = m
                    start = i
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
    ax.set_title("Loss Curve")
    ax.grid(True, alpha=0.3)
    # Legend for modes
    patches = [Patch(facecolor=c, alpha=0.3, label=m.upper())
               for m, c in mode_colors.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    # (0,1) Curvature + gradient norm
    ax = axes[0, 1]
    curv = history.get("curvatures", [])
    gnorms = history.get("grad_norms", [])
    probe_steps = history.get("probe_steps", [])
    if curv and probe_steps:
        ax.plot(probe_steps[:len(curv)], curv, 'b.-', label='Curvature', markersize=3)
        ax.set_ylabel("Mean Curvature", color='b')
        ax.tick_params(axis='y', labelcolor='b')
        if gnorms:
            ax2 = ax.twinx()
            ax2.plot(probe_steps[:len(gnorms)], gnorms, 'r.-',
                     label='Grad Norm', markersize=3)
            ax2.set_ylabel("Gradient Norm", color='r')
            ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title("Curvature & Gradient")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # (1,0) Lorentz warp
    ax = axes[1, 0]
    betas = history.get("betas", [])
    if betas and probe_steps:
        ax.plot(probe_steps[:len(betas)], betas, 'g.-', label='beta', markersize=3)
        gammas = [1.0 / max(math.sqrt(1.0 - b**2), 1e-6) for b in betas]
        ax.plot(probe_steps[:len(gammas)], gammas, 'm.-', label='gamma', markersize=3)
        # Shade plateau regions
        plateaus = history.get("plateaus", [])
        for ps, pe in plateaus:
            ax.axvspan(ps, pe, alpha=0.15, color='yellow')
        ax.legend(fontsize=8)
    ax.set_title("Lorentz Warp State")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # (1,1) Lift oracle events
    ax = axes[1, 1]
    if losses:
        ax.semilogy(steps, losses, 'b-', linewidth=0.5, alpha=0.5)
    lifts = history.get("lifts", [])
    for lift in lifts:
        color = 'green' if lift.get("success", False) else 'red'
        ax.scatter(lift["step"], lift["loss"], c=color, s=40, zorder=5)
        ax.annotate(f'σ={lift.get("sigma", 0):.2f}',
                    (lift["step"], lift["loss"]), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))
    ax.set_title("Lift Oracle Events")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "optimization_trajectory.png")
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

    # (0,0) Per-group FIM
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

    # (0,1) Commutativity matrix heatmap
    ax = axes[0, 1]
    gs = diagnostics.get("geometric_scores", {})
    if "comm_result" in gs:
        mat = gs["comm_result"].commutativity_matrix.cpu().numpy()
        im = ax.imshow(mat, cmap='RdYlGn_r', aspect='auto')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Commutativity Matrix")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Dimension")
        # Overlay coloring partition
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

    # (1,0) Per-group update scales
    ax = axes[1, 0]
    scales = diagnostics.get("scales", [])
    if scales:
        x = list(range(len(scales)))
        # Color by dominant signal
        bar_colors = []
        for s in scales:
            if s > 1.3:
                bar_colors.append('green')    # closure trust
            elif s < 0.5:
                bar_colors.append('red')      # FIM caution
            elif s < 0.8:
                bar_colors.append('orange')   # coherence gate
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

    # (1,1) Grade energy + entropy + closure annotation
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
        print("  [plot_topology_map] Skipping: not a 2D model")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Loss surface contour
    xr = np.linspace(-2.5, 2.5, 100)
    yr = np.linspace(-1.5, 3.5, 100)
    X, Y = np.meshgrid(xr, yr)
    Z = (model.a - X) ** 2 + model.b * (Y - X ** 2) ** 2
    ax.contourf(X, Y, np.log10(Z + 1e-10), levels=30, cmap='terrain', alpha=0.7)
    ax.contour(X, Y, np.log10(Z + 1e-10), levels=15, colors='gray',
               linewidths=0.3, alpha=0.5)

    # Critical points
    cp_markers = {
        CriticalPointType.MINIMUM: ('o', 'blue', 'Minimum'),
        CriticalPointType.SADDLE: ('^', 'red', 'Saddle'),
        CriticalPointType.MAXIMUM: ('s', 'gray', 'Maximum'),
    }
    for cp in landscape.critical_points:
        if cp.params.shape[0] >= 2:
            marker, color, label = cp_markers.get(
                cp.point_type, ('x', 'black', 'Unknown'))
            ax.scatter(cp.params[0].item(), cp.params[1].item(),
                       marker=marker, c=color, s=80, zorder=5,
                       edgecolors='white', linewidths=1)

    # Trajectory
    if trajectory:
        mode_colors_traj = {"explore": "blue", "navigate": "green", "sprint": "orange"}
        for i in range(1, len(trajectory)):
            m = modes[i] if i < len(modes) else "explore"
            ax.plot([trajectory[i - 1][0], trajectory[i][0]],
                    [trajectory[i - 1][1], trajectory[i][1]],
                    color=mode_colors_traj.get(m, "blue"), linewidth=0.5, alpha=0.7)
        # Start and end markers
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


def plot_comparison(
    results: Dict[str, List[float]],
    title: str = "GDO vs Adam",
    output_dir: str = "gdo_plots",
):
    """1x2: overlaid loss curves + final loss bar chart."""
    _ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overlaid curves
    ax = axes[0]
    styles = {"GDO": ("b-", 1.0), "Adam": ("r--", 0.8)}
    for name, losses in results.items():
        style, lw = styles.get(name, ("g-", 0.8))
        ax.semilogy(losses, style, label=name, linewidth=lw)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: final loss bar chart
    ax = axes[1]
    names = list(results.keys())
    finals = [results[n][-1] if results[n] else 0 for n in names]
    bar_colors = ['steelblue' if n == 'GDO' else 'salmon' for n in names]
    ax.bar(names, finals, color=bar_colors, edgecolor='white')
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Loss Comparison")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_registration_result(
    model: RotorRegistrationModel,
    history: Dict,
    output_dir: str = "gdo_plots",
):
    """1x3: 3D point clouds, angular error, loss with modes."""
    _ensure_output_dir(output_dir)
    fig = plt.figure(figsize=(18, 5))

    # Left: 3D scatter
    ax = fig.add_subplot(131, projection='3d')
    src = model.source.cpu().numpy()
    tgt = model.target.cpu().numpy()
    # Predict current
    with torch.no_grad():
        source_mv = model.algebra.embed_vector(model.source).unsqueeze(1)
        pred_mv = model.rotor(source_mv).squeeze(1)
        pred_pts = model._extract_vector(pred_mv).cpu().numpy()
    ax.scatter(src[:, 0], src[:, 1], src[:, 2], c='blue', s=15, alpha=0.6, label='Source')
    ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], c='red', s=15, alpha=0.6, label='Target')
    ax.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2],
               c='green', s=15, alpha=0.6, label='Predicted')
    ax.set_title("Point Clouds")
    ax.legend(fontsize=7)

    # Center: angular error
    ax = fig.add_subplot(132)
    angle_errors = history.get("angle_errors", [])
    if angle_errors:
        ax.plot(angle_errors, 'b-', linewidth=0.8)
        ax.set_ylabel("Angle Error (rad)")
        ax.set_xlabel("Step")
        ax.set_title(f"Angular Error (final={angle_errors[-1]:.4f} rad)")
    ax.grid(True, alpha=0.3)

    # Right: loss with mode transitions
    ax = fig.add_subplot(133)
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

    plt.tight_layout()
    path = os.path.join(output_dir, "registration_result.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_dimension_analysis(
    pre_result: PreExplorationResult,
    title: str = "Dimension Analysis",
    output_dir: str = "gdo_plots",
):
    """Detailed dimension analysis: eigenvalues, explained variance, condition number."""
    if pre_result.dim_result is None:
        return None
    _ensure_output_dir(output_dir)
    d = pre_result.dim_result
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # (0,0) Eigenvalue spectrum with broken-stick overlay
    ax = axes[0, 0]
    ev = d.eigenvalues.cpu().numpy()
    n_comp = len(ev)
    ax.semilogy(range(1, n_comp + 1), ev, 'b.-', label='Eigenvalues')
    # Broken-stick expected values
    bs = np.zeros(n_comp)
    for i in range(n_comp):
        bs[i] = sum(1.0 / (j + 1) for j in range(i, n_comp)) / n_comp
    bs_scaled = bs * ev.sum()
    ax.semilogy(range(1, n_comp + 1), bs_scaled, 'r--', alpha=0.7, label='Broken-stick')
    ax.axvline(d.broken_stick_threshold, color='green', linestyle=':', alpha=0.7,
               label=f'Threshold={d.broken_stick_threshold}')
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalue Spectrum (intrinsic_dim={d.intrinsic_dim})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Explained variance ratio (cumulative)
    ax = axes[0, 1]
    evr = d.explained_variance_ratio.cpu().numpy()
    cum_evr = np.cumsum(evr)
    ax.bar(range(1, len(evr) + 1), evr, color='steelblue', alpha=0.7, label='Individual')
    ax.plot(range(1, len(cum_evr) + 1), cum_evr, 'ro-', markersize=3, label='Cumulative')
    ax.axhline(0.95, color='green', linestyle='--', alpha=0.5, label='95%')
    n95 = int(np.searchsorted(cum_evr, 0.95)) + 1
    ax.axvline(n95, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance Ratio")
    ax.set_title(f"Explained Variance (95% at {n95} components)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Condition number and participation ratio
    ax = axes[1, 0]
    cond = ev[0] / max(ev[-1], 1e-15)
    metrics = ['Participation\nRatio', 'Intrinsic\nDim', 'log10(Cond.\nNumber)']
    values = [d.participation_ratio, float(d.intrinsic_dim), np.log10(max(cond, 1))]
    colors = ['steelblue', 'seagreen', 'coral']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_title("Dimension Summary")
    ax.grid(True, alpha=0.3, axis='y')

    # (1,1) Local dimension distribution (if available) else eigenvalue gap ratios
    ax = axes[1, 1]
    if d.local_dims is not None:
        ld = d.local_dims.cpu().numpy()
        ax.hist(ld, bins=min(30, max(5, int(np.sqrt(len(ld))))),
                color='mediumpurple', edgecolor='white', alpha=0.8)
        ax.axvline(d.participation_ratio, color='red', linestyle='--',
                   label=f'PR={d.participation_ratio:.1f}')
        ax.axvline(d.intrinsic_dim, color='green', linestyle=':',
                   label=f'Intrinsic={d.intrinsic_dim}')
        ax.set_xlabel("Local Dimension")
        ax.set_ylabel("Count")
        ax.set_title("Local Dimension Distribution")
        ax.legend(fontsize=8)
    else:
        # Eigenvalue gap ratios
        if len(ev) > 1:
            gaps = ev[:-1] / np.maximum(ev[1:], 1e-15)
            ax.bar(range(1, len(gaps) + 1), np.log10(np.clip(gaps, 1, None)),
                   color='coral', alpha=0.7, edgecolor='white')
            ax.set_xlabel("Gap Index (i / i+1)")
            ax.set_ylabel("log10(Gap Ratio)")
            ax.set_title("Eigenvalue Gap Ratios")
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "dimension_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_symmetry_analysis(
    pre_result: PreExplorationResult,
    title: str = "Symmetry Analysis",
    output_dir: str = "gdo_plots",
):
    """Symmetry detection results: null scores, reflections, involution, continuous dim."""
    if pre_result.symmetry_result is None:
        return None
    _ensure_output_dir(output_dir)
    sy = pre_result.symmetry_result
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # (0,0) Null scores per basis vector
    ax = axes[0, 0]
    ns = sy.null_scores.cpu().numpy()
    n = len(ns)
    colors = ['red' if i in sy.null_directions else 'steelblue' for i in range(n)]
    ax.bar(range(n), ns, color=colors, alpha=0.7, edgecolor='white')
    ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5, label='Null threshold')
    ax.set_xlabel("Basis Vector Index")
    ax.set_ylabel("Null Score")
    ax.set_title(f"Null Direction Scores ({len(sy.null_directions)} null)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Reflection symmetry scores
    ax = axes[0, 1]
    if sy.reflection_symmetries:
        dirs = [r["direction"] for r in sy.reflection_symmetries]
        scores = [r["score"] for r in sy.reflection_symmetries]
        bar_colors = ['green' if s < 0.1 else 'orange' if s < 0.3 else 'red'
                      for s in scores]
        ax.bar(range(len(dirs)), scores, color=bar_colors, alpha=0.7, edgecolor='white')
        ax.set_xticks(range(len(dirs)))
        ax.set_xticklabels([f'e{d}' for d in dirs], fontsize=8)
        ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='Symmetric')
        n_sym = sum(1 for s in scores if s < 0.1)
        ax.set_title(f"Reflection Symmetries ({n_sym} detected)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No reflections\nanalyzed", ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel("Direction")
    ax.set_ylabel("Asymmetry Score")
    ax.grid(True, alpha=0.3)

    # (1,0) Involution symmetry gauge
    ax = axes[1, 0]
    inv = sy.involution_symmetry
    # Horizontal bar showing even/odd balance
    ax.barh(['Grade\nInvolution'], [inv], color='mediumpurple', alpha=0.7, height=0.4)
    ax.barh(['Continuous\nSymmetry Dim'], [sy.continuous_symmetry_dim],
            color='teal', alpha=0.7, height=0.4)
    ax.set_xlim(0, max(1.0, sy.continuous_symmetry_dim + 0.5))
    ax.set_title("Symmetry Measures")
    # Annotate
    ax.text(inv + 0.02, 0, f'{inv:.3f}', va='center', fontsize=10)
    ax.text(sy.continuous_symmetry_dim + 0.02, 1,
            f'{sy.continuous_symmetry_dim}', va='center', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # (1,1) Summary text panel
    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        f"Null directions: {sy.null_directions}",
        f"Involution symmetry: {inv:.4f}",
        f"  (0 = purely even subalgebra, 1 = balanced)",
        f"Continuous symmetry dim: {sy.continuous_symmetry_dim}",
        f"Reflection symmetries: {sum(1 for r in sy.reflection_symmetries if r['score'] < 0.1)}"
        f" / {len(sy.reflection_symmetries)}",
        "",
        "Interpretation:",
    ]
    if inv < 0.2:
        lines.append("  Data lives near even subalgebra (rotors)")
    elif inv > 0.8:
        lines.append("  Data has balanced even/odd content")
    else:
        lines.append("  Mixed even/odd content")
    if sy.continuous_symmetry_dim > 0:
        lines.append(f"  {sy.continuous_symmetry_dim}D continuous symmetry group detected")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "symmetry_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_geodesic_flow(
    pre_result: PreExplorationResult,
    title: str = "Geodesic Flow Analysis",
    output_dir: str = "gdo_plots",
):
    """Flow bivector magnitudes, per-point coherence, causal report."""
    _ensure_output_dir(output_dir)
    has_flow = pre_result.flow_bivectors is not None
    has_coh = pre_result.per_point_coherence is not None
    has_causal = pre_result.causal_report is not None
    if not (has_flow or has_coh or has_causal
            or pre_result.landscape_coherence > 0):
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # (0,0) Flow bivector magnitudes
    ax = axes[0, 0]
    if has_flow:
        mags = pre_result.flow_bivectors.norm(dim=-1).cpu().numpy()
        ax.bar(range(len(mags)), mags, color='steelblue', alpha=0.7, edgecolor='white')
        ax.set_xlabel("Data Point")
        ax.set_ylabel("Flow Bivector Magnitude")
        ax.set_title(f"Flow Field Magnitudes (mean={mags.mean():.4f})")
        ax.axhline(mags.mean(), color='red', linestyle='--', alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No flow bivectors\n(no algebra)", ha='center',
                va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # (0,1) Per-point coherence distribution
    ax = axes[0, 1]
    if has_coh:
        coh_vals = pre_result.per_point_coherence.cpu().numpy()
        ax.hist(coh_vals, bins=min(25, max(5, len(coh_vals) // 3)),
                color='seagreen', edgecolor='white', alpha=0.8)
        ax.axvline(coh_vals.mean(), color='red', linestyle='--',
                   label=f'Mean={coh_vals.mean():.3f}')
        ax.set_xlabel("Coherence")
        ax.set_ylabel("Count")
        ax.set_title("Per-Point Coherence Distribution")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No per-point\ncoherence", ha='center',
                va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # (1,0) Landscape coherence vs curvature scatter / gauge
    ax = axes[1, 0]
    coh = pre_result.landscape_coherence
    curv = pre_result.landscape_curvature
    ax.scatter([coh], [curv], s=200, c='darkorange', edgecolors='black',
               zorder=5, marker='*')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Shade regions
    ax.axvspan(0.5, 1.0, ymin=0, ymax=0.5, alpha=0.08, color='green')
    ax.axvspan(0, 0.5, ymin=0.5, ymax=1.0, alpha=0.08, color='red')
    ax.text(0.75, 0.2, 'Causal', ha='center', fontsize=10, color='green', alpha=0.6)
    ax.text(0.25, 0.8, 'Noisy', ha='center', fontsize=10, color='red', alpha=0.6)
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Curvature")
    ax.set_title(f"Landscape Flow (coh={coh:.3f}, curv={curv:.3f})")
    ax.grid(True, alpha=0.3)

    # (1,1) Causal report text + loss landscape stats
    ax = axes[1, 1]
    ax.axis('off')
    lines = ["Geodesic Flow Summary", "=" * 30]
    if has_causal:
        cr = pre_result.causal_report
        lines.append(f"  Verdict: {cr['label']}")
        lines.append(f"  Coherence: {cr['coherence']:.4f}")
        lines.append(f"  Curvature: {cr['curvature']:.4f}")
    else:
        lines.append(f"  Landscape coherence: {coh:.4f}")
        lines.append(f"  Landscape curvature: {curv:.4f}")
    lines.append("")
    ls = pre_result.loss_statistics
    if ls:
        lines.append("Loss Landscape Statistics")
        lines.append("=" * 30)
        lines.append(f"  Mean:   {ls.get('mean', 0):.6f}")
        lines.append(f"  Std:    {ls.get('std', 0):.6f}")
        lines.append(f"  Min:    {ls.get('min', 0):.6f}")
        lines.append(f"  Max:    {ls.get('max', 0):.6f}")
        lines.append(f"  Median: {ls.get('median', 0):.6f}")
        lines.append(f"  IQR:    [{ls.get('q25', 0):.6f}, {ls.get('q75', 0):.6f}]")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "geodesic_flow.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_exchange_spectrum(
    pre_result: PreExplorationResult,
    title: str = "Exchange Spectrum & Lie Structure",
    output_dir: str = "gdo_plots",
):
    """Exchange spectrum eigenvalues, structure constants heatmap, Lie bracket closure."""
    if pre_result.commutator_result is None:
        return None
    _ensure_output_dir(output_dir)
    c = pre_result.commutator_result
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # (0,0) Exchange spectrum eigenvalues
    ax = axes[0, 0]
    es = c.exchange_spectrum.cpu().numpy()
    if len(es) > 0:
        ax.semilogy(range(1, len(es) + 1), np.maximum(es, 1e-15), 'b.-')
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue Magnitude")
        ax.set_title(f"Exchange Spectrum (top={es[0]:.4f})")
        # Mark significant eigenvalues
        threshold = es[0] * 0.01 if es[0] > 0 else 0
        n_sig = int(np.sum(es > threshold))
        ax.axhline(threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'{n_sig} significant')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Algebra too large\nfor full spectrum",
                ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    # (0,1) Commutativity matrix heatmap
    ax = axes[0, 1]
    cm = c.commutativity_matrix.cpu().numpy()
    if cm.size > 0:
        im = ax.imshow(cm, cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("Basis Direction j")
        ax.set_ylabel("Basis Direction i")
        ax.set_title(f"Commutativity Matrix (mean={cm.mean():.4f})")
    else:
        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)

    # (1,0) Structure constants heatmap (sum over third index for visualization)
    ax = axes[1, 0]
    sc = c.lie_bracket_structure.get("structure_constants", None)
    if sc is not None and sc.numel() > 0:
        sc_np = sc.cpu().numpy()
        k = sc_np.shape[0]
        # Frobenius norm over the c-index: ||c_{a,b,:}||
        sc_norm = np.sqrt((sc_np ** 2).sum(axis=2))
        im = ax.imshow(sc_norm, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("Bivector b")
        ax.set_ylabel("Bivector a")
        ax.set_title(f"Structure Constants ||c_{{ab}}|| (k={k})")
    else:
        ax.text(0.5, 0.5, "No structure\nconstants", ha='center',
                va='center', transform=ax.transAxes)

    # (1,1) Summary: closure error, mean commutator norm, basis indices
    ax = axes[1, 1]
    ax.axis('off')
    ce = c.lie_bracket_structure.get("closure_error", None)
    bi = c.lie_bracket_structure.get("basis_indices", [])
    lines = [
        "Lie Bracket Analysis",
        "=" * 30,
        f"  Mean commutator norm: {c.mean_commutator_norm:.6f}",
    ]
    if ce is not None:
        lines.append(f"  Lie bracket closure error: {ce:.6f}")
        if ce < 0.05:
            lines.append("  -> Bivectors form a closed Lie subalgebra")
        elif ce < 0.2:
            lines.append("  -> Approximate closure (near-Lie structure)")
        else:
            lines.append("  -> Poor closure (no clear Lie subalgebra)")
    lines.append(f"  Basis bivector indices: {bi[:10]}")
    if len(bi) > 10:
        lines.append(f"    ... ({len(bi)} total)")
    lines.append("")
    lines.append("Exchange Spectrum Summary")
    lines.append("=" * 30)
    if len(es) > 0:
        lines.append(f"  Largest eigenvalue: {es[0]:.6f}")
        lines.append(f"  Spectral gap: {(es[0] - es[1]):.6f}" if len(es) > 1 else "")
        n_nonzero = int(np.sum(es > 1e-8))
        lines.append(f"  Non-zero eigenvalues: {n_nonzero}/{len(es)}")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "exchange_spectrum.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_dimension_lifting(
    pre_result: PreExplorationResult,
    title: str = "Dimension Lifting Analysis",
    output_dir: str = "gdo_plots",
):
    """DimensionLifter results: original vs positive vs null lift coherence/curvature."""
    if pre_result.lifting_report is None:
        return None
    _ensure_output_dir(output_dir)
    lr = pre_result.lifting_report
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    configs = ['original', 'lift_positive', 'lift_null']
    labels = ['Original', 'Positive Lift\n(+1 spacelike)', 'Null Lift\n(+1 timelike)']
    cohs = []
    curvs = []
    sigs = []
    for cfg in configs:
        entry = lr.get(cfg, {})
        cohs.append(entry.get('coherence', 0))
        curvs.append(entry.get('curvature', 0))
        sig = entry.get('signature', (0, 0))
        sigs.append(f"Cl({sig[0]},{sig[1]})")

    best = lr.get('best', 'original')

    # Left: Coherence comparison
    ax = axes[0]
    bar_colors = ['gold' if cfg == best else 'steelblue' for cfg in configs]
    bars = ax.bar(range(3), cohs, color=bar_colors, alpha=0.7, edgecolor='white')
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{l}\n{s}" for l, s in zip(labels, sigs)], fontsize=8)
    ax.set_ylabel("Coherence")
    ax.set_title("Geodesic Flow Coherence")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, cohs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Center: Curvature comparison
    ax = axes[1]
    bar_colors = ['gold' if cfg == best else 'coral' for cfg in configs]
    bars = ax.bar(range(3), curvs, color=bar_colors, alpha=0.7, edgecolor='white')
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{l}\n{s}" for l, s in zip(labels, sigs)], fontsize=8)
    ax.set_ylabel("Curvature")
    ax.set_title("Geodesic Flow Curvature")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, curvs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Summary + recommendation
    ax = axes[2]
    ax.axis('off')
    lines = [
        "Dimension Lifting Report",
        "=" * 30,
    ]
    for cfg, label in zip(configs, ['Original', 'Positive Lift', 'Null Lift']):
        entry = lr.get(cfg, {})
        sig = entry.get('signature', (0, 0))
        coh = entry.get('coherence', 0)
        curv = entry.get('curvature', 0)
        causal = entry.get('causal', False)
        marker = " <-- BEST" if cfg == best else ""
        lines.append(f"  {label} Cl({sig[0]},{sig[1]}):{marker}")
        lines.append(f"    Coherence: {coh:.4f}")
        lines.append(f"    Curvature: {curv:.4f}")
        lines.append(f"    Causal: {'Yes' if causal else 'No'}")
        lines.append("")
    lines.append(f"Recommendation: {best.replace('_', ' ').title()}")
    if best == 'lift_positive':
        lines.append("  -> Adding spacelike dimension reveals structure")
    elif best == 'lift_null':
        lines.append("  -> Adding timelike dimension reveals structure")
    else:
        lines.append("  -> Current algebra is sufficient")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "dimension_lifting.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def plot_landscape_loss(
    pre_result: PreExplorationResult,
    title: str = "Loss Landscape Exploration",
    output_dir: str = "gdo_plots",
):
    """Loss landscape distribution, sorted loss profile, and loss vs distance."""
    if pre_result.landscape_losses is None:
        return None
    _ensure_output_dir(output_dir)
    losses = pre_result.landscape_losses.cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Left: Loss histogram
    ax = axes[0]
    ax.hist(losses, bins=min(30, max(5, len(losses) // 5)),
            color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(losses.mean(), color='red', linestyle='--', label=f'Mean={losses.mean():.4f}')
    ax.axvline(np.median(losses), color='green', linestyle=':', label=f'Median={np.median(losses):.4f}')
    ax.set_xlabel("Loss")
    ax.set_ylabel("Count")
    ax.set_title("Loss Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Center: Sorted loss profile
    ax = axes[1]
    sorted_losses = np.sort(losses)
    ax.plot(sorted_losses, 'b-', linewidth=0.8)
    ax.fill_between(range(len(sorted_losses)), sorted_losses, alpha=0.1, color='blue')
    ax.set_xlabel("Rank (sorted)")
    ax.set_ylabel("Loss")
    ax.set_title("Sorted Loss Profile")
    ax.grid(True, alpha=0.3)

    # Right: Loss vs distance from origin (first sample)
    ax = axes[2]
    if pre_result.landscape_positions is not None:
        positions = pre_result.landscape_positions.cpu().numpy()
        origin = positions[0]
        dists = np.linalg.norm(positions - origin, axis=-1)
        ax.scatter(dists, losses, s=8, alpha=0.5, c='steelblue', edgecolors='none')
        ax.set_xlabel("Distance from Origin")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Perturbation Distance")
        # Fit trend line
        if len(dists) > 3:
            z = np.polyfit(dists, losses, 2)
            d_sorted = np.sort(dists)
            ax.plot(d_sorted, np.polyval(z, d_sorted), 'r--', alpha=0.6, label='Quadratic fit')
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No positions\navailable", ha='center', va='center',
                transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "landscape_loss.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return fig


def _collect_history(optimizer, info, history):
    """Append step data to history dict."""
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
            "sigma": getattr(optimizer.lift_oracle, '_current_sigma', 0),
        })


def _new_history() -> Dict:
    return {
        "losses": [], "modes": [], "probe_steps": [],
        "curvatures": [], "grad_norms": [], "betas": [],
        "lifts": [], "plateaus": [], "trajectory": [],
        "angle_errors": [],
    }


def run_rosenbrock(steps: int = 2000, use_morse: bool = True,
                   init_flat: Optional[torch.Tensor] = None,
                   output_dir: str = "gdo_plots"):
    """Run GDO on Rosenbrock function."""
    print("\n" + "="*60)
    print("DEMO: Rosenbrock Function (a=1, b=100)")
    print("="*60)
    model = RosenbrockModel()
    if init_flat is not None:
        idx = 0
        for p in model.parameters():
            sz = p.numel()
            p.data.copy_(init_flat[idx:idx + sz].reshape(p.shape))
            idx += sz

    losses = []
    history = _new_history()

    # Pre-exploration (no algebra for Rosenbrock, but landscape analysis still useful)
    if use_morse:
        print("  Running pre-exploration analysis...")
        pre_analyzer = PreExplorationAnalyzer(n_samples=100)
        pre_result = pre_analyzer.analyze(model, model.forward)
        print(f"  Strategy: {pre_result.strategy_label}")
        plot_pre_exploration(pre_result, title="Rosenbrock Pre-Exploration", output_dir=output_dir)
        plot_dimension_analysis(pre_result, title="Rosenbrock Dimension", output_dir=output_dir)
        plot_landscape_loss(pre_result, title="Rosenbrock Loss Landscape", output_dir=output_dir)

        optimizer = GeometricDeterministicOptimizer(
            model=model,
            loss_fn=model.forward,
            lr=1e-3,
            probe_interval=100,
            topology_interval=500,
            sprint_after=1000,
        )
        for s in range(steps):
            loss = model()
            info = optimizer.optimize_step(loss)
            losses.append(info['loss'])
            _collect_history(optimizer, info, history)
            history["trajectory"].append((model.x.item(), model.y.item()))
            if s % 500 == 0 or s == steps - 1:
                x, y = model.x.item(), model.y.item()
                print(f"  Step {s:5d}: loss={info['loss']:.6f}  "
                      f"x={x:.4f}, y={y:.4f}  mode={info['mode']}")

        # Visualize
        plot_optimization_trajectory(
            history, title="Rosenbrock - GDO", output_dir=output_dir)
        plot_topology_map(
            model, optimizer.landscape,
            history["trajectory"], history["modes"], output_dir=output_dir)
        if optimizer._controller_diagnostics:
            plot_geometric_controller(
                optimizer._controller_diagnostics,
                title="Rosenbrock - Controller", output_dir=output_dir)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for s in range(steps):
            opt.zero_grad()
            loss = model()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if s % 500 == 0 or s == steps - 1:
                x, y = model.x.item(), model.y.item()
                print(f"  Step {s:5d}: loss={loss.item():.6f}  x={x:.4f}, y={y:.4f}")

    x_final, y_final = model.x.item(), model.y.item()
    print(f"\nFinal: x={x_final:.5f}, y={y_final:.5f}  "
          f"(optimum: x=1.0, y=1.0)")
    return losses


def run_rastrigin(n_dims: int = 4, steps: int = 3000, use_morse: bool = True,
                  init_flat: Optional[torch.Tensor] = None,
                  output_dir: str = "gdo_plots"):
    """Run GDO on Rastrigin function (many local minima)."""
    print("\n" + "="*60)
    print(f"DEMO: Rastrigin Function ({n_dims}D)")
    print("="*60)
    model = RastriginModel(n_dims=n_dims)
    if init_flat is not None:
        idx = 0
        for p in model.parameters():
            sz = p.numel()
            p.data.copy_(init_flat[idx:idx + sz].reshape(p.shape))
            idx += sz

    losses = []
    history = _new_history()

    if use_morse:
        print("  Running pre-exploration analysis...")
        pre_analyzer = PreExplorationAnalyzer(n_samples=100)
        pre_result = pre_analyzer.analyze(model, model.forward)
        print(f"  Strategy: {pre_result.strategy_label}")
        plot_pre_exploration(pre_result, title=f"Rastrigin {n_dims}D Pre-Exploration", output_dir=output_dir)
        plot_dimension_analysis(pre_result, title=f"Rastrigin {n_dims}D Dimension", output_dir=output_dir)
        plot_landscape_loss(pre_result, title=f"Rastrigin {n_dims}D Loss Landscape", output_dir=output_dir)

        optimizer = GeometricDeterministicOptimizer(
            model=model,
            loss_fn=model.forward,
            lr=1e-2,
            probe_interval=50,
            topology_interval=300,
            sprint_after=1500,
        )
        for s in range(steps):
            loss = model()
            info = optimizer.optimize_step(loss)
            losses.append(info['loss'])
            _collect_history(optimizer, info, history)
            if s % 500 == 0 or s == steps - 1:
                print(f"  Step {s:5d}: loss={info['loss']:.4f}  "
                      f"||x||={model.x.norm().item():.4f}  mode={info['mode']}")

        plot_optimization_trajectory(
            history, title=f"Rastrigin {n_dims}D - GDO", output_dir=output_dir)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for s in range(steps):
            opt.zero_grad()
            loss = model()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if s % 500 == 0 or s == steps - 1:
                print(f"  Step {s:5d}: loss={loss.item():.4f}  "
                      f"||x||={model.x.norm().item():.4f}")

    print(f"\nFinal loss: {losses[-1]:.5f}  (global optimum: 0.0)")
    return losses


def run_gbn(epochs: int = 50, device: str = 'cpu', output_dir: str = "gdo_plots"):
    """Run GDO on a small GBN regression task with pre-exploration."""
    print("\n" + "="*60)
    print(f"DEMO: Small GBN Regression (Cl(3,0), {epochs} epochs)")
    print("="*60)

    algebra = CliffordAlgebra(3, 0, device=device)
    model = SmallGBNModel(p=3, q=0, channels=4, device=device)
    dim = 2 ** 3  # = 8

    torch.manual_seed(42)
    X = torch.randn(32, 4, dim, device=device) * 0.3
    y_target = X[:, :, 0].mean(dim=1, keepdim=True)

    def loss_fn():
        out = model(X)
        pred = out[:, :, 0].mean(dim=1, keepdim=True)
        return F.mse_loss(pred, y_target)

    # Pre-exploration
    print("  Running pre-exploration analysis...")
    pre_analyzer = PreExplorationAnalyzer(
        algebra=algebra, n_samples=100, device=device)
    pre_result = pre_analyzer.analyze(model, loss_fn)
    print(f"  Strategy: {pre_result.strategy_label}")
    plot_pre_exploration(pre_result, title="GBN Pre-Exploration", output_dir=output_dir)
    plot_dimension_analysis(pre_result, title="GBN Dimension Analysis", output_dir=output_dir)
    plot_symmetry_analysis(pre_result, title="GBN Symmetry Analysis", output_dir=output_dir)
    plot_geodesic_flow(pre_result, title="GBN Geodesic Flow", output_dir=output_dir)
    plot_exchange_spectrum(pre_result, title="GBN Exchange Spectrum", output_dir=output_dir)
    plot_dimension_lifting(pre_result, title="GBN Dimension Lifting", output_dir=output_dir)
    plot_landscape_loss(pre_result, title="GBN Loss Landscape", output_dir=output_dir)

    optimizer = GeometricDeterministicOptimizer(
        model=model,
        loss_fn=loss_fn,
        lr=5e-4,
        probe_interval=20,
        topology_interval=100,
        sprint_after=300,
        algebra=algebra,
        device=device,
    )

    history = _new_history()
    losses = []
    for epoch in range(epochs):
        loss = loss_fn()
        info = optimizer.optimize_step(loss)
        losses.append(info['loss'])
        _collect_history(optimizer, info, history)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={info['loss']:.6f}  mode={info['mode']}")

    plot_optimization_trajectory(
        history, title="GBN - GDO", output_dir=output_dir)
    if optimizer._controller_diagnostics:
        plot_geometric_controller(
            optimizer._controller_diagnostics,
            title="GBN - Controller", output_dir=output_dir)

    print(f"\nFinal GBN loss: {losses[-1]:.6f}")
    return losses


def run_registration(
    steps: int = 1500,
    noise_std: float = 0.05,
    rotation_angle: float = 2.5,
    use_morse: bool = True,
    device: str = 'cpu',
    output_dir: str = "gdo_plots",
):
    """Run GDO on geometric rotor registration in Cl(3,0)."""
    print("\n" + "="*60)
    print(f"DEMO: Rotor Registration (Cl(3,0), angle={rotation_angle:.2f} rad)")
    print("="*60)

    model = RotorRegistrationModel(
        n_points=50, noise_std=noise_std,
        rotation_angle=rotation_angle, device=device,
    )

    # Pre-exploration
    print("  Running pre-exploration analysis...")
    pre_analyzer = PreExplorationAnalyzer(
        algebra=model.algebra, n_samples=150, device=device)
    pre_result = pre_analyzer.analyze(model, model.forward)
    print(f"  Strategy: {pre_result.strategy_label}")
    plot_pre_exploration(
        pre_result, title="Registration Pre-Exploration", output_dir=output_dir)
    plot_dimension_analysis(
        pre_result, title="Registration Dimension Analysis", output_dir=output_dir)
    plot_symmetry_analysis(
        pre_result, title="Registration Symmetry Analysis", output_dir=output_dir)
    plot_geodesic_flow(
        pre_result, title="Registration Geodesic Flow", output_dir=output_dir)
    plot_exchange_spectrum(
        pre_result, title="Registration Exchange Spectrum", output_dir=output_dir)
    plot_dimension_lifting(
        pre_result, title="Registration Dimension Lifting", output_dir=output_dir)
    plot_landscape_loss(
        pre_result, title="Registration Loss Landscape", output_dir=output_dir)

    losses = []
    history = _new_history()

    if use_morse:
        config = pre_result.recommended_config
        optimizer = GeometricDeterministicOptimizer(
            model=model,
            loss_fn=model.forward,
            algebra=model.algebra,
            device=device,
            config=config,
        )
        for s in range(steps):
            loss = model()
            info = optimizer.optimize_step(loss)
            losses.append(info['loss'])
            _collect_history(optimizer, info, history)
            history["angle_errors"].append(model.angular_error())
            if s % 300 == 0 or s == steps - 1:
                ae = model.angular_error()
                print(f"  Step {s:5d}: loss={info['loss']:.6f}  "
                      f"angle_err={ae:.4f} rad  mode={info['mode']}")

        plot_optimization_trajectory(
            history, title="Registration - GDO", output_dir=output_dir)
        if optimizer._controller_diagnostics:
            plot_geometric_controller(
                optimizer._controller_diagnostics,
                title="Registration - Controller", output_dir=output_dir)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for s in range(steps):
            opt.zero_grad()
            loss = model()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            history["losses"].append(loss.item())
            history["modes"].append("adam")
            history["angle_errors"].append(model.angular_error())
            if s % 300 == 0 or s == steps - 1:
                ae = model.angular_error()
                print(f"  Step {s:5d}: loss={loss.item():.6f}  angle_err={ae:.4f} rad")

    plot_registration_result(model, history, output_dir=output_dir)
    ae = model.angular_error()
    print(f"\nFinal loss: {losses[-1]:.6f}  angle_error: {ae:.4f} rad "
          f"({math.degrees(ae):.1f} deg)")
    return losses


def run_full_demo(steps: int = 500, device: str = 'cpu',
                  output_dir: str = "gdo_plots"):
    """Run all tasks + comparison plot."""
    print("\n" + "#"*60)
    print("# FULL GDO DEMO")
    print("#"*60)

    results = {}
    results["Rosenbrock"] = run_rosenbrock(steps=steps, output_dir=output_dir)
    results["Rastrigin"] = run_rastrigin(steps=steps, output_dir=output_dir)
    results["GBN"] = run_gbn(epochs=min(steps, 100), device=device, output_dir=output_dir)
    results["Registration"] = run_registration(
        steps=steps, device=device, output_dir=output_dir)

    # Comparison plot across all tasks
    plot_comparison(results, title="Full Demo - All Tasks", output_dir=output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, losses in results.items():
        print(f"  {name:15s}: final_loss = {losses[-1]:.6f}")

    return results


def compare_optimizers(task: str = 'rosenbrock', steps: int = 2000,
                       n_dims: int = 4, seed: int = 42,
                       output_dir: str = "gdo_plots"):
    """Compare GeometricDeterministicOptimizer vs baseline Adam side by side."""
    print("\n" + "="*60)
    print(f"COMPARISON: GDO vs Adam  (task={task})")
    print("="*60)

    torch.manual_seed(seed)
    if task == 'rosenbrock':
        ref_model = RosenbrockModel()
    elif task == 'rastrigin':
        ref_model = RastriginModel(n_dims=n_dims)
    elif task == 'registration':
        ref_model = RotorRegistrationModel()
    else:
        raise ValueError(f"Unknown task: {task}")
    init_flat = torch.cat([p.data.clone().reshape(-1) for p in ref_model.parameters()])

    if task == 'rosenbrock':
        print("\n[Adam Baseline]")
        adam_losses = run_rosenbrock(
            steps=steps, use_morse=False, init_flat=init_flat, output_dir=output_dir)
        print("\n[GeometricDeterministicOptimizer]")
        morse_losses = run_rosenbrock(
            steps=steps, use_morse=True, init_flat=init_flat, output_dir=output_dir)
    elif task == 'rastrigin':
        print("\n[Adam Baseline]")
        adam_losses = run_rastrigin(
            n_dims=n_dims, steps=steps, use_morse=False,
            init_flat=init_flat, output_dir=output_dir)
        print("\n[GeometricDeterministicOptimizer]")
        morse_losses = run_rastrigin(
            n_dims=n_dims, steps=steps, use_morse=True,
            init_flat=init_flat, output_dir=output_dir)
    else:
        print("\n[Adam Baseline]")
        adam_losses = run_registration(
            steps=steps, use_morse=False, output_dir=output_dir)
        print("\n[GeometricDeterministicOptimizer]")
        morse_losses = run_registration(
            steps=steps, use_morse=True, output_dir=output_dir)

    print("\n" + "-"*40)
    print(f"Adam final loss:  {adam_losses[-1]:.6f}")
    print(f"GDO final loss:   {morse_losses[-1]:.6f}")
    improvement = (adam_losses[-1] - morse_losses[-1]) / (abs(adam_losses[-1]) + 1e-8)
    print(f"Improvement: {improvement*100:+.1f}%")

    plot_comparison(
        {"GDO": morse_losses, "Adam": adam_losses},
        title=f"{task.capitalize()} - GDO vs Adam",
        output_dir=output_dir,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Geometric Deterministic Optimizer Experiment")
    p.add_argument(
        "--task",
        choices=["rosenbrock", "rastrigin", "gbn", "registration",
                 "compare", "full"],
        default="full",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n-dims", type=int, default=4)
    p.add_argument("--no-morse", action="store_true",
                   help="Use baseline Adam instead of GDO")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="gdo_plots",
                   help="Directory for saving plots")
    p.add_argument("--noise-std", type=float, default=0.05,
                   help="Noise std for registration task")
    p.add_argument("--rotation-angle", type=float, default=2.5,
                   help="Rotation angle (rad) for registration task")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Geometric Deterministic Optimizer (GDO) Experiment")
    print("Theory: Morse topology + curvature probes + geodesic paths")
    print("        + Lorentz warp + geometric parameter controller")
    print("        + dimensional lift oracle")

    od = args.output_dir

    if args.task == "rosenbrock":
        run_rosenbrock(steps=args.steps, use_morse=not args.no_morse, output_dir=od)
    elif args.task == "rastrigin":
        run_rastrigin(n_dims=args.n_dims, steps=args.steps,
                      use_morse=not args.no_morse, output_dir=od)
    elif args.task == "gbn":
        run_gbn(epochs=args.epochs, device=args.device, output_dir=od)
    elif args.task == "registration":
        run_registration(
            steps=args.steps, noise_std=args.noise_std,
            rotation_angle=args.rotation_angle,
            use_morse=not args.no_morse, device=args.device, output_dir=od)
    elif args.task == "full":
        run_full_demo(steps=args.steps, device=args.device, output_dir=od)
    elif args.task == "compare":
        compare_optimizers(task="rosenbrock", steps=args.steps, output_dir=od)
        compare_optimizers(task="rastrigin", steps=args.steps,
                           n_dims=args.n_dims, output_dir=od)
