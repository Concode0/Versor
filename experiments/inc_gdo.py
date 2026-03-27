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
from layers import RotorLayer, CliffordLinear, CliffordLayerNorm
from functional.activation import GeometricGELU


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


class CommutatorAnalyzer:
    """Identifies parameter groups that can be updated simultaneously.

    score(i,j) measures how much a gradient step on group i changes the
    gradient of group j -- i.e. the cross-sensitivity.

    This is a finite-difference approximation of the cross-Hessian block
    H_{ji} = d^2 L / (d theta_j * d theta_i), normalized by gradient magnitudes.

    score ~= 0 -> independent -> safe to update in parallel
    score > threshold -> coupled -> must update sequentially
    """

    def __init__(
        self,
        commutator_threshold: float = 0.3,
        fd_step: float = 1e-3,
    ):
        self.commutator_threshold = commutator_threshold
        self.fd_step = fd_step

    @staticmethod
    def _flat_group_grad(
        loss_fn: Callable[[], torch.Tensor],
        group: List[torch.nn.Parameter],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute flat gradient for a single parameter group."""
        loss = loss_fn()
        grads = torch.autograd.grad(loss, group, allow_unused=True)
        return torch.cat([
            g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
            for g, p in zip(grads, group)
        ])

    def compute_group_scores(
        self,
        loss_fn: Callable[[], torch.Tensor],
        param_groups: List[List[torch.nn.Parameter]],
    ) -> Dict[Tuple[int, int], float]:
        """Pairwise commutator scores via gradient sensitivity (FD cross-Hessian).

        For each group i:
          1. Compute baseline gradient g_j for all j at current theta
          2. Take a unit step on group i: theta_i -= fd_step * g_i / ||g_i||
          3. Recompute gradient g_j' for each j != i
          4. score(i,j) = ||g_j' - g_j|| / (||g_j|| + 1e-8)
        Final score(i,j) = max over directions i and j (symmetric).
        """
        if not param_groups:
            return {}

        n = len(param_groups)
        device = param_groups[0][0].device

        # Baseline gradients for all groups
        baseline: List[torch.Tensor] = []
        for g in param_groups:
            try:
                baseline.append(self._flat_group_grad(loss_fn, g, device))
            except Exception:
                baseline.append(torch.zeros(
                    sum(p.numel() for p in g), device=device
                ))

        # Save original data
        orig = {id(p): p.data.clone() for g in param_groups for p in g}
        scores: Dict[Tuple[int, int], float] = {
            (i, j): 0.0 for i in range(n) for j in range(i + 1, n)
        }

        for i in range(n):
            g_i_norm = baseline[i].norm().item()
            if g_i_norm < 1e-10:
                continue

            # Step on group i
            step_i = baseline[i] / g_i_norm * self.fd_step
            ptr = 0
            for p in param_groups[i]:
                sz = p.numel()
                p.data -= step_i[ptr:ptr + sz].reshape(p.shape)
                ptr += sz

            # Measure gradient shift in every other group
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

            # Restore group i
            for p in param_groups[i]:
                p.data.copy_(orig[id(p)])

        return scores

    def parallel_groups(
        self, scores: Dict[Tuple[int, int], float], n_groups: int
    ) -> List[List[int]]:
        """Partition group indices into parallel update sets.

        Groups are in the same parallel set if all pairwise commutator
        scores are below the threshold (safe to update together).
        Greedy graph coloring approach.
        """
        # Build conflict graph: edge (i,j) if score > threshold
        conflicts = {i: set() for i in range(n_groups)}
        for (i, j), s in scores.items():
            if s > self.commutator_threshold:
                conflicts[i].add(j)
                conflicts[j].add(i)

        # Greedy coloring
        colors = [-1] * n_groups
        for i in range(n_groups):
            used = {colors[c] for c in conflicts[i] if colors[c] >= 0}
            color = 0
            while color in used:
                color += 1
            colors[i] = color

        n_colors = max(colors) + 1
        schedule = [[] for _ in range(n_colors)]
        for i, c in enumerate(colors):
            schedule[c].append(i)

        return schedule


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
    ):
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
        self.commutator = CommutatorAnalyzer()
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

        # Use warp-scaled LR (global warp state set by probes during EXPLORE)
        lr_vec = self.warp.warped_lr(self.lr, group_grad.shape[0], device)

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
            # Build commutator schedule once
            if self._commutator_schedule is None:
                if len(self._param_groups) > 1:
                    print(f"  [Morse] Computing commutator schedule...")
                    scores = self.commutator.compute_group_scores(
                        self.loss_fn, self._param_groups
                    )
                    self.landscape.commutator_scores = {
                        f"({i},{j})": v for (i, j), v in scores.items()
                    }
                    self._commutator_schedule = self.commutator.parallel_groups(
                        scores, len(self._param_groups)
                    )
                    print(f"  [Morse] Commutator schedule: {self._commutator_schedule}")
                else:
                    self._commutator_schedule = [[0]] if self._param_groups else [[]]

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


def run_rosenbrock(steps: int = 2000, use_morse: bool = True,
                   init_flat: Optional[torch.Tensor] = None):
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
    if use_morse:
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
            if s % 500 == 0 or s == steps - 1:
                x, y = model.x.item(), model.y.item()
                print(f"  Step {s:5d}: loss={info['loss']:.6f}  "
                      f"x={x:.4f}, y={y:.4f}  mode={info['mode']}")
    else:
        # Baseline Adam
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
                  init_flat: Optional[torch.Tensor] = None):
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
    if use_morse:
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
            if s % 500 == 0 or s == steps - 1:
                print(f"  Step {s:5d}: loss={info['loss']:.4f}  "
                      f"||x||={model.x.norm().item():.4f}  mode={info['mode']}")
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


def run_gbn(epochs: int = 50, device: str = 'cpu'):
    """Run GDO on a small GBN regression task."""
    print("\n" + "="*60)
    print(f"DEMO: Small GBN Regression (Cl(3,0), {epochs} epochs)")
    print("="*60)

    algebra = CliffordAlgebra(3, 0, device=device)
    model = SmallGBNModel(p=3, q=0, channels=4, device=device)
    dim = 2 ** 3  # = 8

    # Synthetic data: random multivectors, target = grade-0 component
    torch.manual_seed(42)
    X = torch.randn(32, 4, dim, device=device) * 0.3
    y_target = X[:, :, 0].mean(dim=1, keepdim=True)  # grade-0 mean

    def loss_fn():
        out = model(X)
        pred = out[:, :, 0].mean(dim=1, keepdim=True)
        return F.mse_loss(pred, y_target)

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

    losses = []
    for epoch in range(epochs):
        loss = loss_fn()
        info = optimizer.optimize_step(loss)
        losses.append(info['loss'])
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={info['loss']:.6f}  mode={info['mode']}")

    print(f"\nFinal GBN loss: {losses[-1]:.6f}")
    return losses


def compare_optimizers(task: str = 'rosenbrock', steps: int = 2000,
                       n_dims: int = 4, seed: int = 42):
    """Compare GeometricDeterministicOptimizer vs baseline Adam side by side.

    Both optimizers start from the identical initial parameters so the
    comparison is apples-to-apples.
    """
    print("\n" + "="*60)
    print(f"COMPARISON: GDO vs Adam  (task={task})")
    print("="*60)

    # Create a reference model with fixed seed to capture shared initial params
    torch.manual_seed(seed)
    if task == 'rosenbrock':
        ref_model = RosenbrockModel()
    elif task == 'rastrigin':
        ref_model = RastriginModel(n_dims=n_dims)
    else:
        raise ValueError(f"Unknown task: {task}")
    init_flat = torch.cat([p.data.clone().reshape(-1) for p in ref_model.parameters()])

    if task == 'rosenbrock':
        print("\n[Adam Baseline]")
        adam_losses = run_rosenbrock(steps=steps, use_morse=False, init_flat=init_flat)
        print("\n[GeometricDeterministicOptimizer]")
        morse_losses = run_rosenbrock(steps=steps, use_morse=True, init_flat=init_flat)
    else:
        print("\n[Adam Baseline]")
        adam_losses = run_rastrigin(n_dims=n_dims, steps=steps,
                                    use_morse=False, init_flat=init_flat)
        print("\n[GeometricDeterministicOptimizer]")
        morse_losses = run_rastrigin(n_dims=n_dims, steps=steps,
                                     use_morse=True, init_flat=init_flat)

    print("\n" + "-"*40)
    print(f"Adam final loss:  {adam_losses[-1]:.6f}")
    print(f"GDO final loss:   {morse_losses[-1]:.6f}")
    improvement = (adam_losses[-1] - morse_losses[-1]) / (abs(adam_losses[-1]) + 1e-8)
    print(f"Improvement: {improvement*100:+.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description="Geometric Deterministic Optimizer Experiment")
    p.add_argument("--task", choices=["rosenbrock", "rastrigin", "gbn", "compare"],
                   default="compare")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n-dims", type=int, default=4)
    p.add_argument("--no-morse", action="store_true",
                   help="Use baseline Adam instead of GeometricDeterministicOptimizer")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Geometric Deterministic Optimizer (GDO) Experiment")
    print("Theory: Morse topology + curvature probes + geodesic paths")
    print("        + Lorentz warp + commutator groups + dimensional lift oracle")

    if args.task == "rosenbrock":
        run_rosenbrock(steps=args.steps, use_morse=not args.no_morse)
    elif args.task == "rastrigin":
        run_rastrigin(n_dims=args.n_dims, steps=args.steps, use_morse=not args.no_morse)
    elif args.task == "gbn":
        run_gbn(epochs=args.epochs, device=args.device)
    elif args.task == "compare":
        compare_optimizers(task="rosenbrock", steps=args.steps)
        compare_optimizers(task="rastrigin", steps=args.steps, n_dims=args.n_dims)
