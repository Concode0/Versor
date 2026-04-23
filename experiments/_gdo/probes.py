"""Curvature probe, geodesic integrator, Lorentz warp — local geometry sub-components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


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
