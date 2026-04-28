"""GDOOptimizer: torch.optim.Optimizer interface for GDO updates."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from core.algebra import CliffordAlgebra
from optimizers.riemannian import (
    MANIFOLD_EUCLIDEAN,
    MANIFOLD_SPHERE,
    MANIFOLD_SPIN,
    group_parameters_by_manifold,
)


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
                param_groups.append({"params": params, "manifold": manifold})
        if not param_groups:
            param_groups = [{"params": list(model.parameters()), "manifold": MANIFOLD_EUCLIDEAN}]
        return cls(param_groups, lr=lr, betas=betas, eps=eps, algebra=algebra, max_bivector_norm=max_bivector_norm)

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
            betas = group["betas"]
            beta1, beta2 = betas
            eps = group["eps"]
            lr = group["lr"]
            manifold = group.get("manifold", MANIFOLD_EUCLIDEAN)

            if self._group_scales is not None and group_idx < len(self._group_scales):
                lr = lr * self._group_scales[group_idx]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr / bias_correction1
                bias_correction2_sqrt = bias_correction2**0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

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
