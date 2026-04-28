"""Morse-theoretic topology: critical points, landscape map, Lanczos-based search."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class CriticalPointType(Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SADDLE = "saddle"
    UNKNOWN = "unknown"


@dataclass
class CriticalPoint:
    """A detected critical point in the loss landscape."""

    params: torch.Tensor  # Parameter vector at critical point
    loss: float  # Loss value
    morse_index: int  # Number of negative Hessian eigenvalues
    eigenvalues: torch.Tensor  # Hessian eigenvalues (sorted ascending)
    point_type: CriticalPointType
    step: int = 0  # Training step when detected

    @classmethod
    def from_hessian(
        cls, params: torch.Tensor, loss: float, eigenvalues: torch.Tensor, step: int = 0
    ) -> "CriticalPoint":
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
        return (
            f"CriticalPoint({self.point_type.value}, loss={self.loss:.4f}, index={self.morse_index}, step={self.step})"
        )


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
        return [
            cp for cp in self.critical_points if cp.point_type == CriticalPointType.MINIMUM and cp.loss < current_loss
        ]

    def summary(self) -> str:
        n_min = sum(1 for c in self.critical_points if c.point_type == CriticalPointType.MINIMUM)
        n_sad = sum(1 for c in self.critical_points if c.point_type == CriticalPointType.SADDLE)
        return f"LandscapeMap: {n_min} minima, {n_sad} saddles | {len(self.plateau_episodes)} plateau episodes"


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

    def _hessian_vector_product(self, loss: torch.Tensor, params: List[torch.Tensor], v: torch.Tensor) -> torch.Tensor:
        """Compute H*v via double backprop."""
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        flat_grad = torch.cat(
            [g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device) for g, p in zip(grads, params)]
        )
        gv = (flat_grad * v.detach()).sum()
        hvp = torch.autograd.grad(gv, params, allow_unused=True)
        return torch.cat(
            [h.reshape(-1) if h is not None else torch.zeros(p.numel(), device=p.device) for h, p in zip(hvp, params)]
        )

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
        for i, b in enumerate(beta[1 : min(len(beta), m)]):
            T[i, i + 1] = b
            T[i + 1, i] = b

        eigvals, eigvecs_T = torch.linalg.eigh(T)
        Q_mat = torch.stack(Q[:m])
        H_vecs = F.normalize(Q_mat.T @ eigvecs_T, dim=0)
        return eigvals.detach(), H_vecs.detach()

    def check(self, loss: torch.Tensor, params: List[torch.Tensor], step: int) -> Optional[CriticalPoint]:
        """Check if current point is a critical point. Returns CriticalPoint or None."""
        self._step = step
        if step % self.detect_every != 0:
            return None

        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat_g = torch.cat(
            [g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device) for g, p in zip(grads, params)]
        )
        grad_norm = flat_g.norm().item()

        if grad_norm > self.grad_tol * 10:
            return None

        flat_p = self._flat_params(params)
        try:
            eigenvalues, eigenvecs = self._lanczos_eigenvalues(loss, params, k=min(20, flat_p.numel()))
            self._last_eigenvalues = eigenvalues
            self._last_eigenvecs = eigenvecs
            cp = CriticalPoint.from_hessian(flat_p, loss.item(), eigenvalues, step)
            return cp
        except Exception:
            return None
