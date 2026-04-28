"""Dimensional lift oracle: escape local minima via lift -> oracle search -> pull-down."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            p.data.copy_(flat[idx : idx + sz].reshape(p.shape))
            idx += sz

    @staticmethod
    def _flat_grad(model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        return torch.cat(
            [
                p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)
                for p in model.parameters()
            ]
        )

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
            flat_g = torch.cat(
                [
                    g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device)
                    for g, p in zip(grads, params)
                ]
            )
            gv = (flat_g * q_curr.detach()).sum()
            hvp = torch.autograd.grad(gv, params, allow_unused=True)
            Hq = torch.cat(
                [h.reshape(-1) if h is not None else torch.zeros(p.numel(), device=device) for h, p in zip(hvp, params)]
            )

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
        for i, b in enumerate(beta[1 : min(len(beta), m)]):
            T[i, i + 1] = b
            T[i + 1, i] = b

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

        print(
            f"  [LiftOracle] #{self._lift_count}: sigma={sigma:.4f}  oracle_lr={scaled_lr:.5f}  loss={current_loss:.5f}"
        )

        directions: List[torch.Tensor] = []

        if hessian_vecs is None:
            try:
                hessian_vals, hessian_vecs = self._compute_bottom_eigvecs(loss_fn, model, k=min(4, n))
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

        candidates = [flat_orig + sigma * d for d in directions[: self.k]]

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
                    m_hat = m / (1 - beta1**t)
                    v_hat = v / (1 - beta2**t)
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
            self.lift_sigma * (self.sigma_scale**self._consecutive_fails),
            self.max_sigma,
        )
        print(f"  [LiftOracle] [fail] no improvement. Escalating sigma: {sigma:.4f} -> {self._current_sigma:.4f}")
        return None, current_loss
