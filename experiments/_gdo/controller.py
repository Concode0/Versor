"""GDOController: full Morse-geometric optimization pipeline orchestrator."""

from __future__ import annotations

import re
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from optimizers.riemannian import MANIFOLD_EUCLIDEAN

from .config import GDOConfig
from .dimensional_lift import DimensionalLiftOracle
from .optimizer import GDOOptimizer
from .parameter_groups import GeometricParameterController
from .probes import CurvatureProbe, GeodesicIntegrator, LorentzWarpOptimizer
from .topology import CriticalPointType, LandscapeMap, LandscapeTopologySearch


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

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            if algebra is not None:
                self.optimizer = GDOOptimizer.from_model(model, lr=lr, algebra=algebra)
            else:
                self.optimizer = GDOOptimizer(model.parameters(), lr=lr)

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

        self._navigate_steps: int = 0
        self._navigate_best_loss: float = float('inf')
        self._navigate_no_improve: int = 0

        self._sprint_step: int = 0
        self._last_schedule_loss: float = float('inf')
        self._last_schedule_grad_norms: Optional[torch.Tensor] = None

        self._param_group_meta: List[Dict] = []
        self._param_groups: List[List[nn.Parameter]] = self._build_param_groups()
        self._group_ranges: List[List[Tuple[int, int]]] = self._compute_group_ranges()

        self._hessian_vecs: Optional[torch.Tensor] = None
        self._hessian_vals: Optional[torch.Tensor] = None

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

        max_g = self.config.max_groups
        while len(merged_groups) > max_g:
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
        interval_trigger = (
            self._sprint_step > 0
            and self._sprint_step % self.config.reschedule_interval == 0
        )

        loss_trigger = False
        if self._last_schedule_loss < float('inf'):
            rel_improve = (
                (self._last_schedule_loss - current_loss)
                / (abs(self._last_schedule_loss) + 1e-8)
            )
            loss_trigger = rel_improve > self.config.reschedule_loss_delta

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
            old_p = self._last_schedule_grad_norms / (
                self._last_schedule_grad_norms.sum() + 1e-8
            )
            new_p = current_norms / (current_norms.sum() + 1e-8)
            eps = 1e-6
            old_p = old_p.clamp(min=eps)
            new_p = new_p.clamp(min=eps)
            kl = (new_p * (new_p / old_p).log()).sum().item()
            grad_trigger = kl > self.config.reschedule_grad_kl_threshold
            self.model.zero_grad()

        if not (interval_trigger or loss_trigger or grad_trigger):
            return False

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

        self._run_scheduling()
        self._last_schedule_loss = current_loss

        new_n_colors = (len(self._commutator_schedule)
                        if self._commutator_schedule else 0)

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

        elif self.mode == GDOController.Mode.SPRINT:
            if self._commutator_schedule is None:
                self._run_scheduling()

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


GeometricDeterministicOptimizer = GDOController
