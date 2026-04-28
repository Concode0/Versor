"""Geometric parameter controller: FIM, commutator scoring, DSatur coloring."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from core.analysis import (
    CommutatorAnalyzer as CoreCommutatorAnalyzer,
    GeodesicFlow,
    SpectralAnalyzer,
)
from layers import MultiRotorLayer, RotorLayer

from .config import GDOConfig


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
        return torch.cat(
            [g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=device) for g, p in zip(grads, group)]
        )

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
                    bv.shape[0],
                    m.algebra.dim,
                    device=bv.device,
                    dtype=bv.dtype,
                )
                full[:, m.grade_indices] = bv
                mv_list.append(full)
            elif isinstance(m, MultiRotorLayer):
                bv = m.rotor_grade_weights.detach()
                full = torch.zeros(
                    bv.shape[0],
                    m.algebra.dim,
                    device=bv.device,
                    dtype=bv.dtype,
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
            result["closure_error"] = comm_result.lie_bracket_structure.get("closure_error", 1.0)
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
                grade_energy = self.spectral.grade_energy_spectrum(mv_params.unsqueeze(1))
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
                baseline.append(torch.zeros(sum(p.numel() for p in g), device=device))

        orig = {id(p): p.data.clone() for g in param_groups for p in g}
        scores: Dict[Tuple[int, int], float] = {(i, j): 0.0 for i in range(n) for j in range(i + 1, n)}

        for i in range(n):
            g_i_norm = baseline[i].norm().item()
            if g_i_norm < 1e-10:
                continue

            step_i = baseline[i] / g_i_norm * self.fd_step
            ptr = 0
            for p in param_groups[i]:
                sz = p.numel()
                p.data -= step_i[ptr : ptr + sz].reshape(p.shape)
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
        scores: Dict[Tuple[int, int], float] = {(i, j): 0.0 for i in range(n) for j in range(i + 1, n)}

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
        group_grads: List[torch.Tensor] = []
        for g in param_groups:
            try:
                group_grads.append(self._flat_group_grad(loss_fn, g, device))
            except Exception:
                group_grads.append(torch.zeros(sum(p.numel() for p in g), device=device))

        group_norms = torch.tensor(
            [g.norm().item() for g in group_grads],
            device=device,
        )

        cosine_scores: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            gi_norm = group_norms[i].item()
            if gi_norm < 1e-10:
                continue
            gi_dir = group_grads[i] / (gi_norm + 1e-8)
            orig_data = {id(p): p.data.clone() for p in param_groups[i]}
            step = gi_dir * self.fd_step
            ptr = 0
            for p in param_groups[i]:
                sz = p.numel()
                p.data -= step[ptr : ptr + sz].reshape(p.shape)
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

        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                alg = alg_scores.get(key, 0.0)
                fd = cosine_scores.get(key, 0.0)
                scores[key] = 0.4 * alg + 0.6 * fd

        return scores

    def parallel_groups(self, scores: Dict[Tuple[int, int], float], n_groups: int) -> List[List[int]]:
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
        use_manifold = self.config.manifold_compat_constraint and group_meta is not None

        adj: Dict[int, Dict[int, float]] = {i: {} for i in range(n_groups)}
        for (i, j), w in scores.items():
            if w > 0:
                adj[i][j] = w
                adj[j][i] = w

        colors = [-1] * n_groups
        neighbor_colors: List[set] = [set() for _ in range(n_groups)]
        color_members: Dict[int, List[int]] = {}

        for _ in range(n_groups):
            best_node = -1
            best_sat = -1
            best_wdeg = -1.0
            for node in range(n_groups):
                if colors[node] >= 0:
                    continue
                sat = len(neighbor_colors[node])
                wdeg = sum(adj[node].get(nb, 0.0) for nb in range(n_groups) if colors[nb] >= 0)
                if sat > best_sat or (sat == best_sat and wdeg > best_wdeg):
                    best_node = node
                    best_sat = sat
                    best_wdeg = wdeg
            if best_node < 0:
                break

            node_manifold = group_meta[best_node].get("manifold") if use_manifold else None
            assigned_color = -1
            for c in sorted(color_members.keys()):
                has_hard_conflict = any(
                    adj[best_node].get(m, 0.0) > self.commutator_threshold for m in color_members[c]
                )
                if has_hard_conflict:
                    continue
                total_weight = sum(adj[best_node].get(m, 0.0) for m in color_members[c])
                if total_weight > budget:
                    continue
                if use_manifold:
                    compat = all(
                        group_meta[m].get("manifold") == node_manifold or adj[best_node].get(m, 0.0) == 0.0
                        for m in color_members[c]
                    )
                    if not compat:
                        continue
                assigned_color = c
                break

            if assigned_color < 0:
                assigned_color = len(color_members)

            colors[best_node] = assigned_color
            color_members.setdefault(assigned_color, []).append(best_node)

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

        if self.config.interaction_estimation == "efficient":
            hybrid_scores = self.build_hybrid_scores_efficient(
                loss_fn,
                param_groups,
                geo_scores,
            )
        elif self.config.interaction_estimation == "gradient_only":
            hybrid_scores = self.build_hybrid_scores_efficient(
                loss_fn,
                param_groups,
                {},
            )
        else:
            hybrid_scores = self.build_hybrid_scores(
                loss_fn,
                param_groups,
                geo_scores,
            )

        if self.config.dsatur_enabled:
            schedule = self.parallel_groups_dsatur(
                hybrid_scores,
                len(param_groups),
                group_meta,
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
