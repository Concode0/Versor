"""Pre-optimization landscape analyzer: recommends GDOConfig + strategy label."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algebra import CliffordAlgebra
from core.analysis import (
    CommutatorAnalyzer as CoreCommutatorAnalyzer,
)
from core.analysis import (
    DimensionLifter,
    EffectiveDimensionAnalyzer,
    GeodesicFlow,
    SamplingConfig,
    SpectralAnalyzer,
    StatisticalSampler,
    SymmetryDetector,
)
from core.analysis._types import (
    CommutatorResult,
    DimensionResult,
    SpectralResult,
    SymmetryResult,
)

from .config import GDOConfig
from .parameter_groups import GeometricParameterController


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
        device: str = "cpu",
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
            p.data.copy_(flat[idx : idx + sz].reshape(p.shape))
            idx += sz

    def _sample_landscape(self, model: nn.Module, loss_fn: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def analyze(self, model: nn.Module, loss_fn: Callable) -> PreExplorationResult:
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
                    "coherence": result.landscape_coherence,
                    "curvature": result.landscape_curvature,
                    "causal": (result.landscape_coherence > 0.5 and result.landscape_curvature < 0.5),
                    "label": (
                        "Causal - smooth, aligned flow"
                        if (result.landscape_coherence > 0.5 and result.landscape_curvature < 0.5)
                        else "Noisy - fragmented flow"
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
                        reduced_lift, p=lift_dim, q=0, k=min(8, reduced_lift.shape[0] - 1)
                    )
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
