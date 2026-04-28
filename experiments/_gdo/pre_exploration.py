"""Pre-optimization landscape analyzer: recommends GDOConfig + strategy label.

Two analyses run in :meth:`PreExplorationAnalyzer.analyze`:

1. **Landscape side**: samples loss around the init point, runs
   dimension/spectral/symmetry/commutator/coherence on the parameter cloud,
   emits a ``GDOConfig`` recommendation consumed by ``GDOController``.

2. **Architecture side**: walks the model for rotor-bearing layers,
   decomposes each layer's bivector parameter into simple planes, optionally
   runs a single forward pass to capture per-layer activation
   coherence/curvature via ``GeodesicFlow``, and emits a general
   ``TuningRecommendation`` (lr, init scale, freeze list, signature lift,
   notes). Mirrors the way ``StatisticalSampler`` uses coherence to stratify
   data -- here it stratifies layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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
class LayerTopology:
    """Per-layer geometric snapshot.

    ``plane_basis`` holds simple (2-blade) bivector components decomposed from
    the layer's grade-2 parameter; for grade-1 (``ReflectionLayer``) layers it
    holds the unit reflection vectors instead. ``activation_*`` fields are
    populated only when ``analyze()`` was given a ``forward_input``.
    """

    layer_name: str
    layer_type: str
    n_params: int
    plane_basis: torch.Tensor
    activation_coherence: Optional[float] = None
    activation_curvature: Optional[float] = None


@dataclass
class TopologyReport:
    """Cross-layer topology aggregate.

    ``cross_layer_closure_error`` is the mean residual fraction when each
    pairwise commutator ``[B_l, B_m]`` of layer mean bivectors is projected
    back onto the span of layer bivectors. Low = composing layers stays in a
    closed Lie subalgebra; high = composition generates rotation planes.
    """

    layers: List[LayerTopology] = field(default_factory=list)
    cross_layer_closure_error: float = 0.0
    structure_constants_norm: float = 0.0
    coherence_trajectory: List[float] = field(default_factory=list)
    summary_label: str = "unknown"


@dataclass
class TuningRecommendation:
    """General optimizer-agnostic tuning hint derived from the topology report."""

    lr: float = 1e-3
    bivector_init_scale: float = 1.0
    warmup_steps: int = 0
    layers_to_freeze: List[str] = field(default_factory=list)
    suggested_signature_lift: Optional[Tuple[int, int, int]] = None
    notes: List[str] = field(default_factory=list)


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
    topology_report: Optional[TopologyReport] = None
    recommendation: Optional[TuningRecommendation] = None


class PreExplorationAnalyzer:
    """Pre-optimization landscape and architecture analysis pipeline."""

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

    # ---- Architecture-aware analysis ---------------------------------------

    def _walk_rotor_layers(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Find rotor-bearing layers in ``model``.

        Returns list of ``(qualified_name, module)`` for ``RotorLayer``,
        ``MultiRotorLayer``, ``ReflectionLayer``, and ``RotaryBivectorPE``
        instances. Lazy-imports the layer types so this module stays cheap to
        import in non-architecture flows.
        """
        from layers.adapters.embedding import RotaryBivectorPE
        from layers.primitives.multi_rotor import MultiRotorLayer
        from layers.primitives.reflection import ReflectionLayer
        from layers.primitives.rotor import RotorLayer

        targets = (RotorLayer, MultiRotorLayer, ReflectionLayer, RotaryBivectorPE)
        return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, targets)]

    def _scatter_grade(self, weights: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
        """Scatter ``[N, num_grade_elements]`` weights into ``[N, dim]`` MVs."""
        N = weights.shape[0]
        out = torch.zeros(N, dim, device=weights.device, dtype=weights.dtype)
        idx = indices.unsqueeze(0).expand(N, -1)
        out.scatter_(1, idx, weights)
        return out

    def _decompose_layer_planes(self, layer: nn.Module) -> torch.Tensor:
        """Return simple-bivector basis of a layer's grade parameter.

        For grade-2 layers (``RotorLayer``/``MultiRotorLayer``/
        ``RotaryBivectorPE``) this calls :meth:`SpectralAnalyzer.bivector_field_spectrum`
        on the scattered ``[N, 1, dim]`` parameter to extract simple
        components. For ``ReflectionLayer`` it returns the unit grade-1
        vectors directly (no plane decomposition; reflections are vector-
        parameterized).
        """
        from layers.adapters.embedding import RotaryBivectorPE
        from layers.primitives.multi_rotor import MultiRotorLayer
        from layers.primitives.reflection import ReflectionLayer
        from layers.primitives.rotor import RotorLayer

        algebra = self.algebra
        dim = algebra.dim
        device = algebra.device

        if isinstance(layer, RotorLayer):
            mv = self._scatter_grade(layer.grade_weights.detach(), layer.grade_indices, dim)
        elif isinstance(layer, MultiRotorLayer):
            mv = self._scatter_grade(layer.rotor_grade_weights.detach(), layer.grade_indices, dim)
        elif isinstance(layer, RotaryBivectorPE):
            mv = self._scatter_grade(layer.bivector_weights.detach(), layer.bivector_indices, dim)
        elif isinstance(layer, ReflectionLayer):
            return self._scatter_grade(layer.vector_weights.detach(), layer.vector_indices, dim)
        else:
            return torch.zeros(0, dim, device=device)

        if mv.shape[0] == 0 or algebra.n < 2:
            return torch.zeros(0, dim, device=mv.device)

        spectral = SpectralAnalyzer(algebra)
        _sv, components = spectral.bivector_field_spectrum(mv.unsqueeze(1))
        if not components or components[0].abs().sum().item() < 1e-12:
            return torch.zeros(0, dim, device=mv.device)
        return torch.stack(components)

    def _analyze_forward_pass(
        self,
        model: nn.Module,
        sample_input,
        layer_modules: List[Tuple[str, nn.Module]],
    ) -> Dict[str, Tuple[float, float]]:
        """Run one forward pass with hooks; collect per-layer coherence/curvature.

        Returns ``{layer_name: (coherence, curvature)}``. Empty if input or
        algebra is missing, or if the forward pass raises.
        """
        if sample_input is None or self.algebra is None or self.algebra.n < 2:
            return {}

        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name):
            def hook(_module, _inputs, output):
                captured[name] = output.detach() if torch.is_tensor(output) else output

            return hook

        for name, mod in layer_modules:
            handles.append(mod.register_forward_hook(make_hook(name)))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if isinstance(sample_input, (tuple, list)):
                    model(*sample_input)
                elif isinstance(sample_input, dict):
                    model(**sample_input)
                else:
                    model(sample_input)
        except Exception:
            for h in handles:
                h.remove()
            if was_training:
                model.train()
            return {}
        finally:
            for h in handles:
                h.remove()
            if was_training:
                model.train()

        out: Dict[str, Tuple[float, float]] = {}
        dim = self.algebra.dim
        for name, act in captured.items():
            if not torch.is_tensor(act):
                continue
            if act.dim() < 2 or act.shape[-1] != dim:
                continue
            flat = act.reshape(-1, dim)
            if flat.shape[0] < 4:
                continue
            k = min(8, flat.shape[0] - 1)
            if k < 2:
                continue
            try:
                gf = GeodesicFlow(self.algebra, k=k)
                coh = gf.coherence(flat)
                curv = gf.curvature(flat)
                out[name] = (float(coh), float(curv))
            except Exception:
                continue

        return out

    def _cross_layer_closure(self, planes_per_layer: List[torch.Tensor]) -> Tuple[float, float]:
        """Pairwise-commutator closure across layer mean bivectors.

        Sums each layer's simple-component basis into one representative
        bivector, computes pairwise commutators ``[B_l, B_m]``, projects each
        onto the normalized span of layer bivectors, and reports the mean
        residual fraction (closure error) and mean projection magnitude
        (structure-constant norm proxy).
        """
        algebra = self.algebra
        if algebra is None or algebra.n < 2:
            return 0.0, 0.0

        layer_bvs = []
        for planes in planes_per_layer:
            if planes is None or planes.numel() == 0:
                continue
            layer_bvs.append(planes.sum(dim=0))
        if len(layer_bvs) < 2:
            return 0.0, 0.0

        B = torch.stack(layer_bvs)  # [L, dim]
        L = B.shape[0]
        a_idx, b_idx = torch.triu_indices(L, L, offset=1, device=B.device)
        if a_idx.numel() == 0:
            return 0.0, 0.0

        brackets = algebra.commutator(B[a_idx], B[b_idx])
        brackets_bv = algebra.grade_projection(brackets, 2)

        B_norm = F.normalize(B, dim=-1, eps=1e-8)
        coeffs = brackets_bv @ B_norm.T
        projected = coeffs @ B_norm
        residuals = brackets_bv - projected

        bracket_norms = brackets_bv.norm(dim=-1).clamp(min=1e-8)
        residual_norms = residuals.norm(dim=-1)
        valid = brackets_bv.norm(dim=-1) > 1e-8
        if valid.any():
            closure_error = (residual_norms[valid] / bracket_norms[valid]).mean().item()
        else:
            closure_error = 0.0
        structure_norm = coeffs.abs().mean().item()
        return float(closure_error), float(structure_norm)

    def _analyze_architecture(self, model: nn.Module, sample_input=None) -> Optional[TopologyReport]:
        if self.algebra is None:
            return None

        layer_modules = self._walk_rotor_layers(model)
        if not layer_modules:
            return None

        activations = self._analyze_forward_pass(model, sample_input, layer_modules)

        layers: List[LayerTopology] = []
        planes_per_layer: List[torch.Tensor] = []
        for name, mod in layer_modules:
            try:
                planes = self._decompose_layer_planes(mod)
            except Exception:
                planes = torch.zeros(0, self.algebra.dim, device=self.algebra.device)
            planes_per_layer.append(planes)

            n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            coh, curv = activations.get(name, (None, None))
            layers.append(
                LayerTopology(
                    layer_name=name,
                    layer_type=type(mod).__name__,
                    n_params=n_params,
                    plane_basis=planes,
                    activation_coherence=coh,
                    activation_curvature=curv,
                )
            )

        closure_err, struct_norm = self._cross_layer_closure(planes_per_layer)
        traj = [l.activation_coherence for l in layers if l.activation_coherence is not None]
        mean_coh = sum(traj) / len(traj) if traj else 0.0
        max_drop = 0.0
        for a, b in zip(traj, traj[1:]):
            max_drop = max(max_drop, a - b)

        if mean_coh < 0.3 or max_drop > 0.4:
            label = "fragmented"
        elif mean_coh > 0.6 and closure_err < 0.1:
            label = "aligned-stack"
        else:
            label = "mixed-coupling"

        return TopologyReport(
            layers=layers,
            cross_layer_closure_error=closure_err,
            structure_constants_norm=struct_norm,
            coherence_trajectory=traj,
            summary_label=label,
        )

    def _recommend_tuning(
        self,
        result: PreExplorationResult,
        topology: Optional[TopologyReport],
    ) -> TuningRecommendation:
        rec = TuningRecommendation()

        if topology is None:
            return rec

        if topology.summary_label == "aligned-stack":
            rec.lr = 2e-3
            rec.bivector_init_scale = 1.5
        elif topology.summary_label == "fragmented":
            rec.lr = 5e-4
            rec.bivector_init_scale = 0.5
            rec.warmup_steps = 100
        elif topology.cross_layer_closure_error > 0.5:
            rec.lr = 5e-4
            rec.notes.append(
                f"High cross-layer non-commutativity (closure_error="
                f"{topology.cross_layer_closure_error:.2f}); halving lr."
            )

        for layer in topology.layers:
            if layer.activation_curvature is not None and layer.activation_curvature > 0.5:
                rec.bivector_init_scale = min(rec.bivector_init_scale, 0.3)
                rec.notes.append(
                    f"Layer {layer.layer_name} has activation_curvature="
                    f"{layer.activation_curvature:.2f}; consider regularising bivector norm."
                )
            if layer.activation_coherence is not None and layer.activation_coherence > 0.85 and layer.layer_name:
                rec.layers_to_freeze.append(layer.layer_name)

        if result.lifting_report:
            best = result.lifting_report.get("best")
            if best and best != "original":
                sig = result.lifting_report.get(best, {}).get("signature")
                if isinstance(sig, tuple) and len(sig) == 2:
                    rec.suggested_signature_lift = (sig[0], sig[1], 0)
                elif isinstance(sig, tuple) and len(sig) == 3:
                    rec.suggested_signature_lift = sig

        return rec

    # ---- Top-level entry point --------------------------------------------

    def analyze(
        self,
        model: nn.Module,
        loss_fn: Callable,
        forward_input=None,
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

        try:
            result.topology_report = self._analyze_architecture(model, forward_input)
        except Exception:
            result.topology_report = None
        result.recommendation = self._recommend_tuning(result, result.topology_report)

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

        if result.recommendation is not None:
            rec = result.recommendation
            if rec.lr != TuningRecommendation().lr:
                cfg.lr = rec.lr

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
