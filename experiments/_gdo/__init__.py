"""Geometric Deterministic Optimizer (GDO) subpackage.

Split from experiments/inc_gdo.py into class-boundary modules. Behaviour-preserving
refactor -- no logic changes versus the monolith.

Module layout:
    config              -- GDOConfig, ExperimentConfig, ExperimentResult, registry.
    topology            -- CriticalPoint, LandscapeMap, LandscapeTopologySearch.
    probes              -- CurvatureProbe, GeodesicIntegrator, LorentzWarpOptimizer.
    parameter_groups    -- GeometricParameterController (FIM, commutator, grouping).
    dimensional_lift    -- DimensionalLiftOracle (lift -> oracle search -> pull-down).
    pre_exploration     -- PreExplorationAnalyzer + result container.
    optimizer           -- GDOOptimizer (torch.optim.Optimizer subclass).
    controller          -- GDOController (top-level coordinator, EXPLORE/NAVIGATE/SPRINT).
    benchmarks          -- 13 analytic + geometric + GBN benchmark models.
    harness             -- create_optimizer factory, train_loop_*, run_comparison.
    plotting            -- All GDO dashboards and comparison plots.
    analysis            -- Convergence metrics, overhead ratios, cross-exp report.
    experiments         -- @register_experiment-decorated runners + run_category/all.

Future work (design sketch -- NOT yet wired up):

    class GDOSuite:
        '''Single entry point the optimizer can instantiate.
           Owns: GDOConfig, PreExplorationAnalyzer, GDOController.
           Lifecycle: analyze() -> warm_start() -> step() -> diagnostics().
           Lets external training loops call .step(loss) without touching
           sub-components. Implementation deferred to next PR.'''

    The facade will collapse the current 3-step user dance (analyze -> create
    controller -> step loop) into a single .step(loss) method, while still
    exposing get_full_diagnostics() for downstream plotting.
"""

from . import experiments as _experiments  # noqa: F401 -- populates EXPERIMENT_REGISTRY
from .config import (
    EXPERIMENT_REGISTRY,
    ExperimentConfig,
    ExperimentResult,
    GDOConfig,
    register_experiment,
)
from .controller import GDOController, GeometricDeterministicOptimizer
from .optimizer import GDOOptimizer
from .pre_exploration import (
    LayerTopology,
    PreExplorationAnalyzer,
    PreExplorationResult,
    TopologyReport,
    TuningRecommendation,
)

__all__ = [
    "EXPERIMENT_REGISTRY",
    "ExperimentConfig",
    "ExperimentResult",
    "GDOConfig",
    "GDOController",
    "GDOOptimizer",
    "GeometricDeterministicOptimizer",
    "LayerTopology",
    "PreExplorationAnalyzer",
    "PreExplorationResult",
    "TopologyReport",
    "TuningRecommendation",
    "register_experiment",
]
