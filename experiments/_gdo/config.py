"""GDO config dataclasses and the experiment registry decorator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class GDOConfig:
    """Centralized configuration for the Geometric Deterministic Optimizer."""
    lr: float = 1e-3
    probe_interval: int = 50
    topology_interval: int = 200
    sprint_after: int = 500
    max_navigate_steps: int = 150
    lift_patience: int = 80
    lift_sigma: float = 0.05
    lift_k: int = 6
    lorentz_max_beta: float = 0.95
    commutator_threshold: float = 0.3
    # Geometric controller params
    fim_damping: float = 1e-4
    closure_trust_threshold: float = 0.1
    coherence_gate: float = 0.3
    entropy_exploration_threshold: float = 0.7
    # Parameter grouping
    grouping_strategy: str = "geometric"  # "geometric" or "module"
    min_group_size: int = 4
    max_groups: int = 16
    # Coloring algorithm
    dsatur_enabled: bool = True
    color_conflict_budget: float = 0.5
    manifold_compat_constraint: bool = True
    # Interaction estimation
    interaction_estimation: str = "efficient"  # "efficient", "fd", "gradient_only"
    grad_cosine_threshold: float = 0.1
    # Adaptive rescheduling
    adaptive_reschedule: bool = True
    reschedule_interval: int = 50
    reschedule_loss_delta: float = 0.2
    reschedule_grad_kl_threshold: float = 0.5


@dataclass
class ExperimentResult:
    """Collected results from one optimizer run."""
    name: str
    optimizer_name: str
    losses: List[float]
    wall_times: List[float]
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    final_loss: float = 0.0
    total_wall_time: float = 0.0
    gdo_diagnostics: Optional[Dict] = None
    bivector_norms: Optional[List[float]] = None
    mode_history: Optional[List[str]] = None


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    category: str
    steps: int
    lr: float
    seed: int = 42
    device: str = 'cpu'
    algebra_sig: Optional[Tuple[int, int]] = None
    gdo_config: Optional[GDOConfig] = None


EXPERIMENT_REGISTRY: Dict[str, Tuple[Callable, str]] = {}


def register_experiment(name: str, category: str):
    """Decorator for registering experiment functions."""
    def decorator(fn):
        EXPERIMENT_REGISTRY[name] = (fn, category)
        return fn
    return decorator
