"""Optimizer factory, training loops, comparison runner."""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from core.algebra import CliffordAlgebra
from layers import RotorLayer, MultiRotorLayer
from optimizers.riemannian import RiemannianAdam, ExponentialSGD

from .config import ExperimentConfig, ExperimentResult, GDOConfig
from .controller import GDOController
from .optimizer import GDOOptimizer
from .pre_exploration import PreExplorationAnalyzer


def _collect_bivector_norms(model: nn.Module) -> float:
    """Sum of bivector parameter norms across all rotor layers."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, (RotorLayer, MultiRotorLayer)):
            w = m.grade_weights if isinstance(m, RotorLayer) else m.rotor_grade_weights
            total += w.detach().norm().item()
    return total


def create_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    algebra: Optional[CliffordAlgebra] = None,
    loss_fn: Optional[Callable] = None,
    config: Optional[GDOConfig] = None,
    device: str = 'cpu',
) -> Tuple[Union[Optimizer, GDOController], str]:
    """Factory for all optimizer variants."""
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr), "Adam"
    elif name == 'riemannian_adam':
        if algebra is None:
            return torch.optim.Adam(model.parameters(), lr=lr), "Adam (no algebra)"
        return RiemannianAdam.from_model(model, lr=lr, algebra=algebra), "RiemannianAdam"
    elif name == 'exponential_sgd':
        if algebra is None:
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9), "SGD"
        return ExponentialSGD.from_model(model, lr=lr, algebra=algebra), "ExponentialSGD"
    elif name == 'gdo':
        assert loss_fn is not None, "GDO requires loss_fn"
        gdo_opt = (
            GDOOptimizer.from_model(model, lr=lr, algebra=algebra)
            if algebra
            else GDOOptimizer(model.parameters(), lr=lr)
        )
        controller = GDOController(
            model,
            loss_fn,
            optimizer=gdo_opt,
            config=config,
            algebra=algebra,
            device=device,
            lr=lr,
        )
        return controller, "GDO"
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_loop_standard(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    steps: int,
    metric_fn: Optional[Callable] = None,
    log_interval: int = 500,
    label: str = "",
) -> ExperimentResult:
    """Standard training loop for torch.optim.Optimizer (Adam, RiemannianAdam)."""
    losses = []
    wall_times = []
    metrics: Dict[str, List[float]] = {}
    bv_norms = []

    for s in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        wt = time.perf_counter() - t0

        losses.append(loss.item())
        wall_times.append(wt)
        bv_norms.append(_collect_bivector_norms(model))

        if metric_fn is not None:
            for k, v in metric_fn().items():
                metrics.setdefault(k, []).append(v)

        if s % log_interval == 0 or s == steps - 1:
            print(f"  [{label}] Step {s:5d}: loss={loss.item():.6f}")

    return ExperimentResult(
        name="",
        optimizer_name=label,
        losses=losses,
        wall_times=wall_times,
        metrics=metrics,
        final_loss=losses[-1] if losses else float('inf'),
        total_wall_time=sum(wall_times),
        bivector_norms=bv_norms,
    )


def train_loop_gdo(
    model: nn.Module,
    controller: GDOController,
    steps: int,
    metric_fn: Optional[Callable] = None,
    log_interval: int = 500,
) -> ExperimentResult:
    """GDO controller loop with full diagnostic collection."""
    losses = []
    wall_times = []
    metrics: Dict[str, List[float]] = {}
    bv_norms = []

    for s in range(steps):
        t0 = time.perf_counter()
        loss = controller.loss_fn()
        info = controller.optimize_step(loss)
        wt = time.perf_counter() - t0

        losses.append(info['loss'])
        wall_times.append(wt)
        bv_norms.append(_collect_bivector_norms(model))

        if metric_fn is not None:
            for k, v in metric_fn().items():
                metrics.setdefault(k, []).append(v)

        if s % log_interval == 0 or s == steps - 1:
            print(f"  [GDO] Step {s:5d}: loss={info['loss']:.6f}  mode={info['mode']}")

    return ExperimentResult(
        name="",
        optimizer_name="GDO",
        losses=losses,
        wall_times=wall_times,
        metrics=metrics,
        final_loss=losses[-1] if losses else float('inf'),
        total_wall_time=sum(wall_times),
        gdo_diagnostics=controller.get_full_diagnostics(),
        bivector_norms=bv_norms,
        mode_history=controller.get_mode_history(),
    )


def run_comparison(
    task_name: str,
    model_factory: Callable,
    loss_factory: Callable,
    config: ExperimentConfig,
    optimizers: Tuple[str, ...] = ('gdo', 'riemannian_adam', 'adam'),
    metric_factory: Optional[Callable] = None,
    pre_explore: bool = True,
    output_dir: str = "gdo_plots",
) -> Dict[str, ExperimentResult]:
    """Run all optimizers on same task, same init, collect results."""
    results: Dict[str, ExperimentResult] = {}

    torch.manual_seed(config.seed)
    ref_model = model_factory()
    init_state = {k: v.clone() for k, v in ref_model.state_dict().items()}
    algebra = getattr(ref_model, 'algebra', None)
    del ref_model

    for opt_name in optimizers:
        print(f"\n  --- {opt_name.upper()} ---")
        torch.manual_seed(config.seed)
        model = model_factory()
        model.load_state_dict(init_state)
        loss_fn = loss_factory(model)
        metric_fn = metric_factory(model) if metric_factory else None

        if opt_name == 'gdo':
            if pre_explore and algebra is not None:
                try:
                    pre_analyzer = PreExplorationAnalyzer(algebra=algebra, n_samples=100, device=config.device)
                    pre_result = pre_analyzer.analyze(model, loss_fn)
                    print(f"  Strategy: {pre_result.strategy_label}")
                    gdo_config = pre_result.recommended_config
                except Exception:
                    gdo_config = GDOConfig(lr=config.lr)
            else:
                gdo_config = config.gdo_config or GDOConfig(lr=config.lr)

            controller_or_opt, label = create_optimizer(
                'gdo', model, config.lr, algebra=algebra, loss_fn=loss_fn, config=gdo_config, device=config.device
            )
            result = train_loop_gdo(
                model, controller_or_opt, config.steps, metric_fn=metric_fn, log_interval=max(config.steps // 5, 1)
            )
        else:
            opt, label = create_optimizer(opt_name, model, config.lr, algebra=algebra, device=config.device)
            result = train_loop_standard(
                model,
                opt,
                loss_fn,
                config.steps,
                metric_fn=metric_fn,
                log_interval=max(config.steps // 5, 1),
                label=label,
            )

        result.name = task_name
        result.optimizer_name = label
        results[label] = result

    return results


def _new_history() -> Dict:
    return {
        "losses": [],
        "modes": [],
        "probe_steps": [],
        "curvatures": [],
        "grad_norms": [],
        "betas": [],
        "lifts": [],
        "plateaus": [],
        "trajectory": [],
        "angle_errors": [],
    }


def _collect_history(controller, info, history):
    history["losses"].append(info["loss"])
    history["modes"].append(info["mode"])
    if "probe" in info:
        history["probe_steps"].append(info["step"])
        history["curvatures"].append(info["probe"]["mean_curvature"])
        history["grad_norms"].append(info["probe"]["grad_norm"])
        history["betas"].append(info["probe"]["beta"])
    if "lift_oracle" in info:
        history["lifts"].append(
            {
                "step": info["step"],
                "loss": info["loss"],
                "success": "improved" in str(info["lift_oracle"]),
                "sigma": getattr(controller.lift_oracle, '_current_sigma', 0),
            }
        )
