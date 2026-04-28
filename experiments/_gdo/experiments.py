"""Registered experiment runners and category/all helpers.

Three registrations matching the curated benchmark suite:

- ``gbn_small``    -> ``SmallGBNModel`` (primitives showcase).
- ``multi_rotor``  -> ``MultiRotorRegistrationModel`` (rotor-bank showcase).
- ``transformer_toy`` -> ``GeometricTransformerToyModel`` (blocks/transformer showcase).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments._lib import build_visualization_metadata, signature_metadata

from .benchmarks import (
    GeometricTransformerToyModel,
    MultiRotorRegistrationModel,
    SmallGBNModel,
)
from .config import EXPERIMENT_REGISTRY, ExperimentConfig, register_experiment
from .harness import run_comparison
from .plotting import (
    plot_bivector_trajectory,
    plot_optimizer_state_dashboard,
    plot_three_way_comparison,
)


def _gdo_metadata(
    task: str,
    *,
    seed: int,
    algebra_sig: tuple[int, int] | None = None,
) -> str:
    parts = []
    if algebra_sig is not None:
        parts.append(signature_metadata(*algebra_sig))
    return build_visualization_metadata(*parts, task=task, seed=seed)


@register_experiment("gbn_small", "ga_neural")
def run_gbn_small(
    steps: int = 200,
    optimizers=("gdo", "riemannian_adam", "adam"),
    seed: int = 42,
    output_dir: str = "gdo_plots",
    device: str = "cpu",
    cli_args=None,
):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Small GBN (Cl(3,0), 4ch)")
    print("=" * 60)

    config = ExperimentConfig(
        name="gbn_small", category="ga_neural", steps=steps, lr=5e-4, seed=seed, device=device, algebra_sig=(3, 0)
    )

    def model_factory():
        return SmallGBNModel(p=3, q=0, channels=4, device=device)

    def loss_factory(model):
        X = torch.randn(32, 4, model._dim, device=device) * 0.3
        y_target = X[:, :, 0].mean(dim=1, keepdim=True)

        def loss_fn():
            out = model(X)
            pred = out[:, :, 0].mean(dim=1, keepdim=True)
            return F.mse_loss(pred, y_target)

        return loss_fn

    results = run_comparison(
        "gbn_small",
        model_factory=model_factory,
        loss_factory=loss_factory,
        config=config,
        optimizers=optimizers,
        output_dir=output_dir,
    )
    metadata = _gdo_metadata("gbn_small", seed=seed, algebra_sig=(3, 0))
    plot_three_way_comparison(
        results,
        title="Small GBN",
        output_dir=output_dir,
        metadata=metadata,
        args=cli_args,
    )
    plot_bivector_trajectory(
        results,
        title="Small GBN",
        output_dir=output_dir,
        metadata=metadata,
        args=cli_args,
    )

    gdo_res = results.get("GDO")
    if gdo_res:
        plot_optimizer_state_dashboard(
            gdo_res,
            title="Small GBN GDO",
            output_dir=output_dir,
            metadata=metadata,
            args=cli_args,
        )
    return results


@register_experiment("multi_rotor", "geometric")
def run_multi_rotor(
    steps: int = 2000,
    optimizers=("gdo", "riemannian_adam", "adam"),
    seed: int = 42,
    output_dir: str = "gdo_plots",
    device: str = "cpu",
    cli_args=None,
):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Multi-Rotor Registration (Cl(3,0), 3 clusters)")
    print("=" * 60)

    config = ExperimentConfig(
        name="multi_rotor", category="geometric", steps=steps, lr=1e-3, seed=seed, device=device, algebra_sig=(3, 0)
    )
    results = run_comparison(
        "multi_rotor",
        model_factory=lambda: MultiRotorRegistrationModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config,
        optimizers=optimizers,
        output_dir=output_dir,
    )
    metadata = _gdo_metadata("multi_rotor", seed=seed, algebra_sig=(3, 0))
    plot_three_way_comparison(
        results,
        title="Multi-Rotor",
        output_dir=output_dir,
        metadata=metadata,
        args=cli_args,
    )
    plot_bivector_trajectory(
        results,
        title="Multi-Rotor",
        output_dir=output_dir,
        metadata=metadata,
        args=cli_args,
    )
    return results


@register_experiment("transformer_toy", "ga_neural")
def run_transformer_toy(
    steps: int = 200,
    optimizers=("gdo", "riemannian_adam", "adam"),
    seed: int = 42,
    output_dir: str = "gdo_plots",
    device: str = "cpu",
    cli_args=None,
):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Geometric Transformer Toy (Cl(3,0), 1 block)")
    print("=" * 60)

    config = ExperimentConfig(
        name="transformer_toy", category="ga_neural", steps=steps, lr=5e-4, seed=seed, device=device, algebra_sig=(3, 0)
    )
    results = run_comparison(
        "transformer_toy",
        model_factory=lambda: GeometricTransformerToyModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config,
        optimizers=optimizers,
        output_dir=output_dir,
    )
    metadata = _gdo_metadata("transformer_toy", seed=seed, algebra_sig=(3, 0))
    plot_three_way_comparison(
        results,
        title="Geometric Transformer Toy",
        output_dir=output_dir,
        metadata=metadata,
        args=cli_args,
    )
    plot_bivector_trajectory(
        results,
        title="Geometric Transformer Toy",
        output_dir=output_dir,
        metadata=metadata,
        args=cli_args,
    )
    return results


def run_category(category: str, **kwargs):
    """Run all experiments in a category."""
    results = {}
    for name, (fn, cat) in EXPERIMENT_REGISTRY.items():
        if cat == category:
            results[name] = fn(**kwargs)
    return results


def run_all_experiments(**kwargs):
    """Run all registered experiments and produce analysis report."""
    all_results = {}
    for name, (fn, _cat) in EXPERIMENT_REGISTRY.items():
        try:
            all_results[name] = fn(**kwargs)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    return all_results
