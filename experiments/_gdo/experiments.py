"""Registered experiment runners and category/all runners."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .benchmarks import (
    AckleyModel,
    ConformalRegistrationModel,
    DeepGBNModel,
    MediumGBNModel,
    MinkowskiRotorModel,
    MultiRotorRegistrationModel,
    MultiSigGBNModel,
    RastriginModel,
    RosenbrockModel,
    RotorRegistrationModel,
    SO3InterpolationModel,
    SmallGBNModel,
    StyblinskiTangModel,
)
from .config import EXPERIMENT_REGISTRY, ExperimentConfig, register_experiment
from .harness import run_comparison
from .plotting import (
    plot_bivector_trajectory,
    plot_convergence_rate,
    plot_optimizer_state_dashboard,
    plot_three_way_comparison,
    plot_timing_breakdown,
)


@register_experiment("rosenbrock", "analytic")
def run_rosenbrock(steps: int = 2000, optimizers=('gdo', 'riemannian_adam', 'adam'),
                   seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Rosenbrock Function (a=1, b=100)")
    print("=" * 60)

    config = ExperimentConfig(name="rosenbrock", category="analytic",
                              steps=steps, lr=1e-3, seed=seed, device=device)
    results = run_comparison(
        "rosenbrock",
        model_factory=RosenbrockModel,
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Rosenbrock", output_dir=output_dir)
    plot_convergence_rate(results, title="Rosenbrock", output_dir=output_dir)
    plot_timing_breakdown(results, title="Rosenbrock", output_dir=output_dir)
    return results


@register_experiment("rastrigin", "analytic")
def run_rastrigin(n_dims: int = 8, steps: int = 3000,
                  optimizers=('gdo', 'riemannian_adam', 'adam'),
                  seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Rastrigin Function ({n_dims}D)")
    print("=" * 60)

    config = ExperimentConfig(name="rastrigin", category="analytic",
                              steps=steps, lr=1e-2, seed=seed, device=device)
    results = run_comparison(
        "rastrigin",
        model_factory=lambda: RastriginModel(n_dims=n_dims),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title=f"Rastrigin {n_dims}D", output_dir=output_dir)
    plot_convergence_rate(results, title=f"Rastrigin {n_dims}D", output_dir=output_dir)
    return results


@register_experiment("ackley", "analytic")
def run_ackley(n_dims: int = 10, steps: int = 3000,
               optimizers=('gdo', 'riemannian_adam', 'adam'),
               seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Ackley Function ({n_dims}D)")
    print("=" * 60)

    config = ExperimentConfig(name="ackley", category="analytic",
                              steps=steps, lr=1e-2, seed=seed, device=device)
    results = run_comparison(
        "ackley",
        model_factory=lambda: AckleyModel(n_dims=n_dims),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title=f"Ackley {n_dims}D", output_dir=output_dir)
    plot_convergence_rate(results, title=f"Ackley {n_dims}D", output_dir=output_dir)
    return results


@register_experiment("styblinski_tang", "analytic")
def run_styblinski_tang(n_dims: int = 6, steps: int = 2000,
                        optimizers=('gdo', 'riemannian_adam', 'adam'),
                        seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Styblinski-Tang Function ({n_dims}D)")
    print("=" * 60)

    config = ExperimentConfig(name="styblinski_tang", category="analytic",
                              steps=steps, lr=5e-3, seed=seed, device=device)
    results = run_comparison(
        "styblinski_tang",
        model_factory=lambda: StyblinskiTangModel(n_dims=n_dims),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title=f"Styblinski-Tang {n_dims}D", output_dir=output_dir)
    return results


@register_experiment("registration", "geometric")
def run_registration(steps: int = 1500, noise_std: float = 0.05, rotation_angle: float = 2.5,
                     optimizers=('gdo', 'riemannian_adam', 'adam'),
                     seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: Rotor Registration (Cl(3,0), angle={rotation_angle:.2f} rad)")
    print("=" * 60)

    config = ExperimentConfig(name="registration", category="geometric",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "registration",
        model_factory=lambda: RotorRegistrationModel(
            noise_std=noise_std, rotation_angle=rotation_angle, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers,
        metric_factory=lambda m: lambda: {"angle_error": m.angular_error()},
        output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Rotor Registration", output_dir=output_dir)
    plot_convergence_rate(results, title="Rotor Registration", output_dir=output_dir)
    plot_timing_breakdown(results, title="Rotor Registration", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Rotor Registration", output_dir=output_dir)
    return results


@register_experiment("minkowski_rotor", "geometric")
def run_minkowski_rotor(steps: int = 1500,
                        optimizers=('gdo', 'riemannian_adam', 'adam'),
                        seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Minkowski Rotor Registration (Cl(2,1))")
    print("=" * 60)

    config = ExperimentConfig(name="minkowski_rotor", category="geometric",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(2, 1))
    results = run_comparison(
        "minkowski_rotor",
        model_factory=lambda: MinkowskiRotorModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers,
        metric_factory=lambda m: lambda: {"rapidity_error": m.rapidity_error()},
        output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Minkowski Rotor Cl(2,1)", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Minkowski Rotor", output_dir=output_dir)
    return results


@register_experiment("conformal_registration", "geometric")
def run_conformal_registration(steps: int = 2000,
                               optimizers=('gdo', 'riemannian_adam', 'adam'),
                               seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Conformal Registration (Cl(4,1), 32D)")
    print("=" * 60)

    config = ExperimentConfig(name="conformal_registration", category="geometric",
                              steps=steps, lr=5e-4, seed=seed, device=device,
                              algebra_sig=(4, 1))
    results = run_comparison(
        "conformal_registration",
        model_factory=lambda: ConformalRegistrationModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Conformal Cl(4,1)", output_dir=output_dir)
    return results


@register_experiment("multi_rotor", "geometric")
def run_multi_rotor(steps: int = 2000,
                    optimizers=('gdo', 'riemannian_adam', 'adam'),
                    seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Multi-Rotor Registration (Cl(3,0), 3 clusters)")
    print("=" * 60)

    config = ExperimentConfig(name="multi_rotor", category="geometric",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "multi_rotor",
        model_factory=lambda: MultiRotorRegistrationModel(device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Multi-Rotor", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Multi-Rotor", output_dir=output_dir)
    return results


@register_experiment("gbn_small", "ga_neural")
def run_gbn_small(steps: int = 200,
                  optimizers=('gdo', 'riemannian_adam', 'adam'),
                  seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Small GBN (Cl(3,0), 4ch)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_small", category="ga_neural",
                              steps=steps, lr=5e-4, seed=seed, device=device,
                              algebra_sig=(3, 0))

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
        "gbn_small", model_factory=model_factory, loss_factory=loss_factory,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Small GBN", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Small GBN", output_dir=output_dir)

    gdo_res = results.get("GDO")
    if gdo_res:
        plot_optimizer_state_dashboard(gdo_res, title="Small GBN GDO", output_dir=output_dir)
    return results


@register_experiment("gbn_medium", "ga_neural")
def run_gbn_medium(steps: int = 300,
                   optimizers=('gdo', 'riemannian_adam', 'adam'),
                   seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Medium GBN (Cl(3,0), 16ch, 3 layers)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_medium", category="ga_neural",
                              steps=steps, lr=3e-4, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "gbn_medium",
        model_factory=lambda: MediumGBNModel(channels=16, layers=3, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Medium GBN", output_dir=output_dir)
    plot_convergence_rate(results, title="Medium GBN", output_dir=output_dir)
    plot_timing_breakdown(results, title="Medium GBN", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Medium GBN", output_dir=output_dir)
    return results


@register_experiment("gbn_multisig", "ga_neural")
def run_gbn_multisig(steps: int = 250,
                     optimizers=('gdo', 'riemannian_adam', 'adam'),
                     seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Minkowski GBN (Cl(2,1), 8ch)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_multisig", category="ga_neural",
                              steps=steps, lr=3e-4, seed=seed, device=device,
                              algebra_sig=(2, 1))
    results = run_comparison(
        "gbn_multisig",
        model_factory=lambda: MultiSigGBNModel(channels=8, layers=2, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Minkowski GBN Cl(2,1)", output_dir=output_dir)
    plot_bivector_trajectory(results, title="Minkowski GBN", output_dir=output_dir)
    return results


@register_experiment("gbn_deep", "ga_neural")
def run_gbn_deep(steps: int = 300,
                 optimizers=('gdo', 'riemannian_adam', 'adam'),
                 seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: Deep GBN (Cl(3,0), 16ch, 5 layers)")
    print("=" * 60)

    config = ExperimentConfig(name="gbn_deep", category="ga_neural",
                              steps=steps, lr=2e-4, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "gbn_deep",
        model_factory=lambda: DeepGBNModel(channels=32, layers=5, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers, output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="Deep GBN", output_dir=output_dir)
    plot_convergence_rate(results, title="Deep GBN", output_dir=output_dir)
    plot_timing_breakdown(results, title="Deep GBN", output_dir=output_dir)
    return results


@register_experiment("so3_interpolation", "manifold")
def run_so3_interpolation(steps: int = 1500,
                          optimizers=('gdo', 'riemannian_adam', 'adam'),
                          seed: int = 42, output_dir: str = "gdo_plots", device: str = 'cpu'):
    print("\n" + "=" * 60)
    print("EXPERIMENT: SO(3) Rotor Interpolation")
    print("=" * 60)

    config = ExperimentConfig(name="so3_interpolation", category="manifold",
                              steps=steps, lr=1e-3, seed=seed, device=device,
                              algebra_sig=(3, 0))
    results = run_comparison(
        "so3_interpolation",
        model_factory=lambda: SO3InterpolationModel(n_waypoints=8, device=device),
        loss_factory=lambda m: m.forward,
        config=config, optimizers=optimizers,
        metric_factory=lambda m: lambda: {"geodesic_deviation": m.geodesic_deviation()},
        output_dir=output_dir,
    )
    plot_three_way_comparison(results, title="SO(3) Interpolation", output_dir=output_dir)
    plot_convergence_rate(results, title="SO(3) Interpolation", output_dir=output_dir)
    plot_bivector_trajectory(results, title="SO(3) Interpolation", output_dir=output_dir)
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
