# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#


"""
Highly Experimental, See experiments/gdo/README.md.

Geometric Deterministic Optimizer (GDO) entry point.

Thin CLI wrapper over `experiments._gdo`. The actual optimizer core, controller,
benchmarks, plotting, and analysis infrastructure live in the subpackage.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments._lib import make_experiment_parser, set_seed
from experiments._gdo import EXPERIMENT_REGISTRY
from experiments._gdo.analysis import analyze_experiment_results
from experiments._gdo.experiments import run_all_experiments, run_category


def parse_args():
    registered = list(EXPERIMENT_REGISTRY.keys())
    all_choices = registered + [
        "all", "analytic", "geometric", "ga_neural", "manifold", "compare_all",
    ]
    p = make_experiment_parser(
        "Geometric Deterministic Optimizer (GDO) Experiment Suite",
        include=('seed', 'device', 'output_dir'),
        defaults={'output_dir': 'gdo_plots'},
    )
    p.add_argument("--task", choices=all_choices, default="rosenbrock")
    p.add_argument("--optimizers", nargs="+",
                   default=["gdo", "riemannian_adam", "adam"],
                   help="Optimizers to compare")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--n-dims", type=int, default=10)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--rotation-angle", type=float, default=2.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Geometric Deterministic Optimizer (GDO) Experiment Suite")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Optimizers: {args.optimizers}")
    print(f"Steps: {args.steps}")
    print(f"Device: {args.device}")
    print("=" * 60)

    od = args.output_dir
    opts = tuple(args.optimizers)
    common = dict(optimizers=opts, seed=args.seed, output_dir=od, device=args.device)

    if args.task in EXPERIMENT_REGISTRY:
        fn, _cat = EXPERIMENT_REGISTRY[args.task]
        kwargs = dict(steps=args.steps, **common)
        if args.task in ("rastrigin", "ackley", "styblinski_tang"):
            kwargs["n_dims"] = args.n_dims
        if args.task == "registration":
            kwargs["noise_std"] = args.noise_std
            kwargs["rotation_angle"] = args.rotation_angle
        fn(**kwargs)

    elif args.task == "analytic":
        run_category("analytic", steps=args.steps, n_dims=args.n_dims, **common)
    elif args.task == "geometric":
        run_category("geometric", steps=args.steps, **common)
    elif args.task == "ga_neural":
        run_category("ga_neural", steps=args.steps, **common)
    elif args.task == "manifold":
        run_category("manifold", steps=args.steps, **common)
    elif args.task in ("all", "compare_all"):
        all_results = run_all_experiments(steps=min(args.steps, 1000), **common)
        report = analyze_experiment_results(all_results, output_dir=od)
        print("\n" + report)


if __name__ == "__main__":
    main()
