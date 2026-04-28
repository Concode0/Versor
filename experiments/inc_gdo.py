# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#


"""
==============================================================================
VERSOR EXPERIMENT: IDEA INCUBATOR (SPIN-OFF CONCEPT)
==============================================================================

This script serves as an early-stage proof-of-concept for radical, non-Euclidean
architectures. The concepts demonstrated here are strongly driven by geometric
intuition and may currently reside ahead of established academic literature.

Please understand that rigorous mathematical proofs or comprehensive citations
might be incomplete at this stage. If this geometric hypothesis proves
structurally sound, it is planned to be spun off into a dedicated, independent
repository for detailed research.

==============================================================================

Geometric Deterministic Optimizer (GDO) Entry Point. ( See _gdo/ )

Hypothesis
  Geometric deterministic updates should remain competitive across analytic,
  geometric, GA-neural, and manifold objectives when compared against the
  baselines wired into ``experiments._gdo``. This file is the thin CLI entry
  point over that subpackage's optimizer core, controller, benchmarks,
  plotting, and analysis infrastructure.

Execute Command
  uv run python -m experiments.inc_gdo
  uv run python -m experiments.inc_gdo --task all
  uv run python -m experiments.inc_gdo --task rosenbrock --steps 500
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from experiments._gdo import EXPERIMENT_REGISTRY
from experiments._gdo.analysis import analyze_experiment_results
from experiments._gdo.experiments import run_all_experiments, run_category
from experiments._lib import make_experiment_parser, set_seed


def parse_args():
    registered = list(EXPERIMENT_REGISTRY.keys())
    all_choices = registered + ["all", "geometric", "ga_neural", "compare_all"]
    p = make_experiment_parser(
        "Geometric Deterministic Optimizer (GDO) Experiment Suite",
        include=("seed", "device", "output_dir"),
        defaults={"output_dir": "gdo_plots"},
    )
    p.add_argument("--task", choices=all_choices, default="gbn_small")
    p.add_argument("--optimizers", nargs="+", default=["gdo", "riemannian_adam", "adam"], help="Optimizers to compare")
    p.add_argument("--steps", type=int, default=200)
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
    common = dict(
        optimizers=opts,
        seed=args.seed,
        output_dir=od,
        device=args.device,
        cli_args=args,
    )

    if args.task in EXPERIMENT_REGISTRY:
        fn, _cat = EXPERIMENT_REGISTRY[args.task]
        fn(steps=args.steps, **common)
    elif args.task == "geometric":
        run_category("geometric", steps=args.steps, **common)
    elif args.task == "ga_neural":
        run_category("ga_neural", steps=args.steps, **common)
    elif args.task in ("all", "compare_all"):
        all_results = run_all_experiments(steps=min(args.steps, 1000), **common)
        report = analyze_experiment_results(all_results, output_dir=od)
        print("\n" + report)


if __name__ == "__main__":
    main()
