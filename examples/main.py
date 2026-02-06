# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want
# the industry to build upon this "unbending" paradigm.

"""Versor Examples Entry Point. Synthetic/demo tasks.

Run from the project root:
    uv run python -m examples.main task=manifold
"""

import sys
import os

# Ensure project root is on path so core/layers/functional imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig
from examples.tasks.manifold import ManifoldTask
from examples.tasks.hyperbolic import HyperbolicTask
from examples.tasks.sanity_check import SanityCheckTask

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Dispatches example/synthetic tasks."""
    task_name = cfg.name

    task_map = {
        'manifold': ManifoldTask,
        'hyperbolic': HyperbolicTask,
        'sanity': SanityCheckTask,
    }

    if task_name not in task_map:
        raise ValueError(f"Unknown example task: {task_name}. Available: {list(task_map.keys())}")

    TaskClass = task_map[task_name]
    task = TaskClass(cfg)
    task.run()

if __name__ == "__main__":
    main()
