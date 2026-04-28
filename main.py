# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Versor CLI entry point.

Dispatches geometric learning tasks via Hydra configuration.
"""

import hydra
from omegaconf import DictConfig

EXAMPLE_TASKS = {"manifold", "hyperbolic", "sanity"}

_TASK_MODULES = {
    "md17": ("tasks.md17", "MD17Task"),
    "sr": ("tasks.symbolic_regression", "SRTask"),
    "lqa": ("tasks.lqa", "LQATask"),
    "deap_eeg": ("tasks.deap_eeg", "DEAPEEGTask"),
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Entry point for task execution. Delegates to specific task handlers.

    Args:
        cfg (DictConfig): Hydra configuration dictionary.
    """
    task_name = cfg.name

    if task_name in EXAMPLE_TASKS:
        raise ValueError(
            f"'{task_name}' is a synthetic example task. Run it via: uv run python -m examples.main task={task_name}"
        )

    if task_name not in _TASK_MODULES:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(_TASK_MODULES.keys())}")

    module_path, class_name = _TASK_MODULES[task_name]
    import importlib

    TaskClass = getattr(importlib.import_module(module_path), class_name)
    task = TaskClass(cfg)
    task.run()


if __name__ == "__main__":
    main()
