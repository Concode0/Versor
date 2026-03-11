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
from omegaconf import DictConfig, OmegaConf
from tasks.md17 import MD17Task
from tasks.symbolic_regression import SRTask
from tasks.lqa import LQATask
from tasks.deap_eeg import DEAPEEGTask

EXAMPLE_TASKS = {'manifold', 'hyperbolic', 'sanity'}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Entry point for task execution. Delegates to specific task handlers.

    Args:
        cfg (DictConfig): Hydra configuration dictionary.
    """
    task_name = cfg.name

    task_map = {
        'md17': MD17Task,
        'sr': SRTask,
        'lqa': LQATask,
        'deap_eeg': DEAPEEGTask,
    }

    if task_name in EXAMPLE_TASKS:
        raise ValueError(
            f"'{task_name}' is a synthetic example task. "
            f"Run it via: uv run python -m examples.main task={task_name}"
        )

    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(task_map.keys())}")

    TaskClass = task_map[task_name]
    task = TaskClass(cfg)
    task.run()

if __name__ == "__main__":
    main()
