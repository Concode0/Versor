# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

"""Versor CLI Entry Point. Pick a task, any task.

Dispatches geometric learning tasks.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from tasks.motion import MotionAlignmentTask
from tasks.qm9 import QM9Task
from tasks.multi_rotor_qm9 import MultiRotorQM9Task
from tasks.semantic import SemanticTask

EXAMPLE_TASKS = {'manifold', 'hyperbolic', 'sanity'}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Entry point for task execution. Delegates to specific task handlers.

    Args:
        cfg (DictConfig): The plan.
    """
    task_name = cfg.name

    task_map = {
        'motion': MotionAlignmentTask,
        'qm9': QM9Task,
        'multi_rotor_qm9': MultiRotorQM9Task,
        'semantic': SemanticTask,
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