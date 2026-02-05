# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

"""Entry point for the Versor CLI.

Uses Hydra for configuration management to dispatch specific geometric learning tasks.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from tasks.manifold import ManifoldTask
from tasks.cross_modal import CrossModalTask
from tasks.hyperbolic import HyperbolicTask
from tasks.semantic import SemanticTask
from tasks.sanity_check import SanityCheckTask
from tasks.motion import MotionAlignmentTask
from tasks.so3 import SO3InvariantTask
from tasks.qm9 import QM9Task

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main execution function.

    Args:
        cfg (DictConfig): Hydra configuration object.
    
    Raises:
        ValueError: If the requested task is not found in the task map.
    """
    # Determine which task to run based on config
    task_name = cfg.name
    
    task_map = {
        'manifold': ManifoldTask,
        'crossmodal': CrossModalTask,
        'hyperbolic': HyperbolicTask,
        'semantic': SemanticTask,
        'sanity': SanityCheckTask,
        'motion': MotionAlignmentTask,
        'so3': SO3InvariantTask,
        'qm9': QM9Task
    }
    
    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}")
        
    TaskClass = task_map[task_name]
    task = TaskClass(cfg)
    task.run()

if __name__ == "__main__":
    main()