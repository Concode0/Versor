"""Training tasks for the Versor framework.

Each task inherits from :class:`BaseTask` and implements the full lifecycle:
setup_algebra, setup_model, setup_criterion, get_data, train_step, evaluate, visualize.
"""

from .base import BaseTask
from .feynman import FeynmanTask
from .feynman_sweep import FeynmanSweepTask
from .qm9 import QM9Task
from .semantic import SemanticTask
from .md17 import MD17Task
from .motion import MotionAlignmentTask
from .multi_rotor_qm9 import MultiRotorQM9Task
from .abc import ABCTask
from .lm import LanguageModelingTask
from .pdbbind import PDBBindTask
from .weatherbench import WeatherBenchTask

__all__ = [
    "BaseTask",
    "FeynmanTask",
    "FeynmanSweepTask",
    "QM9Task",
    "SemanticTask",
    "MD17Task",
    "MotionAlignmentTask",
    "MultiRotorQM9Task",
    "ABCTask",
    "LanguageModelingTask",
    "PDBBindTask",
    "WeatherBenchTask",
]
