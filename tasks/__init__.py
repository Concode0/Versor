"""Training tasks for the Versor framework.

Each task inherits from :class:`BaseTask` and implements the full lifecycle:
setup_algebra, setup_model, setup_criterion, get_data, train_step, evaluate, visualize.
"""

from .base import BaseTask
from .symbolic_regression import SRTask
from .md17 import MD17Task
from .lqa import LQATask
from .deap_eeg import DEAPEEGTask

__all__ = [
    "BaseTask",
    "SRTask",
    "MD17Task",
    "LQATask",
    "DEAPEEGTask",
]
