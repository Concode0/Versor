"""Training tasks for the Versor framework.

Each task inherits from :class:`BaseTask` and implements the full lifecycle:
setup_algebra, setup_model, setup_criterion, get_data, train_step, evaluate, visualize.
"""

from .base import BaseTask

__all__ = [
    "BaseTask",
    "SRTask",
    "MD17Task",
    "LQATask",
    "DEAPEEGTask",
]


def __getattr__(name):
    if name == "SRTask":
        from .symbolic_regression import SRTask

        return SRTask
    if name == "MD17Task":
        from .md17 import MD17Task

        return MD17Task
    if name == "LQATask":
        from .lqa import LQATask

        return LQATask
    if name == "DEAPEEGTask":
        from .deap_eeg import DEAPEEGTask

        return DEAPEEGTask
    raise AttributeError(f"module 'tasks' has no attribute {name!r}")
