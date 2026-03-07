"""Training tasks for the Versor framework.

Each task inherits from :class:`BaseTask` and implements the full lifecycle:
setup_algebra, setup_model, setup_criterion, get_data, train_step, evaluate, visualize.
"""

from .base import BaseTask
from .symbolic_regression import SRTask
from .md17 import MD17Task
from .lm import LanguageModelingTask
from .deeplense import DeepLenseTask

__all__ = [
    "BaseTask",
    "SRTask",
    "MD17Task",
    "LanguageModelingTask",
    "DeepLenseTask",
]
