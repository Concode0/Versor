"""Task-specific model architectures built on the Versor layer stack.

Models combine rotor layers, linear transformations, normalization,
and activation functions into complete architectures.
"""

from .gbn import GeometricBladeNetwork
from .sr import SRGBN, SRMultiGradeEmbedding
from .multi_rotor import MultiRotorModel
from .time_series import RotorTCN
from .deap import EEGNet
from .lqa import GLRNet, ChainReasoningHead, EntailmentHead, NegationHead

try:
    from .md17 import MD17ForceNet, MD17InteractionBlock
except ImportError:
    MD17ForceNet = None
    MD17InteractionBlock = None

__all__ = [
    "GeometricBladeNetwork",
    "SRGBN",
    "SRMultiGradeEmbedding",
    "MultiRotorModel",
    "RotorTCN",
    "EEGNet",
    "GLRNet",
    "ChainReasoningHead",
    "EntailmentHead",
    "NegationHead",
    # torch_geometric dependent
    "MD17ForceNet",
    "MD17InteractionBlock",
]
