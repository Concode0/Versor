"""Task-specific model architectures built on the Versor layer stack.

Models combine rotor layers, linear transformations, normalization,
and activation functions into complete architectures.
"""

from .gbn import GeometricBladeNetwork
from .sr_net import SRGBN, SRMultiGradeEmbedding
from .multi_rotor import MultiRotorModel
from .ga_transformer import GATransformerBlock, GALanguageModel
from .time_series import RotorTCN
from .lensing_net import LensingGBN

try:
    from .md17_forcenet import MD17ForceNet, MD17InteractionBlock
except ImportError:
    MD17ForceNet = None
    MD17InteractionBlock = None

__all__ = [
    "GeometricBladeNetwork",
    "SRGBN",
    "SRMultiGradeEmbedding",
    "MultiRotorModel",
    "GATransformerBlock",
    "GALanguageModel",
    "RotorTCN",
    "LensingGBN",
    # torch_geometric dependent
    "MD17ForceNet",
    "MD17InteractionBlock",
]
