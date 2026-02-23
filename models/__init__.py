"""Task-specific model architectures built on the Versor layer stack.

Models combine rotor layers, linear transformations, normalization,
and activation functions into complete architectures.
"""

from .gbn import GeometricBladeNetwork
from .feynman_net import FeynmanGBN, FeynmanMultiGradeEmbedding
from .motion import MotionManifoldNetwork
from .multi_rotor import MultiRotorModel
from .ga_transformer import GATransformerBlock, GALanguageModel
from .weather_gbn import SphericalGraphConv, TemporalRotorLayer, WeatherGBN
from .cad_net import (
    ConformalPointNetEncoder,
    PointCloudDecoder,
    PrimitiveDecoder,
    CADAutoEncoder,
)
from .pdbbind_net import (
    ProteinEncoder,
    LigandEncoder,
    GeometricCrossAttention,
    PDBBindNet,
)
from .time_series import RotorTCN

# These models require torch_geometric
try:
    from .molecule import (
        MoleculeGNN,
        MultiRotorQuantumNet,
        GeometricInvariantBlock,
        MultiRotorInteractionBlock,
    )
except ImportError:
    MoleculeGNN = None
    MultiRotorQuantumNet = None
    GeometricInvariantBlock = None
    MultiRotorInteractionBlock = None

try:
    from .md17_forcenet import MD17ForceNet, MD17InteractionBlock
except ImportError:
    MD17ForceNet = None
    MD17InteractionBlock = None

__all__ = [
    "GeometricBladeNetwork",
    "FeynmanGBN",
    "FeynmanMultiGradeEmbedding",
    "MotionManifoldNetwork",
    "MultiRotorModel",
    "GATransformerBlock",
    "GALanguageModel",
    "SphericalGraphConv",
    "TemporalRotorLayer",
    "WeatherGBN",
    "ConformalPointNetEncoder",
    "PointCloudDecoder",
    "PrimitiveDecoder",
    "CADAutoEncoder",
    "ProteinEncoder",
    "LigandEncoder",
    "GeometricCrossAttention",
    "PDBBindNet",
    "RotorTCN",
    # torch_geometric dependent
    "MoleculeGNN",
    "MultiRotorQuantumNet",
    "GeometricInvariantBlock",
    "MultiRotorInteractionBlock",
    "MD17ForceNet",
    "MD17InteractionBlock",
]
