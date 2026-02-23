"""Neural network layers built on Clifford algebra.

Provides rotor layers, linear transformations, normalization,
attention, embeddings, and graph convolution modules.
"""

from .base import CliffordModule
from .rotor import RotorLayer
from .multi_rotor import MultiRotorLayer
from .linear import CliffordLinear
from .rotor_gadget import RotorGadget
from .normalization import CliffordLayerNorm
from .projection import BladeSelector
from .embedding import MultivectorEmbedding, RotaryBivectorPE
from .attention import GeometricProductAttention
from .multi_rotor_ffn import MultiRotorFFN

# CliffordGraphConv requires torch_geometric
try:
    from .gnn import CliffordGraphConv
except ImportError:
    CliffordGraphConv = None

__all__ = [
    "CliffordModule",
    "RotorLayer",
    "MultiRotorLayer",
    "CliffordLinear",
    "RotorGadget",
    "CliffordLayerNorm",
    "BladeSelector",
    "MultivectorEmbedding",
    "RotaryBivectorPE",
    "GeometricProductAttention",
    "MultiRotorFFN",
    "CliffordGraphConv",
]
