"""Neural network layers built on Clifford algebra.

Organized into Primitives, Canonical Blocks, and Task-Specific Adapters.
"""

from .primitives.base import CliffordModule
from .primitives.rotor import RotorLayer
from .primitives.multi_rotor import MultiRotorLayer
from .primitives.linear import CliffordLinear
from .primitives.rotor_gadget import RotorGadget
from .primitives.normalization import CliffordLayerNorm
from .primitives.projection import BladeSelector
from .adapters.embedding import MultivectorEmbedding, RotaryBivectorPE
from .blocks.attention import GeometricProductAttention
from .blocks.multi_rotor_ffn import MultiRotorFFN

# CliffordGraphConv requires torch_geometric
try:
    from .adapters.gnn import CliffordGraphConv
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
