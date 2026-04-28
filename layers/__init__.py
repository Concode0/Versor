"""Neural network layers built on Clifford algebra.

Organized into Primitives, Canonical Blocks, and Task-Specific Adapters.
"""

from .adapters.embedding import MultivectorEmbedding, RotaryBivectorPE
from .adapters.mother import EntropyGatedAttention, MotherEmbedding, PhaseShiftHead
from .blocks.attention import GeometricProductAttention
from .blocks.multi_rotor_ffn import MultiRotorFFN
from .blocks.transformer import GeometricTransformerBlock
from .primitives.base import CliffordModule
from .primitives.linear import CliffordLinear
from .primitives.multi_rotor import MultiRotorLayer
from .primitives.normalization import CliffordLayerNorm
from .primitives.projection import BladeSelector, GeometricNeutralizer
from .primitives.rotor import RotorLayer
from .primitives.rotor_gadget import RotorGadget

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
    "GeometricNeutralizer",
    "MultivectorEmbedding",
    "RotaryBivectorPE",
    "MotherEmbedding",
    "EntropyGatedAttention",
    "PhaseShiftHead",
    "GeometricProductAttention",
    "MultiRotorFFN",
    "GeometricTransformerBlock",
    "CliffordGraphConv",
]
