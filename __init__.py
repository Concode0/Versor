"""Versor - Geometric Algebra Deep Learning framework for PyTorch."""

__version__ = "1.0.0"

from core.algebra import CliffordAlgebra
from core.config import DEFAULT_PARTITION_LEAF_N, AlgebraConfig, PartitionConfig, make_algebra, make_algebra_from_config
from core.module import CliffordModule
from core.partitioned_algebra import MAX_PARTITIONED_DIMENSIONS, PartitionedCliffordAlgebra
from layers import CliffordLinear, RotorLayer

__all__ = [
    "__version__",
    "AlgebraConfig",
    "CliffordAlgebra",
    "CliffordModule",
    "DEFAULT_PARTITION_LEAF_N",
    "MAX_PARTITIONED_DIMENSIONS",
    "PartitionConfig",
    "PartitionedCliffordAlgebra",
    "make_algebra",
    "make_algebra_from_config",
    "RotorLayer",
    "CliffordLinear",
]
