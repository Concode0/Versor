"""Versor - Geometric Algebra Deep Learning framework for PyTorch."""

__version__ = "1.0.0"

from core.algebra import CliffordAlgebra
from layers import CliffordLinear, RotorLayer

__all__ = [
    "__version__",
    "CliffordAlgebra",
    "RotorLayer",
    "CliffordLinear",
]
