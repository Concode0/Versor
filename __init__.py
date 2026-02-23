"""Versor — Geometric Algebra Deep Learning framework for PyTorch."""

__version__ = "0.1.0"

from core.algebra import CliffordAlgebra
from layers.rotor import RotorLayer
from layers.linear import CliffordLinear

__all__ = [
    "__version__",
    "CliffordAlgebra",
    "RotorLayer",
    "CliffordLinear",
]
