# Versor: Universal Geometric Algebra Neural Network (C) 2026 Eunkyum Kim
# Licensed under the Apache License, Version 2.0 | "Unbending" Paradigm

"""Core mathematical kernel for Geometric Algebra.

Provides the Clifford algebra, conformal algebra, multivector wrapper,
metric functions, bivector decomposition, and signature search utilities.
"""

from .algebra import CliffordAlgebra
from .cga import ConformalAlgebra
from .multivector import Multivector
from .search import MetricSearch, GeodesicFlow, DimensionLifter
from .device import DeviceConfig, resolve_device
from .validation import check_multivector, check_channels

from .metric import (
    inner_product,
    induced_norm,
    geometric_distance,
    grade_purity,
    mean_active_grade,
    clifford_conjugate,
    hermitian_inner_product,
    hermitian_norm,
    hermitian_distance,
    hermitian_angle,
    grade_hermitian_norm,
    hermitian_grade_spectrum,
    signature_trace_form,
    signature_norm_squared,
)

from .decomposition import (
    ga_power_iteration,
    differentiable_invariant_decomposition,
    exp_simple_bivector,
    exp_decomposed,
)

__all__ = [
    # algebra
    "CliffordAlgebra",
    "ConformalAlgebra",
    "Multivector",
    # search
    "MetricSearch",
    "GeodesicFlow",
    "DimensionLifter",
    # device / validation
    "DeviceConfig",
    "resolve_device",
    "check_multivector",
    "check_channels",
    # metric
    "inner_product",
    "induced_norm",
    "geometric_distance",
    "grade_purity",
    "mean_active_grade",
    "clifford_conjugate",
    "hermitian_inner_product",
    "hermitian_norm",
    "hermitian_distance",
    "hermitian_angle",
    "grade_hermitian_norm",
    "hermitian_grade_spectrum",
    "signature_trace_form",
    "signature_norm_squared",
    # decomposition
    "ga_power_iteration",
    "differentiable_invariant_decomposition",
    "exp_simple_bivector",
    "exp_decomposed",
]
