# Versor: Universal Geometric Algebra Neural Network (C) 2026 Eunkyum Kim
# Licensed under the Apache License, Version 2.0

"""Core mathematical kernel for Geometric Algebra.

Provides the Clifford algebra, conformal algebra, multivector wrapper,
metric functions, bivector decomposition, and signature search utilities.

The ``core.analysis`` sub-package (``MetricSearch``, ``GeodesicFlow``,
``GeometricAnalyzer``, etc.) is **lazily imported** - it is not loaded
until first access, keeping ``import core`` lightweight.
"""

from .algebra import CliffordAlgebra
from .config import AlgebraConfig, PartitionConfig, make_algebra, make_algebra_from_config
from .decomposition import (
    ExpPolicy,
    compiled_safe_decomposed_exp,
    differentiable_invariant_decomposition,
    exp_simple_bivector,
    ga_power_iteration,
)
from .device import DeviceConfig, dtype_name, optional_dtype, resolve_device, resolve_dtype
from .metric import (
    clifford_conjugate,
    geometric_distance,
    grade_hermitian_norm,
    grade_purity,
    hermitian_angle,
    hermitian_distance,
    hermitian_grade_spectrum,
    hermitian_inner_product,
    hermitian_norm,
    induced_norm,
    inner_product,
    mean_active_grade,
    signature_norm_squared,
    signature_trace_form,
)
from .module import AlgebraLike, CliffordModule
from .multivector import Multivector
from .partitioned_algebra import PartitionedCliffordAlgebra
from .validation import check_channels, check_multivector

__all__ = [
    # algebra
    "CliffordAlgebra",
    "AlgebraConfig",
    "AlgebraLike",
    "CliffordModule",
    "Multivector",
    "PartitionConfig",
    "PartitionedCliffordAlgebra",
    "make_algebra",
    "make_algebra_from_config",
    # device / validation
    "DeviceConfig",
    "dtype_name",
    "optional_dtype",
    "resolve_device",
    "resolve_dtype",
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
    "ExpPolicy",
    "ga_power_iteration",
    "differentiable_invariant_decomposition",
    "exp_simple_bivector",
    "compiled_safe_decomposed_exp",
    # analysis (lazy)
    "MetricSearch",
    "GeodesicFlow",
    "DimensionLifter",
    "GeometricAnalyzer",
    "AnalysisReport",
]

# -----------------------------------------------------------------------
# Lazy imports for the analysis sub-package.
# These names are only resolved when first accessed, keeping
# ``import core`` fast and avoiding circular-import issues.
# -----------------------------------------------------------------------
_ANALYSIS_NAMES = {
    "MetricSearch",
    "GeodesicFlow",
    "DimensionLifter",
    "GeometricAnalyzer",
    "AnalysisReport",
    "compute_uncertainty_and_alignment",
    "SignatureSearchAnalyzer",
    "EffectiveDimensionAnalyzer",
    "SpectralAnalyzer",
    "SymmetryDetector",
    "CommutatorAnalyzer",
    "StatisticalSampler",
    "SamplingConfig",
    "AnalysisConfig",
}


def __getattr__(name: str):
    if name in _ANALYSIS_NAMES:
        from . import analysis as _analysis  # noqa: F811

        obj = getattr(_analysis, name)
        # Cache on the module to avoid repeated __getattr__ calls
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
