# Versor: Universal Geometric Algebra Neural Network (C) 2026 Eunkyum Kim
# Licensed under the Apache License, Version 2.0

"""Core mathematical kernel for Geometric Algebra.

Provides the Clifford algebra, conformal algebra, multivector wrapper,
metric functions, bivector decomposition, and signature search utilities.

The ``core.analysis`` sub-package (``MetricSearch``, ``GeodesicFlow``,
``GeometricAnalyzer``, etc.) is **lazily imported** - it is not loaded
until first access, keeping ``import core`` lightweight.
"""

from .config import AlgebraConfig, make_algebra, make_algebra_from_config
from .foundation.basis import (
    GradeProductOp,
    basis_indices_for_grades,
    basis_product,
    expand_output_grades,
    geometric_product_output_grades,
    normalize_grades,
    operation_coefficient,
    reverse_sign,
)
from .foundation.device import DeviceConfig, dtype_name, optional_dtype, resolve_device, resolve_dtype
from .foundation.layout import AlgebraSpec, GradeLayout
from .foundation.module import AlgebraLike, CliffordModule
from .foundation.validation import check_channels, check_multivector
from .planning.flow import GradeFlow
from .planning.layouts import ProductRequest, build_product_request
from .planning.planner import GradePlanner
from .planning.policy import DEFAULT_PLANNING_LIMITS, PlanCost, PlanningLimits
from .planning.product import GradeProductExecutor, GradeProductPlan, build_grade_product_plan
from .planning.routes import (
    ModuleOptimizationIssue,
    ModuleOptimizationPlan,
    ModuleOptimizationReport,
    collect_module_optimization_plans,
    inspect_module_optimization,
    module_optimization_plan,
)
from .planning.tree import GradePathNode, GradePlanTree, build_grade_plan_tree
from .planning.unary import GradeUnaryExecutor, GradeUnaryOp, GradeUnaryPlan, UnaryRequest, build_unary_request
from .runtime.accessors import (
    as_multivector,
    compact_values,
    grade_indices,
    hermitian_signs,
    materialize_dense,
    resolve_layout,
)
from .runtime.algebra import CliffordAlgebra
from .runtime.context import AlgebraContext
from .runtime.decomposition import (
    ExpPolicy,
    compiled_safe_decomposed_exp,
    differentiable_invariant_decomposition,
    exp_simple_bivector,
    ga_power_iteration,
)
from .runtime.metric import (
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
from .runtime.multivector import Multivector

__all__ = [
    # algebra
    "AlgebraContext",
    "CliffordAlgebra",
    "AlgebraConfig",
    "AlgebraLike",
    "CliffordModule",
    "Multivector",
    "AlgebraSpec",
    "GradeLayout",
    "GradePlanner",
    "PlanningLimits",
    "PlanCost",
    "DEFAULT_PLANNING_LIMITS",
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
    "as_multivector",
    "compact_values",
    "grade_indices",
    "hermitian_signs",
    "materialize_dense",
    "resolve_layout",
    # decomposition
    "ExpPolicy",
    "ga_power_iteration",
    "differentiable_invariant_decomposition",
    "exp_simple_bivector",
    "compiled_safe_decomposed_exp",
    # static sparse grade planning
    "GradeProductOp",
    "GradeProductExecutor",
    "GradeProductPlan",
    "GradePathNode",
    "GradePlanTree",
    "GradeFlow",
    "ModuleOptimizationIssue",
    "ModuleOptimizationPlan",
    "ModuleOptimizationReport",
    "ProductRequest",
    "GradeUnaryExecutor",
    "GradeUnaryOp",
    "GradeUnaryPlan",
    "UnaryRequest",
    "basis_indices_for_grades",
    "basis_product",
    "build_grade_product_plan",
    "build_grade_plan_tree",
    "build_product_request",
    "build_unary_request",
    "collect_module_optimization_plans",
    "expand_output_grades",
    "geometric_product_output_grades",
    "inspect_module_optimization",
    "module_optimization_plan",
    "normalize_grades",
    "operation_coefficient",
    "reverse_sign",
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
