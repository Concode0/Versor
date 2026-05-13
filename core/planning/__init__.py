# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Static grade planning and Torch executor lowering."""

from .flow import GradeFlow
from .layouts import ProductRequest, build_product_request
from .planner import GradePlanner
from .product import GradeProductExecutor, GradeProductPlan, build_grade_product_plan
from .routes import (
    ModuleOptimizationIssue,
    ModuleOptimizationPlan,
    ModuleOptimizationReport,
    collect_module_optimization_plans,
    inspect_module_optimization,
    module_optimization_plan,
)
from .tree import GradePathNode, GradePlanTree, build_grade_plan_tree
from .unary import GradeUnaryExecutor, GradeUnaryOp, GradeUnaryPlan, UnaryRequest, build_unary_request

__all__ = [
    "GradeFlow",
    "GradePathNode",
    "GradeProductExecutor",
    "GradeProductPlan",
    "GradePlanTree",
    "GradePlanner",
    "ModuleOptimizationIssue",
    "ModuleOptimizationPlan",
    "ModuleOptimizationReport",
    "GradeUnaryExecutor",
    "GradeUnaryOp",
    "GradeUnaryPlan",
    "ProductRequest",
    "UnaryRequest",
    "build_grade_product_plan",
    "build_grade_plan_tree",
    "build_product_request",
    "build_unary_request",
    "collect_module_optimization_plans",
    "inspect_module_optimization",
    "module_optimization_plan",
]
