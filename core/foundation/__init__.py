# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Foundation value objects, basis utilities, validation, and device helpers."""

from .basis import (
    GradeProductOp,
    basis_count_for_grades,
    basis_index_tuple_for_grades,
    basis_indices_for_grades,
    basis_product,
    expand_output_grades,
    geometric_product_output_grades,
    normalize_grades,
    operation_coefficient,
    operation_may_be_nonzero,
    product_output_grades,
    reverse_sign,
)
from .device import DeviceConfig, dtype_name, optional_dtype, resolve_device, resolve_dtype
from .layout import AlgebraSpec, GradeLayout
from .module import AlgebraLike, CliffordModule
from .validation import check_channels, check_multivector

__all__ = [
    "AlgebraLike",
    "AlgebraSpec",
    "CliffordModule",
    "DeviceConfig",
    "GradeLayout",
    "GradeProductOp",
    "basis_count_for_grades",
    "basis_index_tuple_for_grades",
    "basis_indices_for_grades",
    "basis_product",
    "check_channels",
    "check_multivector",
    "dtype_name",
    "expand_output_grades",
    "geometric_product_output_grades",
    "normalize_grades",
    "operation_coefficient",
    "operation_may_be_nonzero",
    "optional_dtype",
    "product_output_grades",
    "reverse_sign",
    "resolve_device",
    "resolve_dtype",
]
