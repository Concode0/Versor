# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Runtime algebra hosts and dense reference operations."""

from .accessors import as_multivector, compact_values, grade_indices, hermitian_signs, materialize_dense, resolve_layout
from .algebra import CliffordAlgebra
from .context import AlgebraContext
from .multivector import Multivector
from .projected import AlgebraRuntimeMixin

__all__ = [
    "AlgebraContext",
    "AlgebraRuntimeMixin",
    "CliffordAlgebra",
    "Multivector",
    "as_multivector",
    "compact_values",
    "grade_indices",
    "hermitian_signs",
    "materialize_dense",
    "resolve_layout",
]
