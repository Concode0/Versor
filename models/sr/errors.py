# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Typed exception hierarchy for the SR pipeline.

Replaces broad ``except Exception`` blocks with specific exception types
so failures can be diagnosed and handled at appropriate granularity.
"""


class SRPipelineError(Exception):
    """Base exception for all SR pipeline errors."""

    pass


class MetricSearchError(SRPipelineError):
    """MetricSearch failed to find a valid signature."""

    pass


class SymPyTimeoutError(SRPipelineError):
    """A SymPy operation (solve, simplify) exceeded its timeout."""

    pass


class NumericalInstabilityError(SRPipelineError):
    """A numerical operation produced non-finite values."""

    pass


class ExtractionError(SRPipelineError):
    """Formula extraction failed (no terms found, or translation error)."""

    pass


class AlgebraConstructionError(SRPipelineError):
    """CliffordAlgebra construction or mother-algebra build failed."""

    pass
