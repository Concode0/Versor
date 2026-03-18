# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Virtual Machine components: LLM bridge, grade-aware projections, attention."""

from .bridge import LLMBridge
from .projections import GradeAwareProjectionIn, GradeWeightedProjectionOut
from .attention import GradeMaskedAttention

__all__ = [
    "LLMBridge",
    "GradeAwareProjectionIn",
    "GradeWeightedProjectionOut",
    "GradeMaskedAttention",
]
