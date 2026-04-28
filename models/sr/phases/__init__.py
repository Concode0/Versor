# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Phase mixins for the SR pipeline.

Each mixin groups related methods from the original monolithic
IterativeUnbender class.  The pipeline.py module composes them
via multiple inheritance.
"""

from models.sr.phases.cross_terms import CrossTermMixin
from models.sr.phases.extraction import ExtractionMixin
from models.sr.phases.prep import PrepMixin
from models.sr.phases.refinement import RefinementMixin

__all__ = [
    "PrepMixin",
    "ExtractionMixin",
    "CrossTermMixin",
    "RefinementMixin",
]
