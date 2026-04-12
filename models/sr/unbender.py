# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Re-export shim for backward compatibility.

The implementation has been decomposed into:
  - models/sr/pipeline.py: IterativeUnbender class + dataclasses
  - models/sr/phases/prep.py: Phase 0 (PrepMixin)
  - models/sr/phases/extraction.py: Phase 1 (ExtractionMixin)
  - models/sr/phases/cross_terms.py: Phase 2 (CrossTermMixin)
  - models/sr/phases/refinement.py: Phase 3 (RefinementMixin)
"""

from models.sr.pipeline import (  # noqa: F401
    IterativeUnbender,
    UnbendingResult,
    StageResult,
    OrthogonalEliminationResult,
    _PrepResult,
)
