# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Device compatibility shims.

Holds smart-dispatch wrappers for ops whose MPS kernels have broken
backward passes. Dispatch is resolved at import time so non-MPS
builds get the raw op as the same symbol — no per-call branch, no
Dynamo guard, no chance of a compile-time graph break from the
workaround path.
"""

import torch


if torch.backends.mps.is_available():

    def safe_linalg_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.is_mps:
            return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)
        return torch.linalg.solve(A, B)
else:
    safe_linalg_solve = torch.linalg.solve
