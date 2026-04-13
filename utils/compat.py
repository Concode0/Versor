# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Device compatibility shims.

Holds smart-dispatch wrappers for ops whose MPS kernels have broken
backward passes. Each helper is ``@torch.compiler.disable``'d so the
workaround stays out of ``torch.compile``'d graphs in the layers.
"""

import torch


@torch.compiler.disable
def safe_linalg_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``torch.linalg.solve`` with an MPS-safe backward path.

    The MPS backward of ``linalg_solve_ex`` emits out-of-bounds gather
    indices (uninitialized memory), so for MPS tensors we run the solve
    on CPU and move the result back. Autograd threads gradients through
    the device transfer correctly. All other devices dispatch directly
    to ``torch.linalg.solve``.
    """
    if A.device.type == 'mps':
        return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)
    return torch.linalg.solve(A, B)
