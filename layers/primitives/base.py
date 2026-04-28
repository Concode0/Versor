# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import torch.nn as nn
from core.algebra import CliffordAlgebra


class CliffordModule(nn.Module):
    """Base module for Clifford algebra layers.

    This module securely stores a reference to a shared ``CliffordAlgebra`` instance
    without registering it as a PyTorch submodule. In the Versor architecture,
    a single algebra instance (which contains precomputed geometric tensors)
    is heavily shared across multiple layers.

    By bypassing standard submodule registration (via ``object.__setattr__``) and
    overriding ``_apply``, this base class ensures that:
    1. No ownership conflicts occur in PyTorch's computational graph.
    2. Device and dtype casting (e.g., ``.to(device)``, ``.cuda()``, ``.half()``)
       are automatically and safely propagated to the shared algebra buffers.
    """

    def __init__(self, algebra: CliffordAlgebra):
        """Sets up the module.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
        """
        super().__init__()
        # Bypass nn.Module.__setattr__ to avoid registering algebra as submodule.
        # Multiple layers share the same algebra - only one should "own" it.
        object.__setattr__(self, '_algebra', algebra)

    @property
    def algebra(self) -> CliffordAlgebra:
        """Return the algebra instance."""
        return self._algebra

    @property
    def p(self):
        return self._algebra.p

    @property
    def q(self):
        return self._algebra.q

    @property
    def r(self):
        return self._algebra.r

    def _apply(self, fn):
        """Override to also move the shared algebra tables."""
        result = super()._apply(fn)
        if self._algebra is not None:
            self._algebra._apply(fn)
        return result

    def forward(self, x):
        """Performs the forward pass computation."""
        raise NotImplementedError
