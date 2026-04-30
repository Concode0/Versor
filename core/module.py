# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Base PyTorch module for components that share a Clifford algebra."""

import torch.nn as nn

from .algebra import CliffordAlgebra


class CliffordModule(nn.Module):
    """Base module for Clifford algebra-aware components.

    ``CliffordModule`` belongs to :mod:`core` because it is shared by layers,
    functional losses/activations, models, examples, and experiments. Keeping it
    out of :mod:`layers` prevents functional code from importing the eager layer
    package just to subclass this base type.

    The module stores a shared :class:`CliffordAlgebra` reference without
    registering it as a PyTorch submodule. In Versor, one algebra instance often
    owns the precomputed geometric tensors used by many modules.
    """

    def __init__(self, algebra: CliffordAlgebra):
        """Set up the module with a shared algebra instance."""
        super().__init__()
        # Bypass nn.Module.__setattr__ to avoid registering algebra as a child.
        object.__setattr__(self, "_algebra", algebra)

    @property
    def algebra(self) -> CliffordAlgebra:
        """Return the shared algebra instance."""
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
        """Apply device/dtype moves to this module and its shared algebra."""
        result = super()._apply(fn)
        if self._algebra is not None:
            self._algebra._apply(fn)
        return result

    def forward(self, x):
        """Perform the forward pass computation."""
        raise NotImplementedError
