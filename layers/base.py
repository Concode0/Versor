# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
# https://github.com/Concode0/Versor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [INTELLECTUAL PROPERTY NOTICE]
# This implementation is protected under ROK Patent Application 10-2026-0023023.
# All rights reserved. Commercial use, redistribution, or modification 
# for-profit without an explicit commercial license is strictly prohibited.
#
# Contact for Commercial Licensing: nemonanconcode@gmail.com

import torch.nn as nn
from core.algebra import CliffordAlgebra

class CliffordModule(nn.Module):
    """Base class for all Neural Network layers operating on Multivectors.

    Ensures that the algebra configuration (metric signature) is consistent
    across layers and supports serialization by storing configuration parameters
    instead of the heavy algebra object.

    Attributes:
        p (int): Positive metric dimensions.
        q (int): Negative metric dimensions.
        _algebra (CliffordAlgebra): Lazy-loaded algebra instance.
    """

    def __init__(self, algebra: CliffordAlgebra):
        """Initializes the Clifford Module.

        Args:
            algebra (CliffordAlgebra): The algebra instance defining the geometric space.
        """
        super().__init__()
        # Store minimal config to reconstruct algebra if needed (e.g. after loading state_dict)
        self.p = algebra.p
        self.q = algebra.q
        self._algebra = algebra # transient reference

    @property
    def algebra(self) -> CliffordAlgebra:
        """Returns the Clifford Algebra instance, initializing it if necessary."""
        if self._algebra is None:
            self._algebra = CliffordAlgebra(self.p, self.q)
        return self._algebra
    
    def forward(self, x):
        """Defines the computation performed at every call."""
        raise NotImplementedError

