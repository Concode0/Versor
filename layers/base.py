# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This project is fully open-source, including for commercial use.
# We believe Geometric Algebra is the future of AI, and we want 
# the industry to build upon this "unbending" paradigm.

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

