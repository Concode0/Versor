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
    """Base class. The foundation.

    Manages the algebra configuration.
    """

    def __init__(self, algebra: CliffordAlgebra):
        """Sets up the module.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
        """
        super().__init__()
        # Store minimal config to reconstruct algebra if needed
        self.p = algebra.p
        self.q = algebra.q
        self._algebra = algebra # transient reference

    @property
    def algebra(self) -> CliffordAlgebra:
        """Gets the algebra. Spawns it if missing."""
        if self._algebra is None:
            self._algebra = CliffordAlgebra(self.p, self.q)
        return self._algebra
    
    def forward(self, x):
        """Performs the forward pass computation."""
        raise NotImplementedError