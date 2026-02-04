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

"""Multivector Container Class.

Provides a high-level object-oriented wrapper around raw tensors
to enable operator overloading (e.g., A * B for geometric product).
"""

import torch
from core.algebra import CliffordAlgebra

class Multivector:
    """Object-oriented wrapper for multivector tensors.

    Allows natural mathematical syntax like A * B, A + B, ~A.

    Attributes:
        algebra (CliffordAlgebra): The underlying algebra kernel.
        tensor (torch.Tensor): The raw coefficient tensor [..., Dim].
    """

    def __init__(self, algebra: CliffordAlgebra, tensor: torch.Tensor):
        """Initializes a Multivector.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            tensor (torch.Tensor): Coefficients.
        """
        self.algebra = algebra
        self.tensor = tensor

    @classmethod
    def from_vectors(cls, algebra: CliffordAlgebra, vectors: torch.Tensor):
        """Creates a Multivector from dense vectors (Grade 1).

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            vectors (torch.Tensor): Vectors [Batch, n].

        Returns:
            Multivector: Wrapper instance.
        """
        # Map vectors to multivector coefficients
        # e1 is index 1 (1<<0), e2 is index 2 (1<<1), etc.
        batch_shape = vectors.shape[:-1]
        mv_tensor = torch.zeros(*batch_shape, algebra.dim, device=vectors.device, dtype=vectors.dtype)
        
        for i in range(algebra.n):
            mv_tensor[..., 1 << i] = vectors[..., i]
            
        return cls(algebra, mv_tensor)

    def __repr__(self):
        return f"Multivector(shape={self.tensor.shape}, algebra=Cl({self.algebra.p},{self.algebra.q}))"

    def __add__(self, other):
        """Element-wise addition."""
        if isinstance(other, Multivector):
            assert self.algebra.n == other.algebra.n, "Algebras must match"
            return Multivector(self.algebra, self.tensor + other.tensor)
        elif isinstance(other, (int, float, torch.Tensor)):
            # Add to scalar part? Or broadcast add to tensor?
            # Standard python behavior: add to all coefficients if tensor
            # For scalar, strictly adding to scalar part is more geometric, 
            # but standard pytorch add is elementwise.
            # We follow Pytorch convention for Tensors.
            return Multivector(self.algebra, self.tensor + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Element-wise subtraction."""
        if isinstance(other, Multivector):
            return Multivector(self.algebra, self.tensor - other.tensor)
        else:
            return Multivector(self.algebra, self.tensor - other)

    def __mul__(self, other):
        """Geometric Product (A * B)."""
        if isinstance(other, Multivector):
            res = self.algebra.geometric_product(self.tensor, other.tensor)
            return Multivector(self.algebra, res)
        elif isinstance(other, (int, float)):
            return Multivector(self.algebra, self.tensor * other)
        else:
            return NotImplemented

    def __invert__(self):
        """Reversion (~A)."""
        return Multivector(self.algebra, self.algebra.reverse(self.tensor))

    def norm(self):
        """Metric-induced norm (sqrt(|<A ~A>|))."""
        from core.metric import induced_norm
        return induced_norm(self.algebra, self.tensor)

    def exp(self):
        """Exponential function."""
        return Multivector(self.algebra, self.algebra.exp(self.tensor))

    def grade(self, k: int):
        """Projects to grade k."""
        return Multivector(self.algebra, self.algebra.grade_projection(self.tensor, k))
