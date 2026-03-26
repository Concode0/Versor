# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Object-oriented multivector wrapper with operator overloading."""

import torch
from core.algebra import CliffordAlgebra

class Multivector:
    """Object-oriented multivector wrapper with operator overloading.

    Attributes:
        algebra (CliffordAlgebra): The backend.
        tensor (torch.Tensor): The raw data [..., Dim].
    """

    def __init__(self, algebra: CliffordAlgebra, tensor: torch.Tensor):
        """Wraps the tensor.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            tensor (torch.Tensor): Coefficients.
        """
        self.algebra = algebra
        self.tensor = tensor

    @classmethod
    def from_vectors(cls, algebra: CliffordAlgebra, vectors: torch.Tensor) -> "Multivector":
        """Promotes vectors to multivectors (Grade 1).

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            vectors (torch.Tensor): Vectors [Batch, n].

        Returns:
            Multivector: The wrapper.
        """
        return cls(algebra, algebra.embed_vector(vectors))

    def __repr__(self):
        return f"Multivector(shape={self.tensor.shape}, algebra=Cl({self.algebra.p},{self.algebra.q}))"

    def __add__(self, other):
        """Compute the multivector sum."""
        if isinstance(other, Multivector):
            assert self.algebra.n == other.algebra.n, "Algebras must match"
            return Multivector(self.algebra, self.tensor + other.tensor)
        elif isinstance(other, (int, float, torch.Tensor)):
            # Broadcast add
            return Multivector(self.algebra, self.tensor + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Compute the multivector difference."""
        if isinstance(other, Multivector):
            return Multivector(self.algebra, self.tensor - other.tensor)
        else:
            return Multivector(self.algebra, self.tensor - other)

    def __mul__(self, other):
        """Compute the geometric product A * B."""
        if isinstance(other, Multivector):
            res = self.algebra.geometric_product(self.tensor, other.tensor)
            return Multivector(self.algebra, res)
        elif isinstance(other, (int, float)):
            return Multivector(self.algebra, self.tensor * other)
        else:
            return NotImplemented

    def __invert__(self):
        """Compute the reversion ~A."""
        return Multivector(self.algebra, self.algebra.reverse(self.tensor))

    def norm(self):
        """Compute the induced metric norm."""
        from core.metric import induced_norm
        return induced_norm(self.algebra, self.tensor)

    def exp(self):
        """Exponentiate via the algebra exp map."""
        return Multivector(self.algebra, self.algebra.exp(self.tensor))

    def grade(self, k: int):
        """Extract the grade-k component."""
        return Multivector(self.algebra, self.algebra.grade_projection(self.tensor, k))