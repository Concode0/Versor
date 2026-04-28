# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Object-oriented multivector wrapper with operator overloading."""

from __future__ import annotations

import torch

from core.algebra import CliffordAlgebra


class Multivector:
    """Object-oriented multivector wrapper with operator overloading.

    Wraps a raw coefficient tensor and its parent ``CliffordAlgebra``,
    exposing every core algebra operation as a method or Python operator.

    Attributes:
        algebra (CliffordAlgebra): The backend.
        tensor (torch.Tensor): The raw data [..., Dim].
    """

    __slots__ = ("algebra", "tensor")

    def __init__(self, algebra: CliffordAlgebra, tensor: torch.Tensor):
        self.algebra = algebra
        self.tensor = tensor

    @classmethod
    def from_vectors(cls, algebra: CliffordAlgebra, vectors: torch.Tensor) -> Multivector:
        """Promotes vectors to multivectors (Grade 1)."""
        return cls(algebra, algebra.embed_vector(vectors))

    @classmethod
    def scalar(
        cls, algebra: CliffordAlgebra, value: float | torch.Tensor, batch_shape: tuple[int, ...] = ()
    ) -> Multivector:
        """Creates a scalar multivector (grade 0 only)."""
        dim = 2**algebra.n
        t = torch.zeros(*batch_shape, dim, device=algebra.device, dtype=algebra.dtype)
        t[..., 0] = value
        return cls(algebra, t)

    def __repr__(self):
        return f"Multivector(shape={self.tensor.shape}, algebra=Cl({self.algebra.p},{self.algebra.q},{self.algebra.r}))"

    def _check_algebra(self, other: Multivector) -> None:
        s, o = self.algebra, other.algebra
        if (s.p, s.q, s.r) != (o.p, o.q, o.r):
            raise ValueError(f"Algebra mismatch: Cl({s.p},{s.q},{s.r}) vs Cl({o.p},{o.q},{o.r})")

    def _wrap(self, tensor: torch.Tensor) -> Multivector:
        return Multivector(self.algebra, tensor)

    def __add__(self, other):
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self._wrap(self.tensor + other.tensor)
        if isinstance(other, (int, float, torch.Tensor)):
            return self._wrap(self.tensor + other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self._wrap(self.tensor - other.tensor)
        if isinstance(other, (int, float, torch.Tensor)):
            return self._wrap(self.tensor - other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self._wrap(other - self.tensor)
        return NotImplemented

    def __neg__(self):
        return self._wrap(-self.tensor)

    def __mul__(self, other):
        """Geometric product ``A * B``, or scalar scaling."""
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self._wrap(self.algebra.geometric_product(self.tensor, other.tensor))
        if isinstance(other, (int, float)):
            return self._wrap(self.tensor * other)
        if isinstance(other, torch.Tensor):
            return self._wrap(self.tensor * other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return self._wrap(self.tensor * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self._wrap(self.tensor / other)
        if isinstance(other, torch.Tensor):
            return self._wrap(self.tensor / other)
        return NotImplemented

    def __xor__(self, other):
        """Wedge (outer) product ``A ^ B``."""
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self._wrap(self.algebra.wedge(self.tensor, other.tensor))
        return NotImplemented

    def __or__(self, other):
        """Inner product ``A | B``."""
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self._wrap(self.algebra.inner_product(self.tensor, other.tensor))
        return NotImplemented

    def __invert__(self):
        """Reversion ``~A``."""
        return self._wrap(self.algebra.reverse(self.tensor))

    def grade(self, k: int) -> Multivector:
        """Extract the grade-k component."""
        return self._wrap(self.algebra.grade_projection(self.tensor, k))

    def reverse(self) -> Multivector:
        """Reversion (same as ``~self``)."""
        return self._wrap(self.algebra.reverse(self.tensor))

    def grade_involution(self) -> Multivector:
        """Grade involution (main involution): flips odd-grade signs."""
        return self._wrap(self.algebra.grade_involution(self.tensor))

    def clifford_conjugation(self) -> Multivector:
        """Clifford conjugation: grade_involution(reverse(x))."""
        return self._wrap(self.algebra.clifford_conjugation(self.tensor))

    def dual(self) -> Multivector:
        """Hodge dual: maps grade-k to grade-(n-k)."""
        return self._wrap(self.algebra.dual(self.tensor))

    def inverse(self) -> Multivector:
        """Blade inverse: B^{-1} = ~B / <B~B>_0."""
        return self._wrap(self.algebra.blade_inverse(self.tensor))

    def geometric_product(self, other: Multivector) -> Multivector:
        """Explicit geometric product (same as ``self * other``)."""
        self._check_algebra(other)
        return self._wrap(self.algebra.geometric_product(self.tensor, other.tensor))

    def wedge(self, other: Multivector) -> Multivector:
        """Wedge (outer) product (same as ``self ^ other``)."""
        self._check_algebra(other)
        return self._wrap(self.algebra.wedge(self.tensor, other.tensor))

    def inner(self, other: Multivector) -> Multivector:
        """Inner product (same as ``self | other``)."""
        self._check_algebra(other)
        return self._wrap(self.algebra.inner_product(self.tensor, other.tensor))

    def left_contraction(self, other: Multivector) -> Multivector:
        """Left contraction: ``self _| other``."""
        self._check_algebra(other)
        return self._wrap(self.algebra.left_contraction(self.tensor, other.tensor))

    def right_contraction(self, other: Multivector) -> Multivector:
        """Right contraction: ``self |_ other``."""
        self._check_algebra(other)
        return self._wrap(self.algebra.right_contraction(self.tensor, other.tensor))

    def commutator(self, other: Multivector) -> Multivector:
        """Commutator (Lie bracket): ``[self, other] = self*other - other*self``."""
        self._check_algebra(other)
        return self._wrap(self.algebra.commutator(self.tensor, other.tensor))

    def anti_commutator(self, other: Multivector) -> Multivector:
        """Anti-commutator: ``{self, other} = self*other + other*self``."""
        self._check_algebra(other)
        return self._wrap(self.algebra.anti_commutator(self.tensor, other.tensor))

    def norm(self) -> torch.Tensor:
        """Induced metric norm (returns scalar tensor)."""
        from core.metric import induced_norm

        return induced_norm(self.algebra, self.tensor)

    def norm_sq(self) -> torch.Tensor:
        """Squared norm: <x * ~x>_0 (returns scalar tensor)."""
        return self.algebra.norm_sq(self.tensor)

    def get_grade_norms(self) -> torch.Tensor:
        """Per-grade L2 norms."""
        return self.algebra.get_grade_norms(self.tensor)

    def exp(self) -> Multivector:
        """Exponential map (bivector -> rotor)."""
        return self._wrap(self.algebra.exp(self.tensor))

    def sandwich(self, x: Multivector) -> Multivector:
        """Sandwich product: ``self * x * ~self``.

        Falls back to two geometric products when the tensor shapes
        don't match the optimized [N, D] + [N, C, D] layout.
        """
        self._check_algebra(x)
        R, xt = self.tensor, x.tensor
        # Optimized path: R is [N, D], x is [N, C, D]
        if R.dim() == 2 and xt.dim() == 3:
            return self._wrap(self.algebra.sandwich_product(R, xt))
        # General fallback: two GPs
        R_rev = self.algebra.reverse(R)
        return self._wrap(self.algebra.geometric_product(self.algebra.geometric_product(R, xt), R_rev))

    def reflect(self, n: Multivector) -> Multivector:
        """Reflect self through hyperplane orthogonal to vector n."""
        self._check_algebra(n)
        return self._wrap(self.algebra.reflect(self.tensor, n.tensor))

    def versor_product(self, x: Multivector) -> Multivector:
        """General versor action: ``hat(self) * x * self^{-1}``."""
        self._check_algebra(x)
        return self._wrap(self.algebra.versor_product(self.tensor, x.tensor))

    def blade_project(self, blade: Multivector) -> Multivector:
        """Project onto blade subspace: ``(self · B) B^{-1}``."""
        self._check_algebra(blade)
        return self._wrap(self.algebra.blade_project(self.tensor, blade.tensor))

    def blade_reject(self, blade: Multivector) -> Multivector:
        """Reject from blade subspace: ``self - proj_B(self)``."""
        self._check_algebra(blade)
        return self._wrap(self.algebra.blade_reject(self.tensor, blade.tensor))

    def to(self, *args, **kwargs) -> Multivector:
        """Move/cast the underlying tensor (same API as ``torch.Tensor.to``)."""
        return self._wrap(self.tensor.to(*args, **kwargs))

    def detach(self) -> Multivector:
        """Detach from computation graph."""
        return self._wrap(self.tensor.detach())

    def clone(self) -> Multivector:
        """Clone the underlying tensor."""
        return self._wrap(self.tensor.clone())

    def requires_grad_(self, requires_grad: bool = True) -> Multivector:
        """Set requires_grad in-place."""
        self.tensor.requires_grad_(requires_grad)
        return self

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype
