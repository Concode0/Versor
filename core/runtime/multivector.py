# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Object-oriented multivector wrapper with operator overloading."""

from __future__ import annotations

import torch

from core.foundation.layout import GradeLayout
from core.foundation.module import AlgebraLike
from core.runtime.accessors import as_multivector as _as_multivector
from core.runtime.accessors import materialize_dense


class Multivector:
    """Object-oriented multivector wrapper with operator overloading.

    Wraps a raw coefficient tensor and its parent algebra kernel,
    exposing every core algebra operation as a method or Python operator.

    Attributes:
        algebra (AlgebraLike): The backend.
        tensor (torch.Tensor): Dense coefficients [..., Dim].
        values (torch.Tensor): Optional compact lane values [..., layout.dim].
        layout (GradeLayout): Optional compact layout for ``values``.
    """

    __slots__ = ("_tensor", "algebra", "layout", "values")

    def __init__(
        self,
        algebra: AlgebraLike,
        tensor: torch.Tensor = None,
        *,
        values: torch.Tensor = None,
        layout: GradeLayout = None,
    ):
        self.algebra = algebra
        self.layout = layout
        if layout is None:
            if tensor is None:
                raise ValueError("tensor is required when layout is not provided")
            self._tensor = tensor
            self.values = None
        else:
            self._check_layout(layout)
            if values is None:
                if tensor is None:
                    raise ValueError("values or tensor is required when layout is provided")
                values = layout.compact(tensor)
            if values.shape[-1] != layout.dim:
                raise ValueError(f"compact values last dimension must be {layout.dim}, got {values.shape[-1]}")
            self._tensor = None
            self.values = values

    @classmethod
    def from_tensor(
        cls,
        algebra: AlgebraLike,
        tensor: torch.Tensor,
        *,
        grades=None,
        layout: GradeLayout = None,
    ) -> Multivector:
        """Wrap dense coefficients or compact coefficients with declared layout metadata."""
        if layout is None and grades is None:
            return cls(algebra, tensor)
        return _as_multivector(algebra, tensor, layout=layout, grades=grades)

    @classmethod
    def from_compact(
        cls,
        algebra: AlgebraLike,
        values: torch.Tensor,
        *,
        grades=None,
        layout: GradeLayout = None,
    ) -> Multivector:
        """Create a compact multivector from active lane values."""
        if layout is None and grades is None:
            raise ValueError("layout or grades is required for compact multivectors")
        resolved = algebra.resolve_layout(layout=layout, grades=grades, warn_full=False)
        return cls(algebra, values=values, layout=resolved)

    @classmethod
    def from_vectors(cls, algebra: AlgebraLike, vectors: torch.Tensor) -> Multivector:
        """Promote vectors to compact grade-1 multivectors."""
        layout = algebra.layout((1,))
        if vectors.shape[-1] != layout.dim:
            raise ValueError(f"vectors last dimension must be {layout.dim}, got {vectors.shape[-1]}")
        return cls(algebra, values=vectors, layout=layout)

    @classmethod
    def scalar(
        cls, algebra: AlgebraLike, value: float | torch.Tensor, batch_shape: tuple[int, ...] = ()
    ) -> Multivector:
        """Creates a scalar multivector (grade 0 only)."""
        layout = algebra.layout((0,))
        values = torch.as_tensor(value, device=algebra.device, dtype=algebra.dtype)
        if batch_shape:
            target_shape = torch.Size(batch_shape)
            if values.ndim == 0:
                values = values.expand(*batch_shape).clone()
            elif values.shape == torch.Size((*batch_shape, 1)):
                return cls(algebra, values=values, layout=layout)
            elif values.shape != target_shape:
                values = values.expand(*batch_shape).clone()
        if values.ndim == 0:
            values = values.reshape(1)
        elif values.shape[-1] != 1:
            values = values.unsqueeze(-1)
        return cls(algebra, values=values, layout=layout)

    def __repr__(self):
        storage = "compact" if self.is_compact else "dense"
        return (
            f"Multivector(shape={self.shape}, storage={storage}, "
            f"algebra=Cl({self.algebra.p},{self.algebra.q},{self.algebra.r}))"
        )

    @property
    def tensor(self) -> torch.Tensor:
        """Dense coefficient tensor.

        This property is an explicit dense boundary. Planned paths that operate
        on compact data should use ``values`` or ``coefficients`` directly.
        """
        if self._tensor is not None:
            return self._tensor
        # Do not call this inside core operations that can preserve compact
        # ``values`` and ``layout``; materialization belongs at API boundaries.
        return materialize_dense(self.algebra, self)

    @tensor.setter
    def tensor(self, value: torch.Tensor) -> None:
        self._tensor = value
        self.values = None
        self.layout = None

    @property
    def is_compact(self) -> bool:
        """Whether this multivector stores compact grade lanes."""
        return self.layout is not None

    @property
    def is_dense(self) -> bool:
        """Whether this multivector stores dense coefficients."""
        return not self.is_compact

    @property
    def grades(self) -> tuple[int, ...] | None:
        """Active grades when compact metadata is available."""
        return None if self.layout is None else self.layout.grades

    @property
    def lane_count(self) -> int:
        """Number of stored coefficient lanes."""
        return self.coefficients.shape[-1]

    @property
    def storage(self) -> str:
        """Storage mode name."""
        return "compact" if self.is_compact else "dense"

    @property
    def coefficients(self) -> torch.Tensor:
        """Return the active storage tensor without dense materialization."""
        return self.values if self.is_compact else self._tensor

    def dense(self) -> Multivector:
        """Return a dense-storage multivector."""
        return Multivector(self.algebra, self.tensor)

    def compact(self, grades) -> Multivector:
        """Return a compact-storage multivector containing ``grades``."""
        layout = self.algebra.layout(grades)
        return self.with_layout(layout)

    def with_layout(self, layout: GradeLayout) -> Multivector:
        """Return this multivector represented by ``layout``."""
        self._check_layout(layout)
        if self.layout == layout:
            return Multivector(self.algebra, values=self.values, layout=layout)
        if self.is_compact:
            return Multivector(self.algebra, values=layout.convert(self.values, self.layout), layout=layout)
        return Multivector(self.algebra, values=layout.compact(self.tensor), layout=layout)

    def _check_layout(self, layout: GradeLayout) -> None:
        spec = layout.spec
        if (spec.p, spec.q, spec.r) != (self.algebra.p, self.algebra.q, self.algebra.r):
            raise ValueError(
                f"Layout mismatch: Cl({spec.p},{spec.q},{spec.r}) vs "
                f"Cl({self.algebra.p},{self.algebra.q},{self.algebra.r})"
            )

    def _check_algebra(self, other: Multivector) -> None:
        s, o = self.algebra, other.algebra
        if (s.p, s.q, s.r) != (o.p, o.q, o.r):
            raise ValueError(f"Algebra mismatch: Cl({s.p},{s.q},{s.r}) vs Cl({o.p},{o.q},{o.r})")

    def _wrap(self, tensor: torch.Tensor) -> Multivector:
        return Multivector(self.algebra, tensor)

    def _wrap_compact(self, values: torch.Tensor, layout: GradeLayout) -> Multivector:
        return Multivector(self.algebra, values=values, layout=layout)

    def _values_for_layout(self, layout: GradeLayout) -> torch.Tensor:
        self._check_layout(layout)
        if self.is_compact:
            return layout.convert(self.values, self.layout)
        return layout.compact(self.tensor)

    def _combined_layout(self, other: Multivector) -> GradeLayout:
        left = self.layout if self.is_compact else self.algebra.layout()
        right = other.layout if other.is_compact else other.algebra.layout()
        basis = set(left.basis_indices).union(right.basis_indices)
        grades = sorted({index.bit_count() for index in basis})
        return self.algebra.layout(grades)

    def __add__(self, other):
        if isinstance(other, Multivector):
            self._check_algebra(other)
            if self.is_compact or other.is_compact:
                layout = self._combined_layout(other)
                return self._wrap_compact(self._values_for_layout(layout) + other._values_for_layout(layout), layout)
            return self._wrap(self.tensor + other.tensor)
        if isinstance(other, (int, float, torch.Tensor)):
            if self.is_compact:
                return self._wrap_compact(self.values + other, self.layout)
            return self._wrap(self.tensor + other)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Multivector):
            self._check_algebra(other)
            if self.is_compact or other.is_compact:
                layout = self._combined_layout(other)
                return self._wrap_compact(self._values_for_layout(layout) - other._values_for_layout(layout), layout)
            return self._wrap(self.tensor - other.tensor)
        if isinstance(other, (int, float, torch.Tensor)):
            if self.is_compact:
                return self._wrap_compact(self.values - other, self.layout)
            return self._wrap(self.tensor - other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            if self.is_compact:
                return self._wrap_compact(other - self.values, self.layout)
            return self._wrap(other - self.tensor)
        return NotImplemented

    def __neg__(self):
        if self.is_compact:
            return self._wrap_compact(-self.values, self.layout)
        return self._wrap(-self.tensor)

    def __mul__(self, other):
        """Geometric product ``A * B``, or scalar scaling."""
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self.geometric_product(other)
        if isinstance(other, (int, float)):
            if self.is_compact:
                return self._wrap_compact(self.values * other, self.layout)
            return self._wrap(self.tensor * other)
        if isinstance(other, torch.Tensor):
            if self.is_compact:
                return self._wrap_compact(self.values * other, self.layout)
            return self._wrap(self.tensor * other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            if self.is_compact:
                return self._wrap_compact(self.values * other, self.layout)
            return self._wrap(self.tensor * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if self.is_compact:
                return self._wrap_compact(self.values / other, self.layout)
            return self._wrap(self.tensor / other)
        if isinstance(other, torch.Tensor):
            if self.is_compact:
                return self._wrap_compact(self.values / other, self.layout)
            return self._wrap(self.tensor / other)
        return NotImplemented

    def __xor__(self, other):
        """Wedge/exterior product ``A ^ B``."""
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self.wedge(other)
        return NotImplemented

    def __or__(self, other):
        """Inner product ``A | B``."""
        if isinstance(other, Multivector):
            self._check_algebra(other)
            return self.inner(other)
        return NotImplemented

    def __invert__(self):
        """Reversion ``~A``."""
        return self.reverse()

    def grade(self, k: int) -> Multivector:
        """Extract the grade-k component."""
        if self.is_compact:
            layout = self.algebra.layout((int(k),))
            if not self.layout.contains_grade(k):
                values = self.values.new_zeros(*self.values.shape[:-1], layout.dim)
                return self._wrap_compact(values, layout)
            values, output_layout = self.algebra.planned_unary(
                self.values,
                op="grade_projection",
                input_layout=self.layout,
                output_layout=layout,
                input_compact=True,
                compact_output=True,
                return_layout=True,
            )
            return self._wrap_compact(values, output_layout)
        return self._wrap(self.algebra.grade_projection(self.tensor, k))

    def reverse(self) -> Multivector:
        """Reversion (same as ``~self``)."""
        if self.is_compact:
            values, layout = self.algebra.planned_unary(
                self.values,
                op="reverse",
                input_layout=self.layout,
                input_compact=True,
                compact_output=True,
                return_layout=True,
            )
            return self._wrap_compact(values, layout)
        return self._wrap(self.algebra.reverse(self.tensor))

    def grade_involution(self) -> Multivector:
        """Grade involution (main involution): flips odd-grade signs."""
        if self.is_compact:
            values, layout = self.algebra.planned_unary(
                self.values,
                op="grade_involution",
                input_layout=self.layout,
                input_compact=True,
                compact_output=True,
                return_layout=True,
            )
            return self._wrap_compact(values, layout)
        return self._wrap(self.algebra.grade_involution(self.tensor))

    def clifford_conjugation(self) -> Multivector:
        """Clifford conjugation: grade_involution(reverse(x))."""
        if self.is_compact:
            values, layout = self.algebra.planned_unary(
                self.values,
                op="clifford_conjugation",
                input_layout=self.layout,
                input_compact=True,
                compact_output=True,
                return_layout=True,
            )
            return self._wrap_compact(values, layout)
        return self._wrap(self.algebra.clifford_conjugation(self.tensor))

    def dual(self) -> Multivector:
        """Hodge dual: maps grade-k to grade-(n-k)."""
        # Dense-only algebra APIs below intentionally cross the dense boundary.
        # Planned high-dimensional paths should use grade-declared primitives.
        return self._wrap(self.algebra.dual(self.tensor))

    def inverse(self) -> Multivector:
        """Blade inverse: B^{-1} = ~B / <B~B>_0."""
        return self._wrap(self.algebra.blade_inverse(self.tensor))

    def geometric_product(self, other: Multivector) -> Multivector:
        """Explicit geometric product (same as ``self * other``)."""
        self._check_algebra(other)
        if self.is_compact or other.is_compact:
            return self.projected_product(other, op="gp")
        return self._wrap(self.algebra.geometric_product(self.tensor, other.tensor))

    def projected_product(
        self,
        other: Multivector,
        *,
        output_grades=None,
        op: str = "gp",
        left_grades=None,
        right_grades=None,
    ) -> Multivector:
        """Grade-projected product using compact layouts when available."""
        self._check_algebra(other)
        left_layout = self.layout if self.is_compact else None
        right_layout = other.layout if other.is_compact else None
        left_grades = left_grades if left_grades is not None else _layout_grades(left_layout)
        right_grades = right_grades if right_grades is not None else _layout_grades(right_layout)

        values, layout = self.algebra.projected_product(
            self.values if self.is_compact else self.tensor,
            other.values if other.is_compact else other.tensor,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            left_layout=left_layout,
            right_layout=right_layout,
            op=op,
            left_compact=self.is_compact,
            right_compact=other.is_compact,
            return_layout=True,
        )
        return self._wrap_compact(values, layout)

    def wedge(self, other: Multivector) -> Multivector:
        """Wedge/exterior product (same as ``self ^ other``)."""
        self._check_algebra(other)
        if self.is_compact or other.is_compact:
            return self.projected_product(other, op="wedge")
        return self._wrap(self.algebra.wedge(self.tensor, other.tensor))

    def inner(self, other: Multivector) -> Multivector:
        """Inner product (same as ``self | other``)."""
        self._check_algebra(other)
        if self.is_compact or other.is_compact:
            return self.projected_product(other, op="inner")
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
        if self.is_compact or other.is_compact:
            return self.projected_product(other, op="commutator")
        return self._wrap(self.algebra.commutator(self.tensor, other.tensor))

    def anti_commutator(self, other: Multivector) -> Multivector:
        """Anti-commutator: ``{self, other} = self*other + other*self``."""
        self._check_algebra(other)
        if self.is_compact or other.is_compact:
            return self.projected_product(other, op="anti_commutator")
        return self._wrap(self.algebra.anti_commutator(self.tensor, other.tensor))

    def norm(self) -> torch.Tensor:
        """Induced metric norm (returns scalar tensor)."""
        from core.runtime.metric import induced_norm

        return induced_norm(self.algebra, self.tensor)

    def norm_sq(self) -> torch.Tensor:
        """Squared norm: <x * ~x>_0 (returns scalar tensor)."""
        return self.algebra.norm_sq(self.tensor)

    def get_grade_norms(self) -> torch.Tensor:
        """Per-grade L2 norms."""
        if self.is_compact:
            flat = self.values.pow(2).reshape(-1, self.layout.dim)
            grade_ids = self.layout.grade_indices_tensor(device=self.values.device).unsqueeze(0).expand_as(flat)
            result = flat.new_zeros(flat.shape[0], self.algebra.num_grades)
            result.scatter_add_(1, grade_ids, flat)
            return result.reshape(*self.values.shape[:-1], self.algebra.num_grades).clamp(min=self.algebra.eps).sqrt()
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
        if self.is_compact:
            return Multivector(self.algebra, values=self.values.to(*args, **kwargs), layout=self.layout)
        return self._wrap(self.tensor.to(*args, **kwargs))

    def detach(self) -> Multivector:
        """Detach from computation graph."""
        if self.is_compact:
            return Multivector(self.algebra, values=self.values.detach(), layout=self.layout)
        return self._wrap(self.tensor.detach())

    def clone(self) -> Multivector:
        """Clone the underlying tensor."""
        if self.is_compact:
            return Multivector(self.algebra, values=self.values.clone(), layout=self.layout)
        return self._wrap(self.tensor.clone())

    def requires_grad_(self, requires_grad: bool = True) -> Multivector:
        """Set requires_grad in-place."""
        if self.is_compact:
            self.values.requires_grad_(requires_grad)
        else:
            self.tensor.requires_grad_(requires_grad)
        return self

    @property
    def shape(self) -> torch.Size:
        return self.values.shape if self.is_compact else self.tensor.shape

    @property
    def device(self) -> torch.device:
        return self.values.device if self.is_compact else self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype if self.is_compact else self.tensor.dtype


def _layout_grades(layout: GradeLayout) -> tuple[int, ...] | None:
    return None if layout is None else layout.grades
