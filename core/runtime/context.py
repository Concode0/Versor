# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Lightweight algebra context for planned high-dimensional execution."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from core.foundation.basis import normalize_grades
from core.foundation.device import resolve_device, resolve_dtype
from core.foundation.layout import AlgebraSpec, GradeLayout
from core.planning.planner import GradePlanner
from core.runtime.accessors import FULL_LAYOUT_MAX_N
from core.runtime.accessors import default_layout as _default_layout
from core.runtime.accessors import grade_indices as _grade_indices
from core.runtime.accessors import hermitian_signs as _hermitian_signs
from core.runtime.accessors import resolve_layout as _resolve_layout
from core.runtime.projected import ProjectedProductMixin


class AlgebraContext(ProjectedProductMixin):
    """Signature and planning host without dense Cayley-table materialization."""

    def __init__(
        self,
        p: int,
        q: int = 0,
        r: int = 0,
        *,
        device="cuda",
        dtype: torch.dtype = torch.float32,
        default_grades: Optional[Iterable[int]] = None,
        allow_full_layout_products: Optional[bool] = None,
    ):
        if p < 0 or q < 0 or r < 0:
            raise ValueError(f"signature counts must be non-negative, got Cl({p},{q},{r})")

        self.p = int(p)
        self.q = int(q)
        self.r = int(r)
        self.n = self.p + self.q + self.r
        self.dim = 1 << self.n
        self.num_grades = self.n + 1
        self.spec = AlgebraSpec(self.p, self.q, self.r)
        self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        self._dtype = resolve_dtype(dtype)
        requested_full_layout = self.n <= FULL_LAYOUT_MAX_N if allow_full_layout_products is None else bool(
            allow_full_layout_products
        )
        self.allow_full_layout_products = requested_full_layout and self.n <= FULL_LAYOUT_MAX_N
        self._default_grades = None if default_grades is None else normalize_grades(default_grades, self.n)
        self._default_layout: Optional[GradeLayout] = None
        self.planner = GradePlanner(self)
        self._sync_eps()

    @property
    def device(self):
        """Return the context device used for planned executor buffers."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the context floating-point dtype."""
        return self._dtype

    def layout(self, grades: Optional[Iterable[int]] = None) -> GradeLayout:
        """Return a compact layout, or the full default layout when omitted."""
        if grades is None:
            return self.default_layout()
        return self.planner.layout(grades)

    def default_layout(self) -> GradeLayout:
        """Return the default layout using the central full-layout fallback policy."""
        return _default_layout(self)

    def resolve_layout(
        self,
        *,
        layout: Optional[GradeLayout] = None,
        grades: Optional[Iterable[int]] = None,
        mv=None,
        allow_full: bool = True,
        warn_full: bool = True,
    ) -> GradeLayout:
        """Resolve static layout metadata for tensors or multivectors."""
        return _resolve_layout(
            self,
            layout=layout,
            grades=grades,
            mv=mv,
            allow_full=allow_full,
            warn_full=warn_full,
        )

    def grade_indices(self, grades: Iterable[int], *, device=None) -> torch.Tensor:
        """Return canonical dense basis indices for ``grades`` without dense tables."""
        return _grade_indices(self, grades, device=self.device if device is None else device)

    def hermitian_signs(
        self,
        layout: Optional[GradeLayout] = None,
        *,
        grades: Optional[Iterable[int]] = None,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return Hermitian signs for a dense or compact layout."""
        return _hermitian_signs(self, layout=layout, grades=grades, device=device, dtype=dtype)

    def bivector_squared_signs(self, *, device=None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Return ``(e_ab)^2`` signs in canonical grade-2 layout order."""
        return self.planner.bivector_squared_signs(
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        )

    def _apply(self, fn):
        """Apply a PyTorch module-style device/dtype transform to cached executors."""
        probe = fn(torch.empty((), device=self.device, dtype=self.dtype))
        self._device = probe.device
        if probe.dtype.is_floating_point:
            self._dtype = probe.dtype
        self._sync_eps()
        self.planner._apply(fn)
        return self

    def to(self, device=None, dtype=None):
        """Move the context and cached executors."""
        if device is not None:
            self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        if dtype is not None:
            self._dtype = resolve_dtype(dtype)
        self._sync_eps()
        self.planner.clear_cache()
        return self

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute a geometric product."""
        return self.projected_product(A, B, op="gp", **kwargs)

    def wedge(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute an exterior product."""
        return self.projected_product(A, B, op="wedge", **kwargs)

    def inner_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute the symmetric inner product route."""
        return self.projected_product(A, B, op="inner", **kwargs)

    def commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute a commutator product."""
        return self.projected_product(A, B, op="commutator", **kwargs)

    def anti_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """Plan and execute an anti-commutator product."""
        return self.projected_product(A, B, op="anti_commutator", **kwargs)

    def grade_projection(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        """Project a dense multivector tensor to one grade."""
        return self.planned_unary(mv, op="grade_projection", output_grades=(int(grade),))

    def embed_vector(self, vectors: torch.Tensor) -> torch.Tensor:
        """Embed grade-1 vector coordinates into dense multivector coefficients."""
        if vectors.shape[-1] != self.n:
            raise ValueError(f"vectors last dimension must be {self.n}, got {vectors.shape[-1]}")
        output = vectors.new_zeros(*vectors.shape[:-1], self.dim)
        basis_indices = [1 << bit for bit in range(self.n)]
        return output.index_copy(-1, torch.tensor(basis_indices, dtype=torch.long, device=vectors.device), vectors)

    def reverse(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Reverse dense or compact multivector coefficients."""
        return self.planned_unary(mv, op="reverse", **kwargs)

    def grade_involution(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply grade involution to dense or compact multivector coefficients."""
        return self.planned_unary(mv, op="grade_involution", **kwargs)

    def clifford_conjugation(self, mv: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply Clifford conjugation to dense or compact multivector coefficients."""
        return self.planned_unary(mv, op="clifford_conjugation", **kwargs)

    def planned_unary(
        self,
        values: torch.Tensor,
        *,
        op: str,
        input_grades=None,
        output_grades=None,
        input_layout: Optional[GradeLayout] = None,
        output_layout: Optional[GradeLayout] = None,
        input_compact: bool = False,
        compact_output: bool = False,
        return_layout: bool = False,
    ):
        """Execute a unary planned operation."""
        request = self.planner.unary_request(
            values,
            op=op,
            input_grades=input_grades,
            output_grades=output_grades,
            input_layout=input_layout,
            output_layout=output_layout,
            input_compact=input_compact,
        )
        executor = self.planner.unary_executor_for_request(request)
        output = executor.forward_compact(values) if request.input_compact else executor(values)

        if return_layout:
            return output, executor.output_layout
        if compact_output:
            return output
        return executor.output_layout.dense(output)

    def _sync_eps(self) -> None:
        finfo = torch.finfo(self.dtype)
        self.eps = float(finfo.eps)
        self.eps_sq = float(finfo.eps**2)
