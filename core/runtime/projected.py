# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Shared projected-product facade for algebra hosts."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from core.foundation.layout import GradeLayout
from core.foundation.validation import check_multivector
from core.runtime.accessors import default_layout as _default_layout
from core.runtime.accessors import grade_indices as _grade_indices
from core.runtime.accessors import hermitian_signs as _hermitian_signs
from core.runtime.accessors import materialize_dense
from core.runtime.accessors import resolve_layout as _resolve_layout


class AlgebraRuntimeMixin:
    """Shared runtime protocol for dense kernels and planned contexts."""

    def layout(self, grades: Optional[Iterable[int]] = None) -> GradeLayout:
        """Return a compact grade layout or the algebra's default layout."""
        if grades is None:
            return self.default_layout()
        return self.planner.layout(grades)

    def default_layout(self) -> GradeLayout:
        """Return the default layout using the central fallback policy."""
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
        """Return canonical dense basis indices for ``grades``."""
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

    def projected_product(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        left_grades=None,
        right_grades=None,
        output_grades=None,
        left_layout=None,
        right_layout=None,
        output_layout=None,
        op: str = "gp",
        left_compact: bool = False,
        right_compact: bool = False,
        compact_output: bool = False,
        return_layout: bool = False,
    ):
        """Compute a declared grade-restricted product through a static executor."""
        left_layout = self._declared_layout(left_grades, left_layout)
        right_layout = self._declared_layout(right_grades, right_layout)
        if not left_compact and left_layout is not None and A.shape[-1] == left_layout.dim:
            left_compact = left_layout.dim != self.dim
        if not right_compact and right_layout is not None and B.shape[-1] == right_layout.dim:
            right_compact = right_layout.dim != self.dim
        if not left_compact:
            check_multivector(A, self, "projected_product(A)")
        if not right_compact:
            check_multivector(B, self, "projected_product(B)")

        request = self.planner.product_request(
            A,
            B,
            left_grades=left_grades,
            right_grades=right_grades,
            output_grades=output_grades,
            left_layout=left_layout,
            right_layout=right_layout,
            output_layout=output_layout,
            op=op,
            left_compact=left_compact,
            right_compact=right_compact,
        )
        executor = self.planner.product_executor_for_request(request)

        if request.left_compact or request.right_compact:
            A_values = A if request.left_compact else executor.left_layout.compact(A)
            B_values = B if request.right_compact else executor.right_layout.compact(B)
            values = executor.forward_compact(A_values, B_values)
        else:
            values = executor(A, B)

        if return_layout:
            return values, executor.output_layout
        if compact_output:
            return values
        return materialize_dense(self, values, layout=executor.output_layout)

    def projected_geometric_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected geometric product convenience wrapper."""
        return self.projected_product(A, B, op="gp", **kwargs)

    def projected_wedge(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected wedge product convenience wrapper."""
        return self.projected_product(A, B, op="wedge", **kwargs)

    def projected_inner_product(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected inner product convenience wrapper."""
        return self.projected_product(A, B, op="inner", **kwargs)

    def projected_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected commutator convenience wrapper."""
        return self.projected_product(A, B, op="commutator", **kwargs)

    def projected_anti_commutator(self, A: torch.Tensor, B: torch.Tensor, **kwargs):
        """Projected anti-commutator convenience wrapper."""
        return self.projected_product(A, B, op="anti_commutator", **kwargs)

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
        """Execute a unary operation through the shared static grade planner."""
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
        return materialize_dense(self, output, layout=executor.output_layout)

    def _declared_layout(self, grades, layout):
        if layout is not None:
            return layout
        if grades is not None:
            return self.layout(grades)
        default_grades = getattr(self, "_default_grades", None)
        if default_grades is None:
            return None
        return self.layout(default_grades)
