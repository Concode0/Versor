# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Layer-owned grade declaration helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from core.foundation.layout import GradeLayout
from core.foundation.module import AlgebraLike
from core.foundation.validation import VALIDATE, check_multivector


def resolve_layer_layout(algebra: AlgebraLike, grades: Optional[Iterable[int]]) -> Optional[GradeLayout]:
    """Return a compact layout for declared grades, or ``None`` for dense lanes."""
    if grades is None:
        return None
    return algebra.planner.layout(grades)


def lane_count(algebra: AlgebraLike, layout: Optional[GradeLayout]) -> int:
    """Return the active basis-lane count for a layer-owned layout."""
    return algebra.dim if layout is None else layout.dim


def check_multivector_lanes(
    values: torch.Tensor,
    algebra: AlgebraLike,
    layout: Optional[GradeLayout],
    name: str,
) -> None:
    """Validate dense or declared compact multivector lanes."""
    if layout is None:
        check_multivector(values, algebra, name)
        return
    if not VALIDATE:
        return
    assert values.ndim >= 1, f"{name}: expected ndim >= 1, got shape {tuple(values.shape)}"
    assert values.shape[-1] == layout.dim, (
        f"{name}: last dim should be {layout.dim} for grades {layout.grades}, "
        f"got {values.shape[-1]} (shape {tuple(values.shape)})"
    )
