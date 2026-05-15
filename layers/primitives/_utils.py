"""Shared implementation helpers for primitive layers."""

from __future__ import annotations

from typing import Iterable

import torch

from core.foundation.layout import GradeLayout
from core.planning.action import metric_self_signs


def require_positive_int(value: int, name: str) -> int:
    """Validate a positive integer layer dimension."""
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_choice(value: str, name: str, choices: Iterable[str]) -> str:
    """Validate a string option and return it unchanged."""
    options = tuple(choices)
    if value not in options:
        supported = ", ".join(repr(option) for option in options)
        raise ValueError(f"{name} must be one of {supported}, got {value!r}")
    return value


def grade_indices(algebra, grade: int, *, name: str = "grade") -> torch.Tensor:
    """Return dense basis indices for a grade with consistent errors."""
    grade = int(grade)
    if grade < 0 or grade >= algebra.num_grades:
        raise ValueError(f"{name} must be in [0, {algebra.num_grades - 1}], got {grade}")
    indices = algebra.grade_indices((grade,))
    if indices.numel() == 0:
        raise ValueError(f"{name}={grade} has no basis elements in this algebra")
    return indices


def resolve_layer_layout(algebra, *, layout: GradeLayout = None, grades=None) -> GradeLayout | None:
    """Resolve an optional primitive storage layout."""
    if layout is not None:
        spec = algebra.planner.spec
        if layout.spec != spec:
            raise ValueError(f"layout signature {layout.spec} does not match algebra signature {spec}")
        return layout
    if grades is not None:
        return algebra.layout(grades)
    default_grades = getattr(algebra, "_default_grades", None)
    if default_grades is not None:
        return algebra.layout(default_grades)
    return None


def lane_dim(algebra, layout: GradeLayout | None) -> int:
    """Return the active lane count for dense or compact storage."""
    return algebra.dim if layout is None else layout.dim


def is_compact_layout(algebra, layout: GradeLayout | None) -> bool:
    """Return whether a resolved layout is strictly compact."""
    return layout is not None and layout.dim != algebra.dim


def check_multivector_storage(
    x: torch.Tensor,
    algebra,
    *,
    channels: int,
    name: str,
    layout: GradeLayout | None = None,
    allow_dense: bool = True,
) -> bool:
    """Validate primitive input and return whether it is compact."""
    if x.ndim < 3:
        raise ValueError(f"{name}: expected ndim >= 3, got shape {tuple(x.shape)}")
    if x.shape[-2] != channels:
        raise ValueError(f"{name}: expected {channels} channels, got {x.shape[-2]} (shape {tuple(x.shape)})")

    if layout is not None and x.shape[-1] == layout.dim:
        return layout.dim != algebra.dim
    if allow_dense and x.shape[-1] == algebra.dim:
        return False

    expected = [str(algebra.dim)] if allow_dense else []
    if layout is not None:
        expected.insert(0, f"{layout.dim} for grades {layout.grades}")
    raise ValueError(f"{name}: last dim must be {' or '.join(expected)}, got {x.shape[-1]}")


def scalar_mask(algebra, layout: GradeLayout | None, *, device=None, dtype=None) -> torch.Tensor:
    """Return a scalar-lane mask for dense or compact storage."""
    if layout is None:
        mask = torch.zeros(algebra.dim, device=device, dtype=torch.float32 if dtype is None else dtype)
        mask[0] = 1.0
        return mask
    values = torch.tensor(
        [1.0 if index == 0 else 0.0 for index in layout.basis_indices],
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
    )
    return values


def grade_positions(layout: GradeLayout, grade: int) -> torch.Tensor:
    """Return compact positions for one grade within a layout."""
    positions = [position for position, index in enumerate(layout.basis_indices) if index.bit_count() == int(grade)]
    return torch.tensor(positions, dtype=torch.long)


def layout_metric_signs(layout: GradeLayout, *, device=None, dtype=None) -> torch.Tensor:
    """Return basis self-product signs for a layer layout."""
    return metric_self_signs(layout, device=device, dtype=dtype)


def compact_grade_norms(algebra, values: torch.Tensor, layout: GradeLayout) -> torch.Tensor:
    """Return per-grade coefficient norms for compact values."""
    flat = values.pow(2).reshape(-1, layout.dim)
    grade_ids = layout.grade_indices_tensor(device=values.device).unsqueeze(0).expand_as(flat)
    result = values.new_zeros(flat.shape[0], algebra.num_grades)
    result.scatter_add_(1, grade_ids, flat)
    return result.reshape(*values.shape[:-1], algebra.num_grades).clamp(min=algebra.eps).sqrt()


def dense_from_indices(coefficients: torch.Tensor, indices: torch.Tensor, dense_dim: int) -> torch.Tensor:
    """Scatter compact coefficients into dense multivector storage."""
    dense = coefficients.new_zeros(*coefficients.shape[:-1], dense_dim)
    index = indices.to(device=coefficients.device).expand(*coefficients.shape[:-1], -1)
    return dense.scatter(-1, index, coefficients)


def cache_matches(cache: tuple[torch.Tensor, ...] | None, reference: torch.Tensor) -> bool:
    """Return True when cached tensors can be reused for ``reference``."""
    if cache is None:
        return False
    return all(tensor.device == reference.device and tensor.dtype == reference.dtype for tensor in cache)


def channel_mix(in_channels: int, out_channels: int, *, normalize: bool) -> torch.Tensor:
    """Build a deterministic channel routing matrix [out_channels, in_channels].

    Compression assigns every input channel to one output bin. Expansion repeats
    input channels across output bins. The normalized form averages each output
    bin; the unnormalized form sums it.
    """
    in_channels = require_positive_int(in_channels, "in_channels")
    out_channels = require_positive_int(out_channels, "out_channels")
    mix = torch.zeros(out_channels, in_channels)

    if in_channels >= out_channels:
        source = torch.arange(in_channels)
        target = torch.div(source * out_channels, in_channels, rounding_mode="floor").clamp_max(out_channels - 1)
        mix[target, source] = 1.0
    else:
        target = torch.arange(out_channels)
        source = torch.div(target * in_channels, out_channels, rounding_mode="floor").clamp_max(in_channels - 1)
        mix[target, source] = 1.0

    if normalize:
        mix = mix / mix.sum(dim=1, keepdim=True).clamp_min(1.0)
    return mix


def pair_mean(ch2pair: torch.Tensor, num_pairs: int) -> torch.Tensor:
    """Build [num_pairs, in_channels] means from a channel-to-pair map."""
    mix = torch.zeros(num_pairs, ch2pair.numel())
    source = torch.arange(ch2pair.numel())
    mix[ch2pair, source] = 1.0
    return mix / mix.sum(dim=1, keepdim=True).clamp_min(1.0)
