# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Recursive Clifford algebra kernel for high-dimensional products.

The dense :class:`core.algebra.CliffordAlgebra` stores a full Cayley table and
is therefore practical only for small dimensions. ``PartitionedCliffordAlgebra``
keeps that dense implementation as the leaf kernel, while internal nodes factor
the basis into left and right sub-algebras:

``split_index = (left_index << right_n) | right_index``.

The public basis order is always the canonical bitmask order used by
``CliffordAlgebra``. Some recursive splits use a different internal bit order so
that repeated signature tiles, such as two copies of ``Cl(2,1,1)``, share a
single sub-algebra object. ``_BasisPermutation`` is the only place that converts
between public coefficients and that split-local coefficient order.
"""

import math
from dataclasses import dataclass
from math import gcd
from typing import Optional, Sequence

import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from core.validation import check_multivector

DEFAULT_PARTITION_LEAF_N = 6
MAX_PARTITIONED_DIMENSIONS = 16
_DEFAULT_PRODUCT_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _Signature:
    """Small immutable signature value used by split planning helpers."""

    p: int
    q: int
    r: int

    @property
    def n(self) -> int:
        return self.p + self.q + self.r

    def as_tuple(self) -> tuple[int, int, int]:
        return self.p, self.q, self.r

    def subtract(self, other: "_Signature") -> "_Signature":
        return _Signature(self.p - other.p, self.q - other.q, self.r - other.r)

    def scaled_tile(self, tile_count: int, selected_tiles: int) -> "_Signature":
        return _Signature(
            self.p // tile_count * selected_tiles,
            self.q // tile_count * selected_tiles,
            self.r // tile_count * selected_tiles,
        )


def _signature_for_range(p: int, q: int, r: int, start: int, width: int) -> tuple[int, int, int]:
    """Return ``(p, q, r)`` counts covered by a contiguous public bit range."""
    end = start + width
    p_count = max(0, min(end, p) - start)
    q_count = max(0, min(end, p + q) - max(start, p))
    r_count = max(0, min(end, p + q + r) - max(start, p + q))
    assert p_count + q_count + r_count == width
    return p_count, q_count, r_count


def _signature_gcd(p: int, q: int, r: int) -> int:
    """Return the greatest common divisor across nonzero signature blocks.

    A value larger than one means the signature can be represented as repeated
    copies of a smaller signature tile. Example: ``Cl(8,4,4)`` has gcd 4, so it
    can be split into repeated ``Cl(2,1,1)`` tiles.
    """
    counts = [count for count in (p, q, r) if count > 0]
    if not counts:
        return 1

    result = counts[0]
    for count in counts[1:]:
        result = gcd(result, count)
    return result


def _signature_prefix_dims_for_split(
    signature: _Signature,
    right_signature: _Signature,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return right/left public bit positions for a prefix-by-signature split."""
    right_dims, left_dims = _signature_prefix_dims(
        signature.p,
        signature.q,
        signature.r,
        right_signature.p,
        right_signature.q,
        right_signature.r,
    )
    return tuple(right_dims), tuple(left_dims)


def _signature_prefix_dims(
    p: int,
    q: int,
    r: int,
    p_count: int,
    q_count: int,
    r_count: int,
) -> tuple[list[int], list[int]]:
    """Return selected and remaining public bit positions by signature block.

    The selected bits are prefixes of each signature block: first positive
    dimensions, then negative dimensions, then null dimensions. This preserves
    the local ``(p, q, r)`` order inside a tiled child while still letting the
    child draw dimensions from non-contiguous public bit positions.
    """
    assert 0 <= p_count <= p
    assert 0 <= q_count <= q
    assert 0 <= r_count <= r

    selected = list(range(0, p_count)) + list(range(p, p + q_count)) + list(range(p + q, p + q + r_count))
    remaining = list(range(p_count, p)) + list(range(p + q_count, p + q)) + list(range(p + q + r_count, p + q + r))

    return selected, remaining


@dataclass(frozen=True)
class _PartitionSplit:
    """Child signatures and public bit positions for one recursive split.

    ``right_dims`` become the low split-local bits and ``left_dims`` become the
    high split-local bits. The order is important because recursive products use
    ``(left_index << right_n) | right_index`` after converting into split order.
    """

    right_signature: tuple[int, int, int]
    left_signature: tuple[int, int, int]
    right_dims: tuple[int, ...]
    left_dims: tuple[int, ...]

    @property
    def split_dims(self) -> tuple[int, ...]:
        """Return public bit positions in split-order bit layout."""
        return self.right_dims + self.left_dims


@dataclass(frozen=True)
class _ExpSettings:
    """Resolved exponential dispatch settings for one algebra signature."""

    regime: str
    policy: object
    fixed_iterations: int


@dataclass(frozen=True)
class _StructuralBuffers:
    """Purely generated buffers plus dtype-derived scalar tolerances."""

    buffers: tuple[tuple[str, torch.Tensor], ...]
    eps: float
    eps_sq: float


@dataclass(frozen=True)
class _ProductPlan:
    """Runtime product chunk plan for one recursive node."""

    right_pair_count: int
    chunk_size: int


@dataclass(frozen=True)
class _RightProductSlice:
    """Runtime routing tensors for one range of right-basis products."""

    right_a_indices: torch.Tensor
    right_b_indices: torch.Tensor
    right_result_indices: torch.Tensor
    right_product_signs: torch.Tensor

    def __iter__(self):
        """Preserve tuple-unpack compatibility for older private tests."""
        yield self.right_a_indices
        yield self.right_b_indices
        yield self.right_result_indices
        yield self.right_product_signs


@dataclass(frozen=True)
class _PartitionTreeSpec:
    """Validated relative split tree for one partitioned node."""

    right_dims: tuple[int, ...]
    left_dims: tuple[int, ...]
    right: Optional["_PartitionTreeSpec"] = None
    left: Optional["_PartitionTreeSpec"] = None

    @property
    def split_dims(self) -> tuple[int, ...]:
        return self.right_dims + self.left_dims

    def fingerprint(self) -> tuple:
        """Return a relative structural fingerprint for safe subalgebra reuse."""
        return (
            self.right_dims,
            None if self.right is None else self.right.fingerprint(),
            self.left_dims,
            None if self.left is None else self.left.fingerprint(),
        )


def _partition_split(p: int, q: int, r: int) -> _PartitionSplit:
    """Return the single recursive split used by the partitioned algebra.

    Repeated signature tiles are grouped first so common sub-algebras have the
    same local signature and can share one module instance. Signatures without
    repeatable tiles fall back to a balanced contiguous split.
    """
    signature = _Signature(p, q, r)
    tiled_split = _tiled_signature_split(signature)
    if tiled_split is not None:
        return tiled_split

    return _balanced_contiguous_split(signature)


def _tiled_signature_split(signature: _Signature) -> Optional[_PartitionSplit]:
    """Split repeated signature tiles so identical child trees can be shared."""
    tile_count = _signature_gcd(*signature.as_tuple())
    if tile_count <= 1:
        return None

    right_tile_count = tile_count // 2
    right_signature = signature.scaled_tile(tile_count, right_tile_count)
    left_signature = signature.subtract(right_signature)
    right_dims, left_dims = _signature_prefix_dims_for_split(signature, right_signature)

    return _build_partition_split(
        right_signature=right_signature,
        left_signature=left_signature,
        right_dims=right_dims,
        left_dims=left_dims,
    )


def _balanced_contiguous_split(signature: _Signature) -> _PartitionSplit:
    """Split a signature into balanced contiguous public bit ranges."""
    right_width = signature.n // 2
    left_width = signature.n - right_width
    right_signature = _Signature(*_signature_for_range(*signature.as_tuple(), 0, right_width))
    left_signature = _Signature(*_signature_for_range(*signature.as_tuple(), right_width, left_width))
    right_dims = tuple(range(right_width))
    left_dims = tuple(range(right_width, signature.n))

    return _build_partition_split(
        right_signature=right_signature,
        left_signature=left_signature,
        right_dims=right_dims,
        left_dims=left_dims,
    )


def _build_partition_split(
    *,
    right_signature: _Signature,
    left_signature: _Signature,
    right_dims: tuple[int, ...],
    left_dims: tuple[int, ...],
) -> _PartitionSplit:
    """Create a validated split plan."""
    assert right_signature.n == len(right_dims)
    assert left_signature.n == len(left_dims)
    return _PartitionSplit(
        right_signature=right_signature.as_tuple(),
        left_signature=left_signature.as_tuple(),
        right_dims=right_dims,
        left_dims=left_dims,
    )


def _partition_tree_fingerprint(partition_tree: Optional[_PartitionTreeSpec]) -> object:
    """Return the cache fingerprint for a child partition tree."""
    return "auto" if partition_tree is None else partition_tree.fingerprint()


def _split_from_tree_spec(p: int, q: int, r: int, tree: _PartitionTreeSpec) -> _PartitionSplit:
    """Create a concrete split from an explicit relative tree node."""
    assert sorted(tree.split_dims) == list(range(p + q + r))
    right_signature = _signature_for_dims(p, q, r, tree.right_dims)
    left_signature = _signature_for_dims(p, q, r, tree.left_dims)
    return _PartitionSplit(
        right_signature=right_signature,
        left_signature=left_signature,
        right_dims=tree.right_dims,
        left_dims=tree.left_dims,
    )


def _signature_for_dims(p: int, q: int, r: int, dims: Sequence[int]) -> tuple[int, int, int]:
    """Return child signature counts for selected local basis-vector dims."""
    n = p + q + r
    p_count = 0
    q_count = 0
    r_count = 0
    for dim in dims:
        if dim < 0 or dim >= n:
            raise ValueError(f"Partition dim {dim} is outside [0, {n})")
        if dim < p:
            p_count += 1
        elif dim < p + q:
            q_count += 1
        else:
            r_count += 1
    return p_count, q_count, r_count


def _normalize_partition_tree(
    partition_tree,
    p: int,
    q: int,
    r: int,
) -> Optional[_PartitionTreeSpec]:
    """Resolve a public partition tree expression into a relative tree spec."""
    if partition_tree is None:
        return None
    if isinstance(partition_tree, _PartitionTreeSpec):
        return partition_tree
    if isinstance(partition_tree, str):
        expression = partition_tree.strip()
        if not expression or expression.lower() in {"auto", "none", "null"}:
            return None
        assignments = _parse_partition_tree_expression(expression, p + q + r)
        root_dims = _canonical_global_dims(tuple(range(p + q + r)), p, q, r)
        return _build_partition_tree_from_assignments(assignments, (), root_dims, p, q, r)
    raise TypeError(
        "partition_tree must be None, 'auto', a path expression string, "
        f"or an internal _PartitionTreeSpec, got {type(partition_tree).__name__}"
    )


def _parse_partition_tree_expression(expression: str, n: int) -> dict[tuple[str, ...], tuple[int, ...]]:
    """Parse ``R=0-3; L.R=4-7`` style partition expressions."""
    assignments: dict[tuple[str, ...], tuple[int, ...]] = {}
    used_dims: dict[int, tuple[str, ...]] = {}

    for raw_entry in expression.split(";"):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid partition tree entry {entry!r}; expected PATH=DIMS")
        raw_path, raw_dims = entry.split("=", 1)
        path = _parse_partition_path(raw_path)
        dims = _parse_dim_expression(raw_dims, n)
        if path in assignments:
            raise ValueError(f"Duplicate partition path {'.'.join(path)}")
        for dim in dims:
            if dim in used_dims:
                previous = ".".join(used_dims[dim])
                current = ".".join(path)
                raise ValueError(f"Partition dim {dim} appears in both {previous} and {current}")
            used_dims[dim] = path
        assignments[path] = dims

    if not assignments:
        raise ValueError("Partition tree expression did not contain any PATH=DIMS entries")

    expected_dims = set(range(n))
    actual_dims = set(used_dims)
    if actual_dims != expected_dims:
        missing = sorted(expected_dims - actual_dims)
        extra = sorted(actual_dims - expected_dims)
        raise ValueError(f"Partition tree must cover every dimension exactly once; missing={missing}, extra={extra}")

    return assignments


def _parse_partition_path(raw_path: str) -> tuple[str, ...]:
    """Parse a tree path made of ``L`` and ``R`` segments."""
    path = tuple(segment.strip().upper() for segment in raw_path.strip().split(".") if segment.strip())
    if not path:
        raise ValueError("Partition path cannot be empty")
    invalid = [segment for segment in path if segment not in {"L", "R"}]
    if invalid:
        raise ValueError(f"Invalid partition path segment(s): {invalid}; use only L and R")
    return path


def _parse_dim_expression(raw_dims: str, n: int) -> tuple[int, ...]:
    """Parse comma-separated dimensions and inclusive ranges."""
    dims: list[int] = []
    for raw_token in raw_dims.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            raw_start, raw_end = token.split("-", 1)
            start = int(raw_start.strip())
            end = int(raw_end.strip())
            if end < start:
                raise ValueError(f"Invalid descending partition range {token!r}")
            dims.extend(range(start, end + 1))
        else:
            dims.append(int(token))

    if not dims:
        raise ValueError("Partition dimension expression cannot be empty")

    unique_dims = tuple(dict.fromkeys(dims))
    if len(unique_dims) != len(dims):
        raise ValueError(f"Partition dimension expression contains duplicates: {raw_dims!r}")
    for dim in unique_dims:
        if dim < 0 or dim >= n:
            raise ValueError(f"Partition dim {dim} is outside [0, {n})")
    return unique_dims


def _build_partition_tree_from_assignments(
    assignments: dict[tuple[str, ...], tuple[int, ...]],
    path: tuple[str, ...],
    node_global_dims: tuple[int, ...],
    p: int,
    q: int,
    r: int,
) -> _PartitionTreeSpec:
    """Build a relative tree spec for one node from global path assignments."""
    right_global_dims = _assigned_dims_under(assignments, path + ("R",))
    left_global_dims = _assigned_dims_under(assignments, path + ("L",))

    if not right_global_dims or not left_global_dims:
        label = "root" if not path else ".".join(path)
        raise ValueError(f"Partition node {label} must define both L and R children")

    node_dim_set = set(node_global_dims)
    if set(right_global_dims) | set(left_global_dims) != node_dim_set:
        label = "root" if not path else ".".join(path)
        raise ValueError(f"Partition node {label} children must cover exactly its parent dimensions")
    if set(right_global_dims) & set(left_global_dims):
        label = "root" if not path else ".".join(path)
        raise ValueError(f"Partition node {label} has overlapping L/R dimensions")

    right_child_global_dims = _canonical_global_dims(right_global_dims, p, q, r)
    left_child_global_dims = _canonical_global_dims(left_global_dims, p, q, r)
    local_index = {dim: index for index, dim in enumerate(node_global_dims)}
    right_dims = tuple(local_index[dim] for dim in right_child_global_dims)
    left_dims = tuple(local_index[dim] for dim in left_child_global_dims)

    right_tree = None
    left_tree = None
    if _has_descendant_assignment(assignments, path + ("R",)):
        right_tree = _build_partition_tree_from_assignments(
            assignments,
            path + ("R",),
            right_child_global_dims,
            p,
            q,
            r,
        )
    if _has_descendant_assignment(assignments, path + ("L",)):
        left_tree = _build_partition_tree_from_assignments(
            assignments,
            path + ("L",),
            left_child_global_dims,
            p,
            q,
            r,
        )

    return _PartitionTreeSpec(right_dims=right_dims, left_dims=left_dims, right=right_tree, left=left_tree)


def _assigned_dims_under(
    assignments: dict[tuple[str, ...], tuple[int, ...]],
    path: tuple[str, ...],
) -> tuple[int, ...]:
    """Return all globally assigned dims under a path."""
    dims: list[int] = []
    for assigned_path, assigned_dims in assignments.items():
        if assigned_path[: len(path)] == path:
            dims.extend(assigned_dims)
    return tuple(dims)


def _has_descendant_assignment(
    assignments: dict[tuple[str, ...], tuple[int, ...]],
    path: tuple[str, ...],
) -> bool:
    """Whether a path has deeper assignments than itself."""
    return any(len(assigned_path) > len(path) and assigned_path[: len(path)] == path for assigned_path in assignments)


def _canonical_global_dims(dims: Sequence[int], p: int, q: int, r: int) -> tuple[int, ...]:
    """Order global dimensions by local Clifford signature convention."""
    n = p + q + r
    dim_set = set(dims)
    return tuple(dim for dim in range(n) if dim in dim_set)


def _grade_index(n: int, device) -> torch.Tensor:
    """Return the grade, i.e. popcount, for basis indices ``0..2**n-1``."""
    basis_indices = torch.arange(2**n, dtype=torch.long, device=device)
    grades = torch.zeros_like(basis_indices)
    remaining_bits = basis_indices
    for _ in range(n):
        grades += remaining_bits & 1
        remaining_bits = remaining_bits >> 1
    return grades


def _bit_range_mask(start: int, end: int) -> int:
    """Return an integer bit mask covering ``[start, end)``."""
    mask = 0
    for bit in range(start, end):
        mask |= 1 << bit
    return mask


def _vector_square(bit: int, p: int, q: int) -> float:
    """Return the metric square of one basis vector."""
    if bit < p:
        return 1.0
    if bit < p + q:
        return -1.0
    return 0.0


def _resolve_exp_settings(
    p: int,
    q: int,
    r: int,
    dtype: torch.dtype,
    exp_policy,
    fixed_iterations: Optional[int],
) -> _ExpSettings:
    """Resolve signature-wide exponential policy without mutating a module."""
    if p == 0 or q == 0:
        regime = "elliptic"
    elif p == 1 and q == 1 and r == 0:
        regime = "hyperbolic"
    else:
        regime = "mixed"

    from core.decomposition import ExpPolicy, resolve_fixed_iterations

    policy = exp_policy if isinstance(exp_policy, ExpPolicy) else ExpPolicy(exp_policy)
    iterations = (
        int(fixed_iterations) if fixed_iterations is not None else resolve_fixed_iterations(policy, dtype, p + q + r)
    )
    return _ExpSettings(regime=regime, policy=policy, fixed_iterations=iterations)


def _default_product_chunk_size(pair_count: int) -> int:
    """Choose a memory-conscious right-pair chunk size."""
    return max(1, min(pair_count, _DEFAULT_PRODUCT_CHUNK_SIZE))


def _right_pair_count(right_n: int, right_r: int) -> int:
    """Return the number of right basis pairs that survive the null metric."""
    non_null_n = right_n - right_r
    return (4**non_null_n) * (3**right_r)


def _product_plan(right_n: int, right_r: int, requested_chunk_size: Optional[int]) -> _ProductPlan:
    """Return the recursive product plan without allocating routing tables."""
    right_pair_count = _right_pair_count(right_n, right_r)
    chunk_size = (
        _default_product_chunk_size(right_pair_count)
        if requested_chunk_size is None
        else max(1, int(requested_chunk_size))
    )
    return _ProductPlan(right_pair_count=right_pair_count, chunk_size=chunk_size)


def _product_pair_ranges(pair_count: int, chunk_size: int):
    """Yield static right-pair ranges for recursive product chunks."""
    for start in range(0, pair_count, chunk_size):
        yield start, min(start + chunk_size, pair_count)


def _compact_surviving_basis_pairs(
    compact_pair_indices: torch.Tensor,
    n: int,
    p: int,
    q: int,
    r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map compact mixed-radix pair indices to surviving basis-product pairs.

    Non-null basis-vector bits have four states: absent, left only, right only,
    or both. Null bits have only three active states because the ``both`` case
    squares a null vector and annihilates the product.
    """
    right_a_indices = torch.zeros_like(compact_pair_indices)
    right_b_indices = torch.zeros_like(compact_pair_indices)
    quotient = compact_pair_indices
    non_null_n = p + q

    for bit in range(n):
        if bit < non_null_n:
            digit = quotient.remainder(4)
            quotient = torch.div(quotient, 4, rounding_mode="floor")
            right_a_indices = right_a_indices | ((digit & 1) << bit)
            right_b_indices = right_b_indices | (((digit >> 1) & 1) << bit)
        else:
            digit = quotient.remainder(3)
            quotient = torch.div(quotient, 3, rounding_mode="floor")
            right_a_indices = right_a_indices | ((digit == 1).to(torch.long) << bit)
            right_b_indices = right_b_indices | ((digit == 2).to(torch.long) << bit)

    assert r == n - non_null_n
    return right_a_indices, right_b_indices


def _subalgebra_cache_key(
    p: int,
    q: int,
    r: int,
    device,
    dtype: torch.dtype,
    leaf_n: int,
    product_chunk_size: Optional[int],
    exp_policy,
    fixed_iterations: int,
    accumulation_dtype: Optional[torch.dtype],
    partition_tree: Optional[_PartitionTreeSpec],
) -> tuple:
    """Return the per-tree cache key for structurally identical sub-algebras."""
    return (
        p,
        q,
        r,
        str(torch.device(device)),
        str(dtype),
        leaf_n,
        product_chunk_size,
        getattr(exp_policy, "value", str(exp_policy)),
        fixed_iterations,
        str(accumulation_dtype),
        _partition_tree_fingerprint(partition_tree),
    )


def _basis_product_signs(
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    p: int,
    q: int,
    r: int,
    dtype: torch.dtype,
    popcount: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return basis-product signs for equal-shaped bitmask index tensors.

    The output is the scalar coefficient of ``e_indices_a * e_indices_b`` before
    the XOR result index is applied. Positive dimensions contribute ``+1``,
    negative dimensions contribute ``-1`` when repeated, and null dimensions
    annihilate products that repeat the same null basis vector.
    """
    n = p + q + r
    if popcount is None:
        popcount = _grade_index(n, indices_a.device)

    # Reordering ``A`` basis vectors past lower-numbered ``B`` basis vectors
    # gives the anticommutation sign.
    swap_counts = torch.zeros_like(indices_a)
    for bit in range(n):
        a_bit = (indices_a >> bit) & 1
        lower_bits = indices_b & ((1 << bit) - 1)
        swap_counts += a_bit * popcount[lower_bits]

    sign = torch.where(
        swap_counts % 2 == 0,
        torch.ones((), dtype=dtype, device=indices_a.device),
        -torch.ones((), dtype=dtype, device=indices_a.device),
    )

    # Repeated negative basis vectors square to -1.
    negative_mask = _bit_range_mask(p, p + q)
    negative_intersection = indices_a & indices_b & negative_mask
    negative_count = popcount[negative_intersection]
    sign = torch.where(negative_count % 2 == 0, sign, -sign)

    if r > 0:
        # Repeated null basis vectors square to 0, annihilating the term.
        null_mask = _bit_range_mask(p + q, n)
        sign = torch.where((indices_a & indices_b & null_mask) == 0, sign, torch.zeros_like(sign))

    return sign


def _involution_buffers(
    grade_index: torch.Tensor,
    dtype: torch.dtype,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reversion, grade-involution, and Clifford-conjugation signs."""
    rev_signs = ((-1.0) ** (grade_index * (grade_index - 1) // 2)).to(dtype=dtype)
    involution_signs = torch.where(
        grade_index % 2 == 0,
        torch.ones((), dtype=dtype, device=device),
        -torch.ones((), dtype=dtype, device=device),
    )
    conj_signs = (involution_signs * rev_signs).to(dtype=dtype)
    return rev_signs, involution_signs, conj_signs


def _basis_square_metric_signs(
    basis_indices: torch.Tensor,
    grade_index: torch.Tensor,
    p: int,
    q: int,
    r: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Return the metric-only part of ``e_I * e_I`` for every basis blade."""
    negative_mask = _bit_range_mask(p, p + q)
    negative_count = grade_index[basis_indices & negative_mask]
    metric_signs = torch.where(
        negative_count % 2 == 0,
        torch.ones((), dtype=dtype, device=device),
        -torch.ones((), dtype=dtype, device=device),
    )

    if r > 0:
        null_mask = _bit_range_mask(p + q, p + q + r)
        metric_signs = torch.where(
            (basis_indices & null_mask) == 0,
            metric_signs,
            torch.zeros_like(metric_signs),
        )

    return metric_signs


def _pseudoscalar_buffers(
    basis_indices: torch.Tensor,
    p: int,
    q: int,
    r: int,
    dtype: torch.dtype,
    popcount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source permutation and signs for right multiplication by ``I``."""
    pseudoscalar_index = basis_indices.numel() - 1
    ps_source = basis_indices ^ pseudoscalar_index
    ps_target = torch.full_like(ps_source, pseudoscalar_index)
    ps_signs = _basis_product_signs(ps_source, ps_target, p, q, r, dtype, popcount=popcount)
    return ps_source, ps_signs


def _bivector_buffers(
    n: int,
    p: int,
    q: int,
    grade_index: torch.Tensor,
    dtype: torch.dtype,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bivector indices, squared scalars, and right-contraction action."""
    if n < 2:
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            torch.zeros(0, n, n, dtype=dtype, device=device),
        )

    bivector_indices = [blade_index for blade_index in range(1 << n) if blade_index.bit_count() == 2]
    bv_indices = torch.tensor(bivector_indices, dtype=torch.long, device=device)
    bv_sq_scalar = torch.zeros(len(bv_indices), dtype=dtype, device=device)
    rc_action = torch.zeros(len(bv_indices), n, n, dtype=dtype, device=device)

    for bivector_position, blade_index in enumerate(bv_indices.tolist()):
        active_bits = [bit for bit in range(n) if blade_index & (1 << bit)]
        if len(active_bits) != 2:
            continue
        first_bit, second_bit = active_bits
        first_square = _vector_square(first_bit, p, q)
        second_square = _vector_square(second_bit, p, q)
        bv_sq_scalar[bivector_position] = -first_square * second_square
        rc_action[bivector_position, first_bit, second_bit] = second_square
        rc_action[bivector_position, second_bit, first_bit] = -first_square

    return bv_indices, bv_sq_scalar, rc_action


def _left_contraction_grade_buffers(n: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact grade-pair dispatch vectors for left contraction."""
    grade_pairs = [
        (grade_a, grade_b, grade_b - grade_a) for grade_a in range(n + 1) for grade_b in range(grade_a, n + 1)
    ]
    lc_grade_a, lc_grade_b, lc_grade_result = zip(*grade_pairs)
    return (
        torch.tensor(lc_grade_a, dtype=torch.long, device=device),
        torch.tensor(lc_grade_b, dtype=torch.long, device=device),
        torch.tensor(lc_grade_result, dtype=torch.long, device=device),
    )


def _product_pair_weights(dtype: torch.dtype, device) -> torch.Tensor:
    """Return AB/BA linear-combination weights for common binary products."""
    return torch.tensor(
        [
            [0.5, -0.5],
            [0.5, 0.5],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def _structural_buffers(p: int, q: int, r: int, device, dtype: torch.dtype) -> _StructuralBuffers:
    """Build all linear-size structural tensors without mutating a module."""
    n = p + q + r
    dim = 2**n
    basis_indices = torch.arange(dim, dtype=torch.long, device=device)
    grade_index = _grade_index(n, device)
    grade_values = torch.arange(n + 1, dtype=torch.long, device=device)

    rev_signs, involution_signs, conj_signs = _involution_buffers(grade_index, dtype, device)
    metric_signs = _basis_square_metric_signs(basis_indices, grade_index, p, q, r, dtype, device)
    cayley_diag = rev_signs * metric_signs
    ps_source, ps_signs = _pseudoscalar_buffers(basis_indices, p, q, r, dtype, grade_index)
    bv_indices, bv_sq_scalar, rc_action = _bivector_buffers(n, p, q, grade_index, dtype, device)
    lc_grade_a, lc_grade_b, lc_grade_result = _left_contraction_grade_buffers(n, device)
    g1_indices = (1 << torch.arange(n, device=device)).long()

    finfo = torch.finfo(dtype)
    return _StructuralBuffers(
        buffers=(
            ("grade_index", grade_index),
            ("_grade_values", grade_values),
            ("rev_signs", rev_signs),
            ("_involution_signs", involution_signs),
            ("conj_signs", conj_signs),
            ("_cayley_diag", cayley_diag),
            ("_norm_sq_signs", (rev_signs * cayley_diag).clone()),
            ("_hermitian_signs", (conj_signs * cayley_diag).clone()),
            ("_ps_source", ps_source),
            ("_ps_signs", ps_signs),
            ("_bv_indices", bv_indices),
            ("bv_sq_scalar", bv_sq_scalar),
            ("rc_action", rc_action),
            ("_g1_indices", g1_indices),
            ("_lc_grade_a", lc_grade_a),
            ("_lc_grade_b", lc_grade_b),
            ("_lc_grade_result", lc_grade_result),
            ("_product_pair_weights", _product_pair_weights(dtype, device)),
        ),
        eps=float(finfo.eps),
        eps_sq=float(finfo.eps**2),
    )


class _BasisPermutation(nn.Module):
    """Convert coefficients between public canonical order and split-local order.

    ``split_dims[split_bit]`` tells which public basis-vector bit occupies that
    split-local bit position. For contiguous balanced splits this is the identity
    mapping and all buffers stay empty. For tiled splits, bit positions are
    permuted so identical child signatures can share sub-algebra modules.

    Coefficients need signs as well as index permutations. A basis blade stores
    an ordered wedge of basis vectors; permuting vector dimensions changes that
    orientation by ``(-1) ** inversion_count``.
    """

    def __init__(self, split_dims: Sequence[int], device):
        super().__init__()
        self.split_dims = tuple(split_dims)
        self.n = len(self.split_dims)
        self.dim = 2**self.n
        self.uses_permutation = self.split_dims != tuple(range(self.n))

        if not self.uses_permutation:
            # Keep the identity case allocation-free in the hot path. Empty
            # buffers preserve old introspection behavior and still move with
            # ``module.to(device)``.
            empty = torch.empty(0, dtype=torch.long, device=device)
            self.register_buffer("split_to_public", empty, persistent=False)
            self.register_buffer("public_to_split", empty, persistent=False)
            self.register_buffer("split_signs", torch.empty(0, dtype=torch.int8, device=device), persistent=False)
            return

        split_to_public_indices = []
        split_orientation_signs = []
        for split_index in range(self.dim):
            public_index, orientation_sign = self._split_basis_term(split_index)
            split_to_public_indices.append(public_index)
            split_orientation_signs.append(orientation_sign)

        split_to_public = torch.tensor(split_to_public_indices, dtype=torch.long, device=device)
        public_to_split = torch.empty_like(split_to_public)
        public_to_split[split_to_public] = torch.arange(self.dim, dtype=torch.long, device=device)
        split_signs = torch.tensor(split_orientation_signs, dtype=torch.int8, device=device)

        self.register_buffer("split_to_public", split_to_public, persistent=False)
        self.register_buffer("public_to_split", public_to_split, persistent=False)
        self.register_buffer("split_signs", split_signs, persistent=False)

    def _split_basis_term(self, split_index: int) -> tuple[int, int]:
        """Return ``(public_index, orientation_sign)`` for one split-order blade."""
        public_index = 0
        public_bits = []
        for split_bit, public_bit in enumerate(self.split_dims):
            if split_index & (1 << split_bit):
                public_index |= 1 << public_bit
                public_bits.append(public_bit)

        # ``public_bits`` are encountered in split-local order. Count how many
        # swaps are needed to rewrite the same blade in canonical public order.
        inversions = 0
        for i, public_i in enumerate(public_bits):
            for public_j in public_bits[i + 1 :]:
                if public_i > public_j:
                    inversions += 1

        sign = -1 if inversions % 2 else 1
        return public_index, sign

    def to_split_order(self, mv: torch.Tensor) -> torch.Tensor:
        """Convert public canonical coefficients to split-local coefficient order.

        ``split_to_public[k]`` is the public source index for split coefficient
        ``k``. Multiplying by ``split_signs[k]`` accounts for blade orientation.
        """
        if not self.uses_permutation:
            return mv
        signs = self.split_signs.to(dtype=mv.dtype)
        return torch.index_select(mv, -1, self.split_to_public) * signs

    def to_public_order(self, mv: torch.Tensor) -> torch.Tensor:
        """Convert split-local coefficients back to public canonical order."""
        if not self.uses_permutation:
            return mv
        signs = torch.index_select(self.split_signs, 0, self.public_to_split).to(dtype=mv.dtype)
        return torch.index_select(mv, -1, self.public_to_split) * signs


class PartitionedCliffordAlgebra(nn.Module):
    """Partitioned Clifford algebra kernel using recursive tensor products.

    The basis order matches :class:`core.algebra.CliffordAlgebra`: basis blades
    are indexed by bitmasks, with lower-numbered vector dimensions in lower
    bits. Recursive nodes use a binary split, with the low-order internal block
    on the right and the high-order internal block on the left:

    ``global_index = (left_index << right_n) | right_index``.

    With this layout, the bridge sign for basis factors is
    ``(-1) ** (grade(left_A) * grade(right_B))``. Local metric signs remain in
    the left and right subalgebras.

    ``accumulation_dtype`` can promote recursive product accumulation, e.g.
    fp32 inputs with fp64 intermediate sums, while returning the input dtype.

    The split algorithm keeps the public basis order canonical but permutes
    coefficients internally when repeated signature tiles such as ``Cl(2,1,1)``
    inside ``Cl(8,4,4)`` can share subalgebra modules.

    Args:
        p (int): Positive signature dimensions.
        q (int, optional): Negative signature dimensions. Defaults to 0.
        r (int, optional): Null signature dimensions. Defaults to 0.
        device (str or torch.device, optional): Device for generated buffers.
        dtype (torch.dtype, optional): Floating-point dtype for sign buffers and
            dense leaf algebras.
        leaf_n (int, optional): Maximum basis-vector count handled by local
            leaves. The default targets ``2**leaf_n == 64`` coefficients so
            deep-learning products use small dense kernels and indexed global
            merge routing.
        product_chunk_size (int, optional): Number of right-basis product pairs
            processed per recursive chunk. ``None`` chooses a memory-conscious
            default from the node shape.
        exp_policy (str or ExpPolicy, optional): Bivector exponential policy.
        fixed_iterations (int, optional): Fixed iteration budget for decomposed
            exponential paths. ``None`` derives it from policy, dtype, and n.
        accumulation_dtype (torch.dtype, optional): Optional promoted dtype for
            recursive product accumulation.
        partition_tree (str, optional): Explicit split expression such as
            ``"R=0-3; L.R=4-7; L.L=8-11"``. ``None`` or ``"auto"`` uses the
            automatic repeated-tile/balanced splitter.
    """

    def __init__(
        self,
        p: int,
        q: int = 0,
        r: int = 0,
        device="cuda",
        dtype: torch.dtype = torch.float32,
        leaf_n: int = DEFAULT_PARTITION_LEAF_N,
        product_chunk_size: Optional[int] = None,
        exp_policy: str = "balanced",
        fixed_iterations: Optional[int] = None,
        accumulation_dtype: Optional[torch.dtype] = None,
        partition_tree=None,
        _subalgebra_cache: Optional[dict] = None,
    ):
        super().__init__()

        assert p >= 0, f"p must be non-negative, got {p}"
        assert q >= 0, f"q must be non-negative, got {q}"
        assert r >= 0, f"r must be non-negative, got {r}"
        assert p + q + r <= MAX_PARTITIONED_DIMENSIONS, (
            f"p + q + r must be <= {MAX_PARTITIONED_DIMENSIONS}, got {p + q + r}"
        )
        assert leaf_n >= 1, f"leaf_n must be >= 1, got {leaf_n}"

        self._init_signature(p, q, r, leaf_n, product_chunk_size, accumulation_dtype)
        self._init_exp_settings(dtype, exp_policy, fixed_iterations)
        self._init_structural_buffers(device, dtype)
        self._partition_tree = _normalize_partition_tree(partition_tree, self.p, self.q, self.r)

        if self.n <= leaf_n and self._partition_tree is None:
            self._init_leaf_node(device, dtype)
            return

        subalgebra_cache = {} if _subalgebra_cache is None else _subalgebra_cache
        self._init_recursive_node(device, dtype, subalgebra_cache)

    def _init_signature(
        self,
        p: int,
        q: int,
        r: int,
        leaf_n: int,
        product_chunk_size: Optional[int],
        accumulation_dtype: Optional[torch.dtype],
    ) -> None:
        """Store constructor inputs that define this algebra node."""
        self.p, self.q, self.r = p, q, r
        self.n = p + q + r
        self.dim = 2**self.n
        self.leaf_n = leaf_n
        self.product_chunk_size = product_chunk_size
        self.accumulation_dtype = accumulation_dtype

    def _init_exp_settings(self, dtype: torch.dtype, exp_policy, fixed_iterations: Optional[int]) -> None:
        """Attach resolved exponential settings to this node."""
        settings = _resolve_exp_settings(
            self.p,
            self.q,
            self.r,
            dtype,
            exp_policy,
            fixed_iterations,
        )
        self._exp_regime = settings.regime
        self._exp_policy = settings.policy
        self._exp_fixed_iterations = settings.fixed_iterations

    def _init_leaf_node(self, device, dtype: torch.dtype) -> None:
        """Configure a leaf node backed by the dense local Clifford kernel."""
        self.basis_permutation = _BasisPermutation(tuple(range(self.n)), device)
        self.core = CliffordAlgebra(
            self.p,
            self.q,
            self.r,
            device=device,
            dtype=dtype,
            exp_policy=self._exp_policy,
            fixed_iterations=self._exp_fixed_iterations,
        )
        self.left_sub = None
        self.right_sub = None
        self.left_n = 0
        self.right_n = 0
        self.left_dim = 0
        self.right_dim = 0
        self._right_pair_count = 0
        self._product_chunk_size = 0
        self._right_dims = ()
        self._left_dims = ()

    def _init_recursive_node(self, device, dtype: torch.dtype, subalgebra_cache: dict) -> None:
        """Configure split layout, child modules, and runtime product planning."""
        split = (
            _partition_split(self.p, self.q, self.r)
            if self._partition_tree is None
            else _split_from_tree_spec(self.p, self.q, self.r, self._partition_tree)
        )
        self._init_split_layout(split, device)
        self.core = None
        self.left_sub, self.right_sub = self._create_child_subalgebras(split, device, dtype, subalgebra_cache)
        self._init_product_plan()

    def _init_split_layout(self, split: _PartitionSplit, device) -> None:
        """Store recursive split shape and basis permutation."""
        assert sorted(split.split_dims) == list(range(self.n))

        self.left_n = len(split.left_dims)
        self.right_n = len(split.right_dims)
        self.left_dim = 2**self.left_n
        self.right_dim = 2**self.right_n
        self._right_dims = split.right_dims
        self._left_dims = split.left_dims
        self.basis_permutation = _BasisPermutation(split.split_dims, device)

    def _create_child_subalgebras(
        self,
        split: _PartitionSplit,
        device,
        dtype: torch.dtype,
        subalgebra_cache: dict,
    ) -> tuple["PartitionedCliffordAlgebra", "PartitionedCliffordAlgebra"]:
        """Return cached left and right child modules for a recursive node."""
        child_kwargs = {
            "device": device,
            "dtype": dtype,
            "leaf_n": self.leaf_n,
            "product_chunk_size": self.product_chunk_size,
            "exp_policy": self._exp_policy,
            "fixed_iterations": self._exp_fixed_iterations,
            "accumulation_dtype": self.accumulation_dtype,
            "subalgebra_cache": subalgebra_cache,
        }
        left_tree = None if self._partition_tree is None else self._partition_tree.left
        right_tree = None if self._partition_tree is None else self._partition_tree.right
        left_sub = self._get_or_create_subalgebra(*split.left_signature, partition_tree=left_tree, **child_kwargs)
        right_sub = self._get_or_create_subalgebra(*split.right_signature, partition_tree=right_tree, **child_kwargs)
        return left_sub, right_sub

    @classmethod
    def _get_or_create_subalgebra(
        cls,
        p: int,
        q: int,
        r: int,
        *,
        device,
        dtype: torch.dtype,
        leaf_n: int,
        product_chunk_size: Optional[int],
        exp_policy,
        fixed_iterations: int,
        accumulation_dtype: Optional[torch.dtype],
        partition_tree: Optional[_PartitionTreeSpec],
        subalgebra_cache: dict,
    ) -> "PartitionedCliffordAlgebra":
        """Create or reuse a child with the same algebraic structure.

        The cache is per root construction call, not global. That keeps module
        ownership local to one tree while still ensuring repeated logical
        sub-algebras, for example the left and right ``Cl(4,2,2)`` nodes inside
        ``Cl(8,4,4)``, point at the same Python module object.
        """
        cache_key = _subalgebra_cache_key(
            p,
            q,
            r,
            device,
            dtype,
            leaf_n,
            product_chunk_size,
            exp_policy,
            fixed_iterations,
            accumulation_dtype,
            partition_tree,
        )
        subalgebra = subalgebra_cache.get(cache_key)
        if subalgebra is None:
            subalgebra = cls(
                p,
                q,
                r,
                device=device,
                dtype=dtype,
                leaf_n=leaf_n,
                product_chunk_size=product_chunk_size,
                exp_policy=exp_policy,
                fixed_iterations=fixed_iterations,
                accumulation_dtype=accumulation_dtype,
                partition_tree=partition_tree,
                _subalgebra_cache=subalgebra_cache,
            )
            subalgebra_cache[cache_key] = subalgebra
        return subalgebra

    def _init_structural_buffers(self, device, dtype: torch.dtype) -> None:
        """Register linear-size structural tensors generated by pure builders."""
        structural = _structural_buffers(self.p, self.q, self.r, device, dtype)
        self._register_structural_buffers(structural.buffers)
        self.eps = structural.eps
        self.eps_sq = structural.eps_sq

    def _register_structural_buffers(self, buffers: tuple[tuple[str, torch.Tensor], ...]) -> None:
        """Attach generated tensors as non-persistent buffers."""
        for name, tensor in buffers:
            self.register_buffer(name, tensor, persistent=False)

    def _init_product_plan(self) -> None:
        """Initialize recursive product shape planning without baked routing.

        Right-pair indices, metric signs, and bridge signs are derived per
        product range at runtime. This keeps the
        recursive kernel from carrying signature-specific pair tables.
        """
        plan = _product_plan(self.right_n, self.right_sub.r, self.product_chunk_size)
        self._right_pair_count = plan.right_pair_count
        self._product_chunk_size = plan.chunk_size

    def _to_split_order(self, mv: torch.Tensor) -> torch.Tensor:
        """Convert public canonical coefficients to this node's split order."""
        return self.basis_permutation.to_split_order(mv)

    def _to_public_order(self, mv: torch.Tensor) -> torch.Tensor:
        """Convert split-order coefficients back to public canonical order."""
        return self.basis_permutation.to_public_order(mv)

    def _bridge_signs_for_right_b(self, right_b_indices: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Compute ``(-1) ** (grade(left_A) * grade(right_B))`` for a pair slice."""
        right_grades = torch.index_select(self.right_sub.grade_index, 0, right_b_indices).unsqueeze(1)
        left_grades = self.left_sub.grade_index.unsqueeze(0)
        signs = torch.where(
            (right_grades * left_grades) % 2 == 0,
            torch.ones((), dtype=dtype, device=right_b_indices.device),
            -torch.ones((), dtype=dtype, device=right_b_indices.device),
        )
        return signs

    def _right_product_signs(
        self,
        right_a_indices: torch.Tensor,
        right_b_indices: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute right-child basis-product signs for one runtime pair slice."""
        return _basis_product_signs(
            right_a_indices,
            right_b_indices,
            self.right_sub.p,
            self.right_sub.q,
            self.right_sub.r,
            dtype,
            popcount=self.right_sub.grade_index,
        )

    @property
    def device(self):
        """Return the device of the algebra buffers."""
        return self.grade_index.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype used by structural sign buffers."""
        return self.rev_signs.dtype

    def _apply(self, fn):
        """Propagate device/dtype moves and keep eps tolerances in sync."""
        result = super()._apply(fn)
        _finfo = torch.finfo(self.dtype)
        self.eps = float(_finfo.eps)
        self.eps_sq = float(_finfo.eps**2)
        return result

    @property
    def grade_masks(self):
        """Grade masks indexed by grade: ``grade_masks[k]`` -> ``[dim]`` bool."""
        return self.grade_index.unsqueeze(0) == self._grade_values.unsqueeze(1)

    @property
    def grade_masks_float(self):
        """Float grade masks indexed by grade: ``grade_masks_float[k]`` -> ``[dim]`` float."""
        return self.grade_masks.to(dtype=self.dtype)

    @property
    def is_leaf(self) -> bool:
        """Whether this node delegates directly to the monolithic atomic kernel."""
        return self.core is not None

    def describe_tree(self) -> str:
        """Return a readable split tree for debugging partition structure.

        The reported bit ranges use global basis-vector bit positions. Because
        structurally identical subalgebras are shared, repeated module objects
        are annotated with ``shared_with=<first_path>`` while still being shown
        at each logical tree position.
        """
        lines: list[str] = []
        seen: dict[int, str] = {}

        self._describe_tree_node(
            lines=lines,
            path="root",
            public_bits=tuple(range(self.n)),
            depth=0,
            seen=seen,
        )

        return "\n".join(lines)

    @staticmethod
    def _format_public_bits(public_bits: tuple[int, ...]) -> str:
        """Format global public bit positions compactly when contiguous."""
        if not public_bits:
            return "[]"

        start = public_bits[0]
        contiguous = tuple(range(start, start + len(public_bits)))
        if public_bits == contiguous:
            return f"[{start}, {start + len(public_bits)})"

        return "[" + ", ".join(str(bit) for bit in public_bits) + "]"

    def print_tree(self) -> None:
        """Print ``describe_tree()`` for interactive debugging."""
        print(self.describe_tree())

    def _describe_tree_node(
        self,
        *,
        lines: list[str],
        path: str,
        public_bits: tuple[int, ...],
        depth: int,
        seen: dict[int, str],
    ) -> None:
        """Append this node and children to a tree description."""
        indent = "  " * depth
        signature = f"Cl({self.p},{self.q},{self.r})"
        bits_text = self._format_public_bits(public_bits)

        node_id = id(self)
        shared_suffix = ""
        if node_id in seen:
            shared_suffix = f", shared_with={seen[node_id]}"
        else:
            seen[node_id] = path

        if self.core is not None:
            lines.append(
                f"{indent}{path}: {signature}, n={self.n}, dim={self.dim}, bits={bits_text}, leaf_core{shared_suffix}"
            )
            return

        right_bits = tuple(public_bits[bit] for bit in self._right_dims)
        left_bits = tuple(public_bits[bit] for bit in self._left_dims)
        right_bits_text = self._format_public_bits(right_bits)
        left_bits_text = self._format_public_bits(left_bits)

        lines.append(
            f"{indent}{path}: {signature}, n={self.n}, dim={self.dim}, "
            f"bits={bits_text}, split left={self.left_n} bits={left_bits_text}, right={self.right_n} "
            f"bits={right_bits_text}, pairs={self._right_pair_count}, "
            f"chunk={self._product_chunk_size}{shared_suffix}"
        )

        self.left_sub._describe_tree_node(
            lines=lines,
            path=f"{path}.L",
            public_bits=left_bits,
            depth=depth + 1,
            seen=seen,
        )
        self.right_sub._describe_tree_node(
            lines=lines,
            path=f"{path}.R",
            public_bits=right_bits,
            depth=depth + 1,
            seen=seen,
        )

    @property
    def exp_policy(self):
        """Active :class:`core.decomposition.ExpPolicy` controlling ``exp()`` dispatch."""
        return self._exp_policy

    @exp_policy.setter
    def exp_policy(self, value):
        from core.decomposition import ExpPolicy, resolve_fixed_iterations

        self._exp_policy = value if isinstance(value, ExpPolicy) else ExpPolicy(value)
        self._exp_fixed_iterations = resolve_fixed_iterations(self._exp_policy, self.dtype, self.n)
        if self.core is not None:
            self.core.exp_policy = self._exp_policy
        else:
            self.left_sub.exp_policy = self._exp_policy
            self.right_sub.exp_policy = self._exp_policy

    @property
    def num_grades(self) -> int:
        """Counts the number of grades."""
        return self.n + 1

    def embed_vector(self, vectors: torch.Tensor) -> torch.Tensor:
        """Inject vectors into the grade-1 subspace."""
        if self.core is not None:
            return self.core.embed_vector(vectors)
        mv = torch.zeros(*vectors.shape[:-1], self.dim, device=vectors.device, dtype=vectors.dtype)
        mv.scatter_(-1, self._g1_indices.expand_as(vectors), vectors)
        return mv

    def get_grade_norms(self, mv: torch.Tensor) -> torch.Tensor:
        """Calculate per-grade Euclidean coefficient norms."""
        if self.core is not None:
            return self.core.get_grade_norms(mv)
        check_multivector(mv, self, "get_grade_norms(mv)")
        batch_shape = mv.shape[:-1]
        sq = mv.pow(2)
        flat = sq.reshape(-1, self.dim)
        grade_index = self.grade_index.unsqueeze(0).expand_as(flat)
        result = torch.zeros(flat.shape[0], self.num_grades, device=mv.device, dtype=mv.dtype)
        result.scatter_add_(1, grade_index, flat)
        return result.reshape(*batch_shape, self.num_grades).clamp(min=self.eps).sqrt()

    def _combine_ab_ba(self, A: torch.Tensor, B: torch.Tensor, weight_index: int) -> torch.Tensor:
        """Compute a weighted combination of ``AB`` and ``BA`` in one recursive pass."""
        output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
        A = A.to(dtype=output_dtype)
        B = B.to(dtype=output_dtype)

        A_broadcast, B_broadcast = torch.broadcast_tensors(A, B)
        left_operands = torch.stack((A_broadcast, B_broadcast), dim=-2)
        right_operands = torch.stack((B_broadcast, A_broadcast), dim=-2)
        products = self.geometric_product(left_operands, right_operands)

        weights = self._product_pair_weights[weight_index]
        if weights.dtype != products.dtype:
            weights = weights.to(dtype=products.dtype)
        return torch.einsum("...pd,p->...d", products, weights)

    def _leaf_geometric_product(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        output_dtype: torch.dtype,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute a dense leaf product with the local dense kernel."""
        A_compute = A.to(dtype=compute_dtype)
        B_compute = B.to(dtype=compute_dtype)

        result = self.core.geometric_product(A_compute, B_compute)

        if result.dtype != output_dtype:
            result = result.to(dtype=output_dtype)
        return result

    def _right_blade_blocks(self, mv: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        """Return split-order coefficients grouped as right-indexed left multivectors."""
        split_order = self._to_split_order(mv.to(dtype=compute_dtype))
        by_left_then_right = split_order.reshape(*mv.shape[:-1], self.left_dim, self.right_dim)
        return by_left_then_right.transpose(-1, -2)

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute ``A * B`` through recursive tensor-product partitioning.

        For recursive nodes the public coefficient vector is first converted to
        split order, then reshaped into ``[..., right_dim, left_dim]``. Each
        right basis-pair selects two left-subalgebra multivectors, multiplies
        them recursively, and merges the result into the appropriate right block.
        """
        check_multivector(A, self, "geometric_product(A)")
        check_multivector(B, self, "geometric_product(B)")

        output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
        compute_dtype = self._geometric_product_compute_dtype(output_dtype)

        if self.core is not None:
            return self._leaf_geometric_product(A, B, output_dtype, compute_dtype)

        return self._recursive_geometric_product(A, B, output_dtype, compute_dtype)

    def _recursive_geometric_product(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        output_dtype: torch.dtype,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute a recursive node product after validation and dtype resolution."""
        A_by_right_blade = self._right_blade_blocks(A, compute_dtype)
        B_by_right_blade = self._right_blade_blocks(B, compute_dtype)
        result_blocks = self._accumulate_right_pair_chunks(A_by_right_blade, B_by_right_blade)
        output_shape = result_blocks.shape[:-2]
        result = result_blocks.reshape(*output_shape, self.dim)
        return self._to_public_order(result).to(dtype=output_dtype)

    def _accumulate_right_pair_chunks(
        self,
        A_by_right_blade: torch.Tensor,
        B_by_right_blade: torch.Tensor,
    ) -> torch.Tensor:
        """Accumulate sparse right-block interactions over static product chunks."""
        result_blocks = None
        for start, end in _product_pair_ranges(self._right_pair_count, self._product_chunk_size):
            chunk_blocks = self._geometric_product_pair_range(A_by_right_blade, B_by_right_blade, start, end)
            if result_blocks is None:
                result_blocks = chunk_blocks
            else:
                result_blocks = result_blocks + chunk_blocks

        assert result_blocks is not None
        return result_blocks

    def _geometric_product_pair_range(
        self,
        A_by_right_blade: torch.Tensor,
        B_by_right_blade: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Compute all contributions from a contiguous right-pair range."""
        product_slice = self._right_product_slice(start, end)

        if product_slice.right_product_signs.numel() == 0:
            batch_shape = torch.broadcast_shapes(A_by_right_blade.shape[:-2], B_by_right_blade.shape[:-2])
            return A_by_right_blade.new_zeros(*batch_shape, self.left_dim, self.right_dim)

        A_terms = torch.index_select(A_by_right_blade, -2, product_slice.right_a_indices)
        B_terms = torch.index_select(B_by_right_blade, -2, product_slice.right_b_indices)

        # ``bridge_signs[right_b, left_a]`` depends on the left basis index of
        # each selected A term, so broadcasting over the final left_dim axis
        # attaches the sign before the recursive left product.
        bridge_signs = self._bridge_signs_for_right_b(product_slice.right_b_indices, A_terms.dtype)
        if bridge_signs.dtype != A_terms.dtype:
            bridge_signs = bridge_signs.to(dtype=A_terms.dtype)
        A_terms = A_terms * bridge_signs

        left_products = self.left_sub.geometric_product(A_terms, B_terms)

        return self._merge_right_interactions(
            left_products,
            product_slice.right_result_indices,
            product_slice.right_product_signs,
        )

    def _merge_right_interactions(
        self,
        left_products: torch.Tensor,
        right_result_indices: torch.Tensor,
        right_product_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Merge left products into ``[..., left_dim, right_dim]`` result blocks."""
        if right_product_signs.numel() == 0:
            return left_products.new_zeros(
                *left_products.shape[:-2],
                self.left_dim,
                self.right_dim,
            )

        return self._merge_right_interactions_index_add(left_products, right_result_indices, right_product_signs)

    def _merge_right_interactions_index_add(
        self,
        left_products: torch.Tensor,
        right_result_indices: torch.Tensor,
        right_product_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Merge right-pair contributions with direct indexed accumulation."""
        signed_products = left_products.transpose(-1, -2) * right_product_signs.to(dtype=left_products.dtype)
        result_blocks = left_products.new_zeros(
            *left_products.shape[:-2],
            self.left_dim,
            self.right_dim,
        )
        result_blocks.index_add_(-1, right_result_indices, signed_products)
        return result_blocks

    def _geometric_product_compute_dtype(self, output_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype used for product accumulation."""
        if self.accumulation_dtype is None or not output_dtype.is_floating_point:
            return output_dtype
        return torch.promote_types(output_dtype, self.accumulation_dtype)

    def _promote_with_algebra_dtype(self, *dtypes: torch.dtype) -> torch.dtype:
        """Promote operand dtypes with the algebra's floating-point table dtype."""
        result = self.dtype
        for dtype in dtypes:
            result = torch.promote_types(result, dtype)
        return result

    def _right_product_slice(
        self,
        start: int,
        end: int,
    ) -> _RightProductSlice:
        """Derive right-block routing tensors for pair range ``[start, end)``.

        Returns:
            _RightProductSlice: Each position describes one right basis-pair
            contribution in the recursive product. The result still supports
            tuple unpacking for older private tests.
        """
        pair_indices = torch.arange(start, end, dtype=torch.long, device=self.device)
        if self.right_sub.r == 0:
            right_a_indices = torch.div(pair_indices, self.right_dim, rounding_mode="floor")
            right_b_indices = pair_indices.remainder(self.right_dim)
        else:
            right_a_indices, right_b_indices = _compact_surviving_basis_pairs(
                pair_indices,
                self.right_n,
                self.right_sub.p,
                self.right_sub.q,
                self.right_sub.r,
            )

        right_result_indices = right_a_indices ^ right_b_indices
        right_product_signs = self._right_product_signs(right_a_indices, right_b_indices, torch.int8)

        return _RightProductSlice(
            right_a_indices=right_a_indices,
            right_b_indices=right_b_indices,
            right_result_indices=right_result_indices,
            right_product_signs=right_product_signs,
        )

    def grade_projection(self, mv: torch.Tensor, grade: int) -> torch.Tensor:
        """Project a multivector onto a grade using the same mask contract as the core algebra."""
        if self.core is not None:
            return self.core.grade_projection(mv, grade)
        check_multivector(mv, self, "grade_projection(mv)")
        return mv * (self.grade_index == grade).to(dtype=mv.dtype)

    def reverse(self, mv: torch.Tensor) -> torch.Tensor:
        """Compute Clifford reversion."""
        if self.core is not None:
            return self.core.reverse(mv)
        check_multivector(mv, self, "reverse(mv)")
        return mv * self.rev_signs.to(dtype=mv.dtype)

    def wedge(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute ``(AB - BA) / 2`` through the partitioned product."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
            return self.core.wedge(A.to(dtype=output_dtype), B.to(dtype=output_dtype))
        return self._combine_ab_ba(A, B, 0)

    def right_contraction(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute the bivector-vector right contraction used by decomposition."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
            return self.core.right_contraction(A.to(dtype=output_dtype), B.to(dtype=output_dtype))
        check_multivector(A, self, "right_contraction(A)")
        check_multivector(B, self, "right_contraction(B)")

        output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
        bv_coeffs = torch.index_select(A, -1, self._bv_indices).to(dtype=output_dtype)
        v_coeffs = torch.index_select(B, -1, self._g1_indices).to(dtype=output_dtype)

        rc = self.rc_action.to(dtype=output_dtype)
        action = torch.einsum("...b, bij -> ...ij", bv_coeffs, rc)
        result_v = torch.matmul(action, v_coeffs.unsqueeze(-1)).squeeze(-1)

        result = result_v.new_zeros(*result_v.shape[:-1], self.dim)
        g1_idx_exp = self._g1_indices.expand(*result_v.shape[:-1], -1)
        result.scatter_(-1, g1_idx_exp, result_v)
        return result

    def inner_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute ``(AB + BA) / 2`` through the partitioned product."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
            return self.core.inner_product(A.to(dtype=output_dtype), B.to(dtype=output_dtype))
        return self._combine_ab_ba(A, B, 1)

    def commutator(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute the Lie bracket ``AB - BA``."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
            return self.core.commutator(A.to(dtype=output_dtype), B.to(dtype=output_dtype))
        return self._combine_ab_ba(A, B, 2)

    def anti_commutator(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute the anti-commutator ``AB + BA``."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
            return self.core.anti_commutator(A.to(dtype=output_dtype), B.to(dtype=output_dtype))
        return self._combine_ab_ba(A, B, 3)

    def blade_inverse(self, blade: torch.Tensor) -> torch.Tensor:
        """Compute the inverse of a non-degenerate blade."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(blade.dtype)
            return self.core.blade_inverse(blade.to(dtype=output_dtype))
        blade_rev = self.reverse(blade)
        blade_sq = self.geometric_product(blade, blade_rev)
        scalar = blade_sq[..., 0:1].clamp(min=self.eps_sq)
        return blade_rev / scalar

    def sandwich_product(self, R: torch.Tensor, x: torch.Tensor, R_rev: torch.Tensor = None) -> torch.Tensor:
        """Compute ``R x R~`` using recursive products."""
        if self.core is not None:
            dtypes = [R.dtype, x.dtype]
            if R_rev is not None:
                dtypes.append(R_rev.dtype)
            output_dtype = self._promote_with_algebra_dtype(*dtypes)
            R_rev = None if R_rev is None else R_rev.to(dtype=output_dtype)
            return self.core.sandwich_product(R.to(dtype=output_dtype), x.to(dtype=output_dtype), R_rev)
        if R_rev is None:
            R_rev = self.reverse(R)
        left = self.geometric_product(R.unsqueeze(-2), x)
        return self.geometric_product(left, R_rev.unsqueeze(-2))

    def per_channel_sandwich(self, R: torch.Tensor, x: torch.Tensor, R_rev: torch.Tensor = None) -> torch.Tensor:
        """Compute per-channel sandwich products using recursive products."""
        if self.core is not None:
            dtypes = [R.dtype, x.dtype]
            if R_rev is not None:
                dtypes.append(R_rev.dtype)
            output_dtype = self._promote_with_algebra_dtype(*dtypes)
            R_rev = None if R_rev is None else R_rev.to(dtype=output_dtype)
            return self.core.per_channel_sandwich(R.to(dtype=output_dtype), x.to(dtype=output_dtype), R_rev)
        if R_rev is None:
            R_rev = self.reverse(R)
        left = self.geometric_product(R.unsqueeze(0), x)
        return self.geometric_product(left, R_rev.unsqueeze(0))

    def multi_rotor_sandwich(self, R: torch.Tensor, x: torch.Tensor, R_rev: torch.Tensor = None) -> torch.Tensor:
        """Apply K rotors to ``[B, C, D]`` inputs using recursive products."""
        if self.core is not None:
            dtypes = [R.dtype, x.dtype]
            if R_rev is not None:
                dtypes.append(R_rev.dtype)
            output_dtype = self._promote_with_algebra_dtype(*dtypes)
            R_rev = None if R_rev is None else R_rev.to(dtype=output_dtype)
            return self.core.multi_rotor_sandwich(R.to(dtype=output_dtype), x.to(dtype=output_dtype), R_rev)
        if R_rev is None:
            R_rev = self.reverse(R)
        left = self.geometric_product(R.view(1, 1, *R.shape), x.unsqueeze(2))
        return self.geometric_product(left, R_rev.view(1, 1, *R_rev.shape))

    def pseudoscalar_product(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by the unit pseudoscalar using a static permutation/sign vector."""
        if self.core is not None:
            return self.core.pseudoscalar_product(x)
        check_multivector(x, self, "pseudoscalar_product(x)")
        return x[..., self._ps_source] * self._ps_signs.to(dtype=x.dtype)

    def blade_project(self, mv: torch.Tensor, blade: torch.Tensor) -> torch.Tensor:
        """Project a multivector onto a blade subspace."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(mv.dtype, blade.dtype)
            return self.core.blade_project(mv.to(dtype=output_dtype), blade.to(dtype=output_dtype))
        inner = self.inner_product(mv, blade)
        return self.geometric_product(inner, self.blade_inverse(blade))

    def blade_reject(self, mv: torch.Tensor, blade: torch.Tensor) -> torch.Tensor:
        """Reject a multivector from a blade subspace."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(mv.dtype, blade.dtype)
            return self.core.blade_reject(mv.to(dtype=output_dtype), blade.to(dtype=output_dtype))
        return mv - self.blade_project(mv, blade)

    def grade_involution(self, mv: torch.Tensor) -> torch.Tensor:
        """Apply the main involution."""
        if self.core is not None:
            return self.core.grade_involution(mv)
        check_multivector(mv, self, "grade_involution(mv)")
        return mv * self._involution_signs.to(dtype=mv.dtype)

    def clifford_conjugation(self, mv: torch.Tensor) -> torch.Tensor:
        """Apply Clifford conjugation."""
        if self.core is not None:
            return self.core.clifford_conjugation(mv)
        check_multivector(mv, self, "clifford_conjugation(mv)")
        return mv * self.conj_signs.to(dtype=mv.dtype)

    def norm_sq(self, mv: torch.Tensor) -> torch.Tensor:
        """Compute ``<x reverse(x)>_0`` using pre-merged static signs."""
        if self.core is not None:
            return self.core.norm_sq(mv)
        check_multivector(mv, self, "norm_sq(mv)")
        signs = self._norm_sq_signs
        if signs.dtype != mv.dtype:
            signs = signs.to(dtype=mv.dtype)
        return torch.matmul(mv * mv, signs.unsqueeze(-1))

    def left_contraction(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute left contraction by static grade-pair dispatch."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
            return self.core.left_contraction(A.to(dtype=output_dtype), B.to(dtype=output_dtype))
        check_multivector(A, self, "left_contraction(A)")
        check_multivector(B, self, "left_contraction(B)")

        output_dtype = self._promote_with_algebra_dtype(A.dtype, B.dtype)
        A = A.to(dtype=output_dtype)
        B = B.to(dtype=output_dtype)

        A_b, B_b = torch.broadcast_tensors(A, B)
        result = None
        pair_count = int(self._lc_grade_a.numel())
        chunk_size = max(
            1,
            min(
                pair_count,
                self._product_chunk_size if self._product_chunk_size > 0 else pair_count,
            ),
        )
        grade_index = self.grade_index.unsqueeze(0)
        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            a_masks = grade_index == self._lc_grade_a[start:end].unsqueeze(1)
            b_masks = grade_index == self._lc_grade_b[start:end].unsqueeze(1)
            result_masks = grade_index == self._lc_grade_result[start:end].unsqueeze(1)
            a_masks = a_masks.to(dtype=A_b.dtype)
            b_masks = b_masks.to(dtype=A_b.dtype)
            result_masks = result_masks.to(dtype=A_b.dtype)

            A_terms = A_b.unsqueeze(-2) * a_masks
            B_terms = B_b.unsqueeze(-2) * b_masks
            products = self.geometric_product(A_terms, B_terms)
            chunk = torch.einsum("...pd,pd->...d", products, result_masks)
            result = chunk if result is None else result + chunk
        return result

    def dual(self, mv: torch.Tensor) -> torch.Tensor:
        """Hodge dual alias for pseudoscalar multiplication."""
        if self.core is not None:
            return self.core.dual(mv)
        return self.pseudoscalar_product(mv)

    def reflect(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Reflect ``x`` through the hyperplane orthogonal to vector ``n``."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(x.dtype, n.dtype)
            return self.core.reflect(x.to(dtype=output_dtype), n.to(dtype=output_dtype))
        n_hat = self.grade_involution(n)
        n_inv = self.blade_inverse(n)
        if x.dim() == 3 and n.dim() == 2 and x.shape[0] != n.shape[0]:
            n_hat = n_hat.unsqueeze(0)
            n_inv = n_inv.unsqueeze(0)
        elif x.dim() == 3 and n.dim() == 2 and x.shape[0] == n.shape[0]:
            n_hat = n_hat.unsqueeze(1)
            n_inv = n_inv.unsqueeze(1)
        return self.geometric_product(self.geometric_product(n_hat, x), n_inv)

    def versor_product(self, V: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the general versor transformation ``hat(V) x V^{-1}``."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(V.dtype, x.dtype)
            return self.core.versor_product(V.to(dtype=output_dtype), x.to(dtype=output_dtype))
        V_inv = self.blade_inverse(V)
        V_hat = self.grade_involution(V)
        return self.geometric_product(self.geometric_product(V_hat, x), V_inv)

    def exp(self, mv: torch.Tensor) -> torch.Tensor:
        """Exponentiates a bivector to produce a rotor.

        Dispatch mirrors :class:`core.algebra.CliffordAlgebra`:

        - ``n <= 3`` -- every bivector is simple; closed-form is exact.
        - ``n >= 4`` -- compiled-safe decomposition; per-element selects
          closed-form vs decomposed via ``torch.where(simple)``.

        Args:
            mv (torch.Tensor): Pure bivector [..., dim].

        Returns:
            torch.Tensor: Rotor exp(mv) [..., dim].
        """
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(mv.dtype)
            return self.core.exp(mv.to(dtype=output_dtype))
        if self.n <= 3:
            return self._exp_bivector_closed(mv)
        return self._exp_compiled_safe(mv)

    def _exp_bivector_closed(self, B: torch.Tensor) -> torch.Tensor:
        """Closed-form exponential for simple bivectors in arbitrary signature.

        Uses zero geometric products. Exact for simple bivectors in any
        Clifford algebra Cl(p,q,r).
        """
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(B.dtype)
            return self.core._exp_bivector_closed(B.to(dtype=output_dtype))

        output_dtype = self._promote_with_algebra_dtype(B.dtype)
        B = B.to(dtype=output_dtype)

        bv_coeffs = torch.index_select(B, -1, self._bv_indices)
        bv_sq_scalar = self.bv_sq_scalar.to(dtype=output_dtype)

        # Signed squared norm: alpha = Sum_k b_k^2 . (e_k)^2
        # alpha < 0 -> elliptic (Euclidean-like), alpha > 0 -> hyperbolic
        alpha = torch.matmul(bv_coeffs * bv_coeffs, bv_sq_scalar.unsqueeze(-1))

        abs_alpha = alpha.abs().clamp(min=self.eps_sq)
        theta = torch.sqrt(abs_alpha)

        g0_mask = self.grade_masks_float[0].to(dtype=output_dtype)

        # Dispatch by signature regime (Python branch, no graph break)
        if self._exp_regime == "elliptic":
            # Pure Euclidean: alpha is always negative, only cos/sinc needed
            cos_theta = torch.cos(theta)
            sinc_theta = torch.where(
                theta > self.eps,
                torch.sin(theta) / theta,
                1.0 - abs_alpha / 6.0,
            )
            return cos_theta * g0_mask + sinc_theta * B

        if self._exp_regime == "hyperbolic":
            # Pure negative: alpha is always positive, only cosh/sinhc needed
            cosh_theta = torch.cosh(theta)
            sinhc_theta = torch.where(
                theta > self.eps,
                torch.sinh(theta) / theta,
                1.0 + abs_alpha / 6.0,
            )
            return cosh_theta * g0_mask + sinhc_theta * B

        # Mixed signature: need both branches + runtime select
        cos_theta = torch.cos(theta)
        sinc_theta = torch.where(
            theta > self.eps,
            torch.sin(theta) / theta,
            1.0 - abs_alpha / 6.0,
        )
        cosh_theta = torch.cosh(theta)
        sinhc_theta = torch.where(
            theta > self.eps,
            torch.sinh(theta) / theta,
            1.0 + abs_alpha / 6.0,
        )

        is_elliptic = alpha < -self.eps_sq
        is_hyperbolic = alpha > self.eps_sq

        scalar_part = torch.where(
            is_elliptic,
            cos_theta,
            torch.where(is_hyperbolic, cosh_theta, torch.ones_like(theta)),
        )
        coeff_part = torch.where(
            is_elliptic,
            sinc_theta,
            torch.where(is_hyperbolic, sinhc_theta, torch.ones_like(theta)),
        )

        return scalar_part * g0_mask + coeff_part * B

    def _exp_compiled_safe(self, B: torch.Tensor) -> torch.Tensor:
        """Compiled-safe exponential using partitioned products.

        Runs both closed-form and decomposed paths, then selects per element
        via ``torch.where`` based on simplicity. Both paths are computed
        unconditionally so there is no data-dependent branching.
        """
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(B.dtype)
            return self.core._exp_compiled_safe(B.to(dtype=output_dtype))
        from core.decomposition import compiled_safe_decomposed_exp

        R_closed = self._exp_bivector_closed(B)
        R_decomposed = compiled_safe_decomposed_exp(
            self,
            B,
            fixed_iterations=self._exp_fixed_iterations,
        )

        BB = self.geometric_product(B, B)
        # Subtract scalar part, check if residual is negligible
        scalar_part = self.grade_projection(BB, 0)
        non_scalar_energy = (BB - scalar_part).norm(dim=-1, keepdim=True)
        is_simple = non_scalar_energy < self.eps * 100

        return torch.where(is_simple, R_closed, R_decomposed)

    def _exp_taylor(self, mv: torch.Tensor, order: int = 8) -> torch.Tensor:
        """Taylor series exponential with scaling-and-squaring."""
        if self.core is not None:
            output_dtype = self._promote_with_algebra_dtype(mv.dtype)
            return self.core._exp_taylor(mv.to(dtype=output_dtype), order=order)
        norm = mv.norm(dim=-1, keepdim=True)
        k = torch.ceil(torch.log2(torch.clamp(norm, min=1.0))).int()

        max_k = k.max().item()
        if max_k > 0:
            mv_scaled = mv / (2.0**max_k)
        else:
            mv_scaled = mv

        res = torch.zeros_like(mv)
        res[..., 0] = 1.0

        term = torch.zeros_like(mv)
        term[..., 0] = 1.0

        for i in range(1, order + 1):
            term = self.geometric_product(term, mv_scaled)
            res = res + term / math.factorial(i)

        if max_k > 0:
            for _ in range(int(max_k)):
                res = self.geometric_product(res, res)

        return res
