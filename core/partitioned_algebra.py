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

_MAX_LEFT_MATRIX_LEAF_N = 6


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


def _partition_split(p: int, q: int, r: int) -> _PartitionSplit:
    """Return the single recursive split used by the partitioned algebra.

    Repeated signature tiles are grouped first so common sub-algebras have the
    same local signature and can share one module instance. Signatures without
    repeatable tiles fall back to a balanced contiguous split.
    """
    n = p + q + r

    tile_count = _signature_gcd(p, q, r)
    if tile_count > 1:
        tile_p = p // tile_count
        tile_q = q // tile_count
        tile_r = r // tile_count
        right_tile_count = tile_count // 2

        right_p = tile_p * right_tile_count
        right_q = tile_q * right_tile_count
        right_r = tile_r * right_tile_count

        right_dims, left_dims = _signature_prefix_dims(
            p,
            q,
            r,
            right_p,
            right_q,
            right_r,
        )
        left_p = p - right_p
        left_q = q - right_q
        left_r = r - right_r
        return _PartitionSplit(
            right_signature=(right_p, right_q, right_r),
            left_signature=(left_p, left_q, left_r),
            right_dims=tuple(right_dims),
            left_dims=tuple(left_dims),
        )

    right_width = n // 2
    left_width = n - right_width
    right_p, right_q, right_r = _signature_for_range(p, q, r, 0, right_width)
    left_p, left_q, left_r = _signature_for_range(p, q, r, right_width, left_width)
    right_dims = tuple(range(right_width))
    left_dims = tuple(range(right_width, n))

    return _PartitionSplit(
        right_signature=(right_p, right_q, right_r),
        left_signature=(left_p, left_q, left_r),
        right_dims=right_dims,
        left_dims=left_dims,
    )


def _grade_index(n: int, device) -> torch.Tensor:
    """Return the grade, i.e. popcount, for basis indices ``0..2**n-1``."""
    basis_indices = torch.arange(2**n, dtype=torch.long, device=device)
    grades = torch.zeros_like(basis_indices)
    remaining_bits = basis_indices
    for _ in range(n):
        grades += remaining_bits & 1
        remaining_bits = remaining_bits >> 1
    return grades


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
    )


def _basis_product_signs(
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
    p: int,
    q: int,
    r: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return basis-product signs for equal-shaped bitmask index tensors.

    The output is the scalar coefficient of ``e_indices_a * e_indices_b`` before
    the XOR result index is applied. Positive dimensions contribute ``+1``,
    negative dimensions contribute ``-1`` when repeated, and null dimensions
    annihilate products that repeat the same null basis vector.
    """
    n = p + q + r
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
    negative_mask = 0
    for bit in range(p, p + q):
        negative_mask |= 1 << bit
    negative_intersection = indices_a & indices_b & negative_mask
    negative_count = popcount[negative_intersection]
    sign = torch.where(negative_count % 2 == 0, sign, -sign)

    if r > 0:
        # Repeated null basis vectors square to 0, annihilating the term.
        null_mask = 0
        for bit in range(p + q, n):
            null_mask |= 1 << bit
        sign = torch.where((indices_a & indices_b & null_mask) == 0, sign, torch.zeros_like(sign))

    return sign


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
        leaf_n (int, optional): Maximum dimension handled by dense
            ``CliffordAlgebra`` leaves.
        product_chunk_size (int, optional): Number of right-basis product pairs
            processed per recursive chunk. ``None`` chooses a memory-conscious
            default from the node shape.
        exp_policy (str or ExpPolicy, optional): Bivector exponential policy.
        fixed_iterations (int, optional): Fixed iteration budget for decomposed
            exponential paths. ``None`` derives it from policy, dtype, and n.
        accumulation_dtype (torch.dtype, optional): Optional promoted dtype for
            recursive product accumulation.
    """

    def __init__(
        self,
        p: int,
        q: int = 0,
        r: int = 0,
        device="cuda",
        dtype: torch.dtype = torch.float32,
        leaf_n: int = 6,
        product_chunk_size: Optional[int] = None,
        exp_policy: str = "balanced",
        fixed_iterations: Optional[int] = None,
        accumulation_dtype: Optional[torch.dtype] = None,
        _subalgebra_cache: Optional[dict] = None,
    ):
        super().__init__()

        assert p >= 0, f"p must be non-negative, got {p}"
        assert q >= 0, f"q must be non-negative, got {q}"
        assert r >= 0, f"r must be non-negative, got {r}"
        assert leaf_n >= 1, f"leaf_n must be >= 1, got {leaf_n}"

        self.p, self.q, self.r = p, q, r
        self.n = p + q + r
        self.dim = 2**self.n
        self.leaf_n = leaf_n
        self.product_chunk_size = product_chunk_size
        self.accumulation_dtype = accumulation_dtype

        # Exp regime: dispatch at init. The branch is signature-wide, so it can
        # remain a Python branch without causing data-dependent graph breaks.
        if p == 0 or q == 0:
            self._exp_regime = "elliptic"
        elif p == 1 and q == 1 and r == 0:
            self._exp_regime = "hyperbolic"
        else:
            self._exp_regime = "mixed"

        # Exp policy: controls the decomposition iteration budget used by
        # compiled-safe bivector exponentials.
        from core.decomposition import ExpPolicy, resolve_fixed_iterations

        self._exp_policy = exp_policy if isinstance(exp_policy, ExpPolicy) else ExpPolicy(exp_policy)

        self._exp_fixed_iterations: int = (
            int(fixed_iterations)
            if fixed_iterations is not None
            else resolve_fixed_iterations(self._exp_policy, dtype, self.n)
        )

        if _subalgebra_cache is None:
            _subalgebra_cache = {}

        grade_index = _grade_index(self.n, device)
        self.register_buffer("grade_index", grade_index, persistent=False)
        self.register_buffer(
            "_grade_values",
            torch.arange(self.n + 1, dtype=torch.long, device=device),
            persistent=False,
        )

        self._init_structural_buffers(device, dtype)

        # Leaf nodes delegate to the existing dense Cayley-table engine.
        if self.n <= leaf_n:
            # Leaves operate in public order and therefore use the identity
            # permutation. Keeping ``basis_permutation`` present on every node
            # lets product code treat leaves and recursive nodes uniformly.
            self.basis_permutation = _BasisPermutation(tuple(range(self.n)), device)
            self.core = CliffordAlgebra(
                p,
                q,
                r,
                device=device,
                dtype=dtype,
                exp_policy=self._exp_policy,
                fixed_iterations=self._exp_fixed_iterations,
            )
            self._init_leaf_product_buffers(device, dtype)
            self.left_sub = None
            self.right_sub = None
            self.left_n = 0
            self.right_n = 0
            self.left_dim = 0
            self.right_dim = 0
            self._right_pair_count = 0
            self._product_chunk_size = 0
            self._right_pair_full = False
            self._has_sparse_right_interaction = False
            self._right_dims = ()
            self._left_dims = ()
            return

        split = _partition_split(p, q, r)
        right_signature = split.right_signature
        left_signature = split.left_signature

        # The split order keeps right-child public coordinates in the low
        # internal bits and left-child public coordinates in the high bits:
        # split_index = (left_index << right_n) | right_index.
        split_order_dims = split.split_dims
        assert sorted(split_order_dims) == list(range(self.n))

        right_n = len(split.right_dims)
        left_n = len(split.left_dims)
        self.left_n = left_n
        self.right_n = right_n
        self.left_dim = 2**left_n
        self.right_dim = 2**right_n
        self._right_dims = split.right_dims
        self._left_dims = split.left_dims
        self.basis_permutation = _BasisPermutation(split_order_dims, device)

        self.core = None
        self.left_sub = self._get_or_create_subalgebra(
            *left_signature,
            device=device,
            dtype=dtype,
            leaf_n=leaf_n,
            product_chunk_size=product_chunk_size,
            exp_policy=self._exp_policy,
            fixed_iterations=self._exp_fixed_iterations,
            accumulation_dtype=accumulation_dtype,
            subalgebra_cache=_subalgebra_cache,
        )
        self.right_sub = self._get_or_create_subalgebra(
            *right_signature,
            device=device,
            dtype=dtype,
            leaf_n=leaf_n,
            product_chunk_size=product_chunk_size,
            exp_policy=self._exp_policy,
            fixed_iterations=self._exp_fixed_iterations,
            accumulation_dtype=accumulation_dtype,
            subalgebra_cache=_subalgebra_cache,
        )

        # A left factor from A must cross a right factor from B:
        # (L_A R_A)(L_B R_B) = (-1)^(grade(L_A) grade(R_B)) (L_A L_B)(R_A R_B).
        left_grade_by_index = _grade_index(left_n, device)
        right_grade_by_index = _grade_index(right_n, device)
        bridge_signs = torch.where(
            (right_grade_by_index.unsqueeze(1) * left_grade_by_index.unsqueeze(0)) % 2 == 0,
            torch.ones((), device=device, dtype=torch.int8),
            -torch.ones((), device=device, dtype=torch.int8),
        )
        self.register_buffer("bridge_signs", bridge_signs, persistent=False)

        self._init_product_buffers()

    def _init_leaf_product_buffers(self, device, dtype: torch.dtype) -> None:
        """Precompute a small left-multiplication tensor for dense leaf GP.

        ``CliffordAlgebra.geometric_product`` gathers ``B[..., cayley_indices]``.
        In partitioned products, leaves receive large leading dimensions from
        right-pair batching, so that gather became the dominant allocation in
        profiles. For small leaves we instead build
        ``left_gp_mats[i, j, k]`` such that:

        ``(basis_i * basis_j)`` contributes ``left_gp_mats[i, j, k]`` to
        output basis ``k``.

        Runtime then contracts ``A`` into a left-multiplication matrix and
        applies it to ``B``. The tensor is ``O(dim^3)``, so large user-forced
        leaves fall back to the dense core kernel rather than allocating an
        unreasonable table.
        """
        if self.n > _MAX_LEFT_MATRIX_LEAF_N:
            self.register_buffer("_leaf_left_gp_mats", torch.empty(0, dtype=dtype, device=device), persistent=False)
            return

        left_gp_mats = torch.zeros(self.dim, self.dim, self.dim, dtype=dtype, device=device)
        for left_index in range(self.dim):
            for output_index in range(self.dim):
                right_index = int(self.core.cayley_indices[left_index, output_index].item())
                left_gp_mats[left_index, right_index, output_index] = self.core.gp_signs[
                    left_index,
                    output_index,
                ]
        self.register_buffer("_leaf_left_gp_mats", left_gp_mats, persistent=False)

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
                _subalgebra_cache=subalgebra_cache,
            )
            subalgebra_cache[cache_key] = subalgebra
        return subalgebra

    def _init_structural_buffers(self, device, dtype: torch.dtype) -> None:
        """Precompute static tables that scale linearly in basis dimension.

        Recursive nodes intentionally avoid full ``[dim, dim]`` Cayley tables,
        but many unary operations need only a per-basis sign or index vector.
        These buffers mirror the dense ``CliffordAlgebra`` public contract while
        keeping memory usage ``O(dim)``.
        """
        # Reversion sign: reverse a grade-k blade by k(k-1)/2 swaps.
        rev_signs = ((-1.0) ** (self.grade_index * (self.grade_index - 1) // 2)).to(
            dtype=dtype,
        )

        # Main involution and Clifford conjugation are diagonal operations in
        # the canonical basis.
        involution_signs = torch.where(
            self.grade_index % 2 == 0,
            torch.ones((), dtype=dtype, device=device),
            -torch.ones((), dtype=dtype, device=device),
        )
        conj_signs = (involution_signs * rev_signs).to(dtype=dtype)

        self.register_buffer("rev_signs", rev_signs, persistent=False)
        self.register_buffer("_involution_signs", involution_signs, persistent=False)
        self.register_buffer("conj_signs", conj_signs, persistent=False)

        # ``cayley_diag[i]`` is the scalar sign of basis_i * reverse(basis_i).
        # It is enough to implement norm and Hermitian forms without a full
        # Cayley table.
        basis_indices = torch.arange(self.dim, dtype=torch.long, device=device)
        negative_mask = 0
        for bit in range(self.p, self.p + self.q):
            negative_mask |= 1 << bit
        negative_count = self.grade_index[basis_indices & negative_mask]
        metric_signs = torch.where(
            negative_count % 2 == 0,
            torch.ones((), dtype=dtype, device=device),
            -torch.ones((), dtype=dtype, device=device),
        )
        if self.r > 0:
            null_mask = 0
            for bit in range(self.p + self.q, self.n):
                null_mask |= 1 << bit
            metric_signs = torch.where(
                (basis_indices & null_mask) == 0,
                metric_signs,
                torch.zeros_like(metric_signs),
            )

        cayley_diag = rev_signs * metric_signs
        self.register_buffer("_cayley_diag", cayley_diag, persistent=False)
        self.register_buffer("_norm_sq_signs", (rev_signs * cayley_diag).clone(), persistent=False)
        self.register_buffer("_hermitian_signs", (conj_signs * cayley_diag).clone(), persistent=False)

        # Multiplication by the pseudoscalar is also a fixed permutation/sign
        # vector: x * I maps source basis ``i ^ I`` into target basis ``i``.
        pseudoscalar_index = self.dim - 1
        ps_source = basis_indices ^ pseudoscalar_index
        ps_target = torch.full_like(ps_source, pseudoscalar_index)
        ps_signs = _basis_product_signs(ps_source, ps_target, self.p, self.q, self.r, dtype)
        self.register_buffer("_ps_source", ps_source, persistent=False)
        self.register_buffer("_ps_signs", ps_signs, persistent=False)

        if self.n >= 2:
            bv_indices = (self.grade_index == 2).nonzero(as_tuple=False).squeeze(-1)
            bv_sq_scalar = torch.zeros(len(bv_indices), dtype=dtype, device=device)
            rc_action = torch.zeros(len(bv_indices), self.n, self.n, dtype=dtype, device=device)
            for bivector_position, blade_index in enumerate(bv_indices.tolist()):
                active_bits = [bit for bit in range(self.n) if blade_index & (1 << bit)]
                if len(active_bits) != 2:
                    continue
                first_bit, second_bit = active_bits
                first_square = self._vector_square(first_bit)
                second_square = self._vector_square(second_bit)
                bv_sq_scalar[bivector_position] = -first_square * second_square
                rc_action[bivector_position, first_bit, second_bit] = second_square
                rc_action[bivector_position, second_bit, first_bit] = -first_square
        else:
            bv_indices = torch.zeros(0, dtype=torch.long, device=device)
            bv_sq_scalar = torch.zeros(0, dtype=dtype, device=device)
            rc_action = torch.zeros(0, self.n, self.n, dtype=dtype, device=device)

        self.register_buffer("_bv_indices", bv_indices, persistent=False)
        self.register_buffer("bv_sq_scalar", bv_sq_scalar, persistent=False)
        self.register_buffer("rc_action", rc_action, persistent=False)

        g1_idx = (1 << torch.arange(self.n, device=device)).long()
        self.register_buffer("_g1_indices", g1_idx, persistent=False)

        # Left contraction keeps grade pairs (a, b) where a <= b and then
        # projects the product to grade b-a.
        lc_grade_a = []
        lc_grade_b = []
        lc_grade_result = []
        for grade_a in range(self.n + 1):
            for grade_b in range(grade_a, self.n + 1):
                lc_grade_a.append(grade_a)
                lc_grade_b.append(grade_b)
                lc_grade_result.append(grade_b - grade_a)

        self.register_buffer(
            "_lc_grade_a",
            torch.tensor(lc_grade_a, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_lc_grade_b",
            torch.tensor(lc_grade_b, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_lc_grade_result",
            torch.tensor(lc_grade_result, dtype=torch.long, device=device),
            persistent=False,
        )

        # Common products are linear combinations of AB and BA. Rows encode:
        # wedge, inner, commutator, anti-commutator.
        product_weights = torch.tensor(
            [
                [0.5, -0.5],
                [0.5, 0.5],
                [1.0, -1.0],
                [1.0, 1.0],
            ],
            dtype=dtype,
            device=device,
        )
        self.register_buffer("_product_pair_weights", product_weights, persistent=False)

        _finfo = torch.finfo(dtype)
        self.eps: float = float(_finfo.eps)
        self.eps_sq: float = float(_finfo.eps**2)

    def _init_product_buffers(self) -> None:
        """Precompute right-block product routing for recursive GP.

        A recursive product sums over all right-child basis products:

        ``(A_l,a * B_l,b)`` contributes to right result ``a ^ b`` with the
        right-subalgebra metric sign and the bridge sign. The left products are
        still computed recursively at runtime; these buffers only describe how
        to select right blocks and merge them back.
        """
        right_indices = torch.arange(self.right_dim, device=self.device)
        right_a_indices, right_b_indices = torch.meshgrid(right_indices, right_indices, indexing="ij")
        right_a_indices = right_a_indices.reshape(-1)
        right_b_indices = right_b_indices.reshape(-1)
        right_result_indices = (right_a_indices ^ right_b_indices).long()

        right_product_signs = _basis_product_signs(
            right_a_indices,
            right_b_indices,
            self.right_sub.p,
            self.right_sub.q,
            self.right_sub.r,
            torch.int8,
        )

        if self.right_sub.r == 0:
            # Non-degenerate right algebras have no zero products, so
            # pair_a/pair_b can be reconstructed from a linear range instead
            # of stored as two extra ``right_dim ** 2`` buffers.
            pair_count = self.right_dim * self.right_dim
            right_product_signs = right_product_signs.reshape(-1)
            self._right_pair_full = True
            self.register_buffer("_right_pair_signs", right_product_signs, persistent=False)
        else:
            # Degenerate signatures have repeated null factors that produce
            # zero. Store only nonzero right interactions so runtime never
            # computes left products that will be discarded.
            nonzero = right_product_signs != 0
            right_a_pair_indices = right_a_indices[nonzero].long()
            right_b_pair_indices = right_b_indices[nonzero].long()
            right_result_indices = right_result_indices[nonzero].long()
            right_product_signs = right_product_signs[nonzero]
            pair_count = int(right_a_pair_indices.numel())

            self._right_pair_full = False
            self.register_buffer("_right_pair_a", right_a_pair_indices, persistent=False)
            self.register_buffer("_right_pair_b", right_b_pair_indices, persistent=False)
            self.register_buffer("_right_pair_result", right_result_indices, persistent=False)
            self.register_buffer("_right_pair_signs", right_product_signs, persistent=False)

        self._right_pair_count = pair_count
        if self.product_chunk_size is None:
            # Full-vectorize shallow nodes. Deeper nodes default to chunks so a
            # high-dimensional product does not materialize all right-pair left
            # products at once.
            default_chunk = pair_count if self.left_n <= self.leaf_n else min(pair_count, 64)
            self._product_chunk_size = max(1, default_chunk)
        else:
            self._product_chunk_size = max(1, int(self.product_chunk_size))

        self._has_sparse_right_interaction = self._product_chunk_size >= self._right_pair_count
        if self._has_sparse_right_interaction:
            self._init_right_interaction_buffers(right_result_indices, right_product_signs)

    def _init_right_interaction_buffers(
        self,
        right_result_indices: torch.Tensor,
        right_product_signs: torch.Tensor,
    ) -> None:
        """Precompute sparse right-product routing from pair terms to result blocks."""
        pair_columns = torch.arange(self._right_pair_count, dtype=torch.long, device=self.device)
        interaction_indices = torch.stack((right_result_indices.long(), pair_columns))
        interaction = torch.sparse_coo_tensor(
            interaction_indices,
            right_product_signs.to(dtype=self.dtype),
            (self.right_dim, self._right_pair_count),
            device=self.device,
        ).coalesce()
        self.register_buffer("_right_interaction", interaction, persistent=False)

    @property
    def _uses_basis_permutation(self) -> bool:
        """Whether this node needs public/split basis conversion."""
        return self.basis_permutation.uses_permutation

    @property
    def _to_split_basis(self) -> torch.Tensor:
        """Public source indices for split-order coefficients."""
        return self.basis_permutation.split_to_public

    @property
    def _to_public_basis(self) -> torch.Tensor:
        """Split source indices for public-order coefficients."""
        return self.basis_permutation.public_to_split

    @property
    def _split_basis_signs(self) -> torch.Tensor:
        """Orientation signs indexed by split-order basis index."""
        return self.basis_permutation.split_signs

    def _to_split_order(self, mv: torch.Tensor) -> torch.Tensor:
        """Convert public canonical coefficients to this node's split order."""
        return self.basis_permutation.to_split_order(mv)

    def _to_public_order(self, mv: torch.Tensor) -> torch.Tensor:
        """Convert split-order coefficients back to public canonical order."""
        return self.basis_permutation.to_public_order(mv)

    def _vector_square(self, bit: int) -> float:
        """Return ``e_bit ** 2`` from the global signature."""
        if bit < self.p:
            return 1.0
        if bit < self.p + self.q:
            return -1.0
        return 0.0

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
        """Compute a dense leaf product with the profiled small-leaf kernel."""
        A_compute = A.to(dtype=compute_dtype)
        B_compute = B.to(dtype=compute_dtype)

        if self._leaf_left_gp_mats.numel() == 0:
            result = self.core.geometric_product(A_compute, B_compute)
        else:
            left_gp_mats = self._leaf_left_gp_mats
            if left_gp_mats.dtype != compute_dtype:
                left_gp_mats = left_gp_mats.to(dtype=compute_dtype)

            # left_matrices[..., j, k] = sum_i A[..., i] * sign(i,j,k).
            # Multiplying B as a row vector then gives result[..., k].
            left_matrices = torch.einsum("...i,ijk->...jk", A_compute, left_gp_mats)
            result = torch.matmul(B_compute.unsqueeze(-2), left_matrices).squeeze(-2)

        if result.dtype != output_dtype:
            result = result.to(dtype=output_dtype)
        return result

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

        # Store each right-block coefficient as a contiguous left-subalgebra
        # multivector, so recursive products receive cache-local dense leaves.
        A_split = self._to_split_order(A.to(dtype=compute_dtype))
        B_split = self._to_split_order(B.to(dtype=compute_dtype))

        A_by_left_then_right = A_split.reshape(
            *A.shape[:-1],
            self.left_dim,
            self.right_dim,
        )
        B_by_left_then_right = B_split.reshape(
            *B.shape[:-1],
            self.left_dim,
            self.right_dim,
        )
        A_by_right_blade = A_by_left_then_right.transpose(-1, -2).contiguous()
        B_by_right_blade = B_by_left_then_right.transpose(-1, -2).contiguous()

        if self._product_chunk_size >= self._right_pair_count:
            result_blocks = self._geometric_product_pair_range(
                A_by_right_blade,
                B_by_right_blade,
                0,
                self._right_pair_count,
            )
            output_shape = result_blocks.shape[:-2]
            result = result_blocks.reshape(*output_shape, self.dim)
            return self._to_public_order(result).to(dtype=output_dtype)

        result_blocks = None
        for start in range(0, self._right_pair_count, self._product_chunk_size):
            end = min(start + self._product_chunk_size, self._right_pair_count)
            chunk_blocks = self._geometric_product_pair_range(A_by_right_blade, B_by_right_blade, start, end)
            if result_blocks is None:
                result_blocks = chunk_blocks
            else:
                result_blocks.add_(chunk_blocks)

        output_shape = result_blocks.shape[:-2]
        result = result_blocks.reshape(*output_shape, self.dim)
        return self._to_public_order(result).to(dtype=output_dtype)

    def _geometric_product_pair_range(
        self,
        A_by_right_blade: torch.Tensor,
        B_by_right_blade: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Compute all contributions from a contiguous right-pair range."""
        (
            right_a_indices,
            right_b_indices,
            right_result_indices,
            right_product_signs,
        ) = self._right_product_slice(start, end)

        A_terms = torch.index_select(A_by_right_blade, -2, right_a_indices)
        B_terms = torch.index_select(B_by_right_blade, -2, right_b_indices)

        # ``bridge_signs[right_b, left_a]`` depends on the left basis index of
        # each selected A term, so broadcasting over the final left_dim axis
        # attaches the sign before the recursive left product.
        bridge_signs = torch.index_select(self.bridge_signs, 0, right_b_indices)
        if bridge_signs.dtype != A_terms.dtype:
            bridge_signs = bridge_signs.to(dtype=A_terms.dtype)
        A_terms = A_terms * bridge_signs

        left_products = self.left_sub.geometric_product(A_terms, B_terms)

        return self._merge_right_interactions(left_products, start, end, right_result_indices, right_product_signs)

    def _merge_right_interactions(
        self,
        left_products: torch.Tensor,
        start: int,
        end: int,
        right_result_indices: torch.Tensor,
        right_product_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Merge left products into ``[..., left_dim, right_dim]`` result blocks."""
        if not self._use_sparse_right_interaction(start, end):
            return self._merge_right_interactions_index_add(left_products, right_result_indices, right_product_signs)

        return self._merge_right_interactions_sparse(left_products)

    def _use_sparse_right_interaction(self, start: int, end: int) -> bool:
        """Return whether the static sparse interaction should handle this range."""
        return (
            self._has_sparse_right_interaction
            and self.device.type == "cuda"
            and start == 0
            and end == self._right_pair_count
        )

    def _merge_right_interactions_sparse(self, left_products: torch.Tensor) -> torch.Tensor:
        """Merge a full right-pair range with the baked sparse interaction matrix."""
        if not self._has_sparse_right_interaction:
            raise RuntimeError("sparse right interaction is only available for full-pair product nodes")
        pair_count = self._right_pair_count
        interaction = self._right_interaction_tensor(left_products.dtype)
        batch_shape = left_products.shape[:-2]
        # sparse.mm expects [right_dim, pair_count] @ [pair_count, batch*left_dim].
        flat_terms = left_products.transpose(-1, -2).reshape(-1, pair_count)
        merged = torch.sparse.mm(interaction, flat_terms.transpose(0, 1))
        return merged.transpose(0, 1).reshape(*batch_shape, self.left_dim, self.right_dim)

    def _right_interaction_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the full sparse interaction tensor in the requested dtype."""
        interaction = self._right_interaction
        return interaction if interaction.dtype == dtype else interaction.to(dtype=dtype)

    def _merge_right_interactions_index_add(
        self,
        left_products: torch.Tensor,
        right_result_indices: torch.Tensor,
        right_product_signs: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback merge for devices without sparse COO matmul support."""
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return right-block routing tensors for pair range ``[start, end)``.

        Returns:
            tuple: ``(right_a_indices, right_b_indices, right_result_indices,
            right_product_signs)``. Each position describes one right basis-pair
            contribution in the recursive product.
        """
        if self._right_pair_full:
            linear_pair_indices = torch.arange(start, end, dtype=torch.long, device=self.device)
            right_a_indices = torch.div(linear_pair_indices, self.right_dim, rounding_mode="floor")
            right_b_indices = linear_pair_indices.remainder(self.right_dim)
            right_result_indices = right_a_indices ^ right_b_indices
            right_product_signs = self._right_pair_signs.reshape(-1)[start:end]
            return right_a_indices, right_b_indices, right_result_indices, right_product_signs

        return (
            self._right_pair_a[start:end],
            self._right_pair_b[start:end],
            self._right_pair_result[start:end],
            self._right_pair_signs[start:end],
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
