# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""High-dimensional verification for partitioned Clifford algebra.

These tests avoid monolithic Cayley-table references. For n >= 12, the
reference is computed from axiomatic bitmask rules, sub-algebraic isomorphisms,
algebraic identities, or closed forms inside known two-dimensional subalgebras.

Bitmask convention used throughout this file:

* Basis blade ``e_I`` is stored at integer index ``I``.
* Bit ``i`` in ``I`` means vector basis element ``e_i`` participates in the
  blade. For example, ``0b0101`` represents ``e_0 e_2``.
* The geometric product result blade is ``index_a ^ index_b`` unless a repeated
  null basis vector annihilates the term.
* The product sign is the parity of swaps needed to move the right blade's
  vectors past the left blade's vectors, followed by metric signs from repeated
  negative basis vectors.

When adding high-dimensional tests, prefer constructing sparse multivectors as
``[(bitmask_index, coefficient), ...]`` and comparing against
``_sparse_product_reference``. This keeps the reference independent from both
the dense Cayley table and the partitioned implementation.
"""

import math

import pytest
import torch

from core.algebra import CliffordAlgebra
from core.partitioned_algebra import PartitionedCliffordAlgebra

pytestmark = pytest.mark.slow

DEVICE = "cpu"


def _signature_for_range_reference(p: int, q: int, r: int, start: int, width: int) -> tuple[int, int, int]:
    """Return the local signature for a contiguous range of public dimensions.

    Dimensions are ordered by signature block: positive ``[0, p)``, negative
    ``[p, p + q)``, then null ``[p + q, p + q + r)``. This helper is used when a
    test embeds a lower-dimensional algebra inside a contiguous slice of a
    larger algebra and needs to build the matching local reference algebra.
    """
    end = start + width
    p_count = max(0, min(end, p) - start)
    q_count = max(0, min(end, p + q) - max(start, p))
    r_count = max(0, min(end, p + q + r) - max(start, p + q))
    assert p_count + q_count + r_count == width
    return p_count, q_count, r_count


def _shift_index(index: int, offset: int) -> int:
    """Shift every set basis bit in ``index`` by ``offset`` public dimensions.

    Example: local blade ``0b101`` at offset 3 becomes global blade
    ``0b101000``. This is the bitmask equivalent of embedding a local blade
    into a higher-dimensional subspace without changing the local blade order.
    """
    shifted = 0
    bit = 0
    while index:
        if index & 1:
            shifted |= 1 << (bit + offset)
        index >>= 1
        bit += 1
    return shifted


def _make_orthonormal_subspace_basis(
    algebra: PartitionedCliffordAlgebra,
    frame: torch.Tensor,
) -> torch.Tensor:
    """Return embedded basis blades for an orthonormal frame.

    ``frame`` contains ``width`` orthonormal vectors in the ambient algebra's
    vector space. The returned matrix has one row for every local bitmask
    ``0..2**width-1``. Row ``basis_index`` is built by multiplying the selected
    frame vectors in increasing bit order, matching the canonical bitmask basis
    order used by ``CliffordAlgebra``.
    """
    width = frame.shape[-1]
    vector_indices = (1 << torch.arange(algebra.n, dtype=torch.long, device=frame.device)).long()
    vectors = torch.zeros(width, algebra.dim, dtype=frame.dtype, device=frame.device)
    vectors[:, vector_indices] = frame.transpose(0, 1)

    basis = torch.zeros(2**width, algebra.dim, dtype=frame.dtype, device=frame.device)
    basis[0, 0] = 1.0
    for basis_index in range(1, basis.shape[0]):
        blade = torch.zeros(1, algebra.dim, dtype=frame.dtype, device=frame.device)
        blade[0, 0] = 1.0
        for bit in range(width):
            if basis_index & (1 << bit):
                blade = algebra.geometric_product(blade, vectors[bit : bit + 1])
        basis[basis_index] = blade[0]
    return basis


def _basis_product_reference(index_a: int, index_b: int, p: int, q: int, r: int) -> tuple[int, float]:
    """Reference product for two canonical basis blades using bit operations.

    This is the smallest trusted oracle in the partitioned tests. It intentionally
    does not call either dense or partitioned algebra code.

    The algorithm mirrors the Clifford product rules:

    1. The result blade contains basis vectors that appear in exactly one input,
       so its index is ``index_a ^ index_b``.
    2. Every vector in ``index_a`` must move past lower-numbered vectors in
       ``index_b``. The parity of those swaps gives the exterior sign.
    3. Repeated negative basis vectors contribute an extra ``-1`` metric sign.
    4. Repeated null basis vectors make the entire product zero.
    """
    n = p + q + r
    swap_count = 0
    for bit in range(n):
        if index_a & (1 << bit):
            # Count vectors from B with lower public index than the current A
            # vector. Each such pair must be swapped once.
            swap_count += (index_b & ((1 << bit) - 1)).bit_count()

    sign = -1.0 if swap_count % 2 else 1.0

    # Negative dimensions occupy [p, p + q). Repeating one of those vectors
    # squares it to -1, so an odd number of repeated negative bits flips sign.
    negative_mask = sum(1 << bit for bit in range(p, p + q))
    if ((index_a & index_b & negative_mask).bit_count() % 2) == 1:
        sign = -sign

    # Null dimensions occupy [p + q, n). Repeating a null basis vector squares
    # it to zero, annihilating that basis-pair contribution.
    null_mask = sum(1 << bit for bit in range(p + q, n))
    if (index_a & index_b & null_mask) != 0:
        sign = 0.0

    return index_a ^ index_b, sign


def _partitioned_basis_product(algebra, index_a: int, index_b: int) -> tuple[int, float]:
    """Evaluate a basis product by walking the partition tree directly.

    This helper verifies the recursive routing logic without constructing full
    multivectors. At each internal node, public indices are first converted into
    split-local order if the node uses a repeated-tile permutation. The low
    ``right_n`` bits are handled by the right child, the remaining high bits by
    the left child, and the bridge sign accounts for commuting right-child
    factors across left-child factors.
    """
    if algebra.core is not None:
        # Leaves are intentionally backed by the dense local kernel. Reading the
        # leaf Cayley table here checks the partition traversal and merge signs,
        # not the local dense product itself.
        result_index = int(algebra.core.cayley_indices[index_a, index_b].item())
        sign = float(algebra.core.cayley_signs[index_a, index_b].item())
        return result_index, sign

    input_sign = 1.0
    if algebra.basis_permutation.uses_permutation:
        # Some signatures are internally reordered to share identical child
        # subalgebras. The permutation carries signs because changing vector
        # order changes canonical blade orientation.
        split_a = int(algebra.basis_permutation.public_to_split[index_a].item())
        split_b = int(algebra.basis_permutation.public_to_split[index_b].item())
        input_sign *= float(algebra.basis_permutation.split_signs[split_a].item())
        input_sign *= float(algebra.basis_permutation.split_signs[split_b].item())
        index_a = split_a
        index_b = split_b

    right_mask = algebra.right_dim - 1
    left_a, right_a = index_a >> algebra.right_n, index_a & right_mask
    left_b, right_b = index_b >> algebra.right_n, index_b & right_mask

    # Recurse independently in the child algebras. The child result indices are
    # then packed back into the split-local index layout.
    left_result, left_sign = _partitioned_basis_product(algebra.left_sub, left_a, left_b)
    right_result, right_sign = _partitioned_basis_product(algebra.right_sub, right_a, right_b)
    right_b_index = torch.tensor([right_b], dtype=torch.long, device=algebra.device)
    # Bridge sign = (-1) ** (grade(left_A) * grade(right_B)). It is the only
    # sign that couples the two child products at an internal node.
    bridge_sign = float(algebra._bridge_signs_for_right_b(right_b_index, torch.float64)[0, left_a].item())
    result_index = (left_result << algebra.right_n) | right_result
    sign = input_sign * left_sign * right_sign * bridge_sign

    if algebra.basis_permutation.uses_permutation:
        # Convert split-local result orientation back to public canonical order.
        sign *= float(algebra.basis_permutation.split_signs[result_index].item())
        result_index = int(algebra.basis_permutation.split_to_public[result_index].item())

    return result_index, sign


def _sparse_product_reference(
    entries_a: list[tuple[int, float]],
    entries_b: list[tuple[int, float]],
    p: int,
    q: int,
    r: int,
) -> dict[int, float]:
    """Sparse multivector product using only the bitmask basis oracle."""
    result = {}
    for index_a, value_a in entries_a:
        for index_b, value_b in entries_b:
            result_index, sign = _basis_product_reference(index_a, index_b, p, q, r)
            if sign == 0.0:
                continue
            result[result_index] = result.get(result_index, 0.0) + value_a * value_b * sign
    return result


def _signature_sweep_entries(p: int, q: int, r: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Build sparse inputs that exercise low, middle, high, and null bits.

    The exact coefficients are arbitrary but non-symmetric. The bit positions
    are chosen to cross split boundaries for balanced trees and to hit positive,
    negative, and null signature blocks when present. Reuse this helper for
    broad signature sweeps; add explicit hand-picked entries when a test needs a
    very specific blade interaction.
    """
    n = p + q + r
    entries_a = [
        # Scalar term: verifies scalar multiplication and accumulation into an
        # already-populated output blade.
        (0, 0.375),
        # Low plus middle bit: likely crosses left/right children in Cl12+.
        ((1 << 0) | (1 << (n // 2)), -0.5),
        # Two high bits: catches sign handling away from the low-order block.
        ((1 << (n - 3)) | (1 << (n - 1)), 0.875),
        # Low plus high bit: exercises bridge signs across the partition split.
        ((1 << 1) | (1 << (n - 2)), -1.125),
    ]
    entries_b = [
        (1 << 0, -0.75),
        ((1 << (n // 2)) | (1 << (n - 4)), 1.25),
        ((1 << (n - 1)) | (1 << 2), -0.625),
        ((1 << 1) | (1 << (n - 5)) | (1 << (n - 2)), 0.5),
    ]

    if r > 0:
        # Include repeated null-bit products. Any pair that repeats ``null_bit``
        # should vanish in the reference and in the partitioned kernel.
        null_bit = p + q
        entries_a.append(((1 << null_bit) | (1 << 1), 0.25))
        entries_b.append(((1 << null_bit) | (1 << 2), -1.5))

    return entries_a, entries_b


def _make_sparse_multivector(algebra: PartitionedCliffordAlgebra, entries, dtype: torch.dtype) -> torch.Tensor:
    """Materialize ``[(bitmask, coeff), ...]`` entries into a dense tensor."""
    mv = torch.zeros(1, algebra.dim, dtype=dtype)
    for index, value in entries:
        mv[0, index] += value
    return mv


def _make_expected_multivector(algebra: PartitionedCliffordAlgebra, entries, dtype: torch.dtype) -> torch.Tensor:
    """Materialize a sparse dictionary reference into a dense output tensor."""
    expected = torch.zeros(1, algebra.dim, dtype=dtype)
    for index, value in entries.items():
        expected[0, index] = value
    return expected


def _long_taylor_simple_bivector(theta: float, square: float, order: int = 80) -> tuple[float, float]:
    """Reference exp(theta B) for a simple bivector with known ``B**2`` scalar."""
    scalar = 0.0
    bivector = 0.0
    power = 1.0
    for k in range(order + 1):
        if k > 0:
            power *= theta
        if k % 2 == 0:
            scalar += power * (square ** (k // 2)) / math.factorial(k)
        else:
            bivector += power * (square ** ((k - 1) // 2)) / math.factorial(k)
    return scalar, bivector


def _canonical_basis_term(index: int, coefficient: float) -> tuple[int, float]:
    """Normalize zero coefficients so identity checks compare exact tuples."""
    if coefficient == 0.0:
        return 0, 0.0
    return index, coefficient


def _multiply_signed_basis(
    algebra: PartitionedCliffordAlgebra,
    left: tuple[int, float],
    right: tuple[int, float],
) -> tuple[int, float]:
    """Multiply two signed basis terms through the partition-tree basis oracle."""
    left_index, left_coeff = left
    right_index, right_coeff = right
    if left_coeff == 0.0 or right_coeff == 0.0:
        return 0, 0.0
    result_index, sign = _partitioned_basis_product(algebra, left_index, right_index)
    return _canonical_basis_term(result_index, left_coeff * right_coeff * sign)


def _reverse_sign(index: int) -> float:
    """Sign applied by reversion to a basis blade of grade ``popcount(index)``."""
    grade = index.bit_count()
    return -1.0 if (grade * (grade - 1) // 2) % 2 else 1.0


def _grade_involution_sign(index: int) -> float:
    """Sign applied by grade involution to a basis blade."""
    return -1.0 if index.bit_count() % 2 else 1.0


class TestPartitionedHighDimensionalVerification:
    def test_cl12_sparse_multivector_product_matches_direct_bitmask_reference(self):
        # Cl(8,3,1) is dense enough to exercise recursive routing, includes a
        # negative block and a null block, and still keeps the sparse reference
        # human-readable. The selected indices include low, middle, high, and
        # all-bits-set blades so sign and null handling are both visible.
        p, q, r = 8, 3, 1
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=6,
            product_chunk_size=32,
        )
        entries_a = [(0, 0.25), (3, -1.5), (257, 0.75), (2049, 2.0)]
        entries_b = [(1, -0.5), (384, 1.25), (1025, -2.0), (4095, 0.5)]

        A = torch.zeros(1, algebra.dim, dtype=torch.float64)
        B = torch.zeros(1, algebra.dim, dtype=torch.float64)
        for index, value in entries_a:
            A[0, index] = value
        for index, value in entries_b:
            B[0, index] = value

        actual = algebra.geometric_product(A, B)
        expected_sparse = _sparse_product_reference(entries_a, entries_b, p, q, r)

        expected = torch.zeros_like(actual)
        for index, value in expected_sparse.items():
            expected[0, index] = value

        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize(
        ("p", "q", "r", "dtype", "atol"),
        [
            (12, 0, 0, torch.float64, 1e-12),
            (0, 12, 0, torch.float64, 1e-12),
            (6, 6, 0, torch.float64, 1e-12),
            (8, 3, 1, torch.float64, 1e-12),
            (4, 4, 4, torch.float64, 1e-12),
            (10, 4, 2, torch.float32, 1e-6),
        ],
    )
    def test_sparse_multivector_products_match_bitmask_rules_across_highdim_signatures(
        self,
        p,
        q,
        r,
        dtype,
        atol,
    ):
        # This sweep is the broad regression surface for metric signatures:
        # pure positive, pure negative, mixed nondegenerate, mixed degenerate,
        # and the current 16D ceiling. Each case uses the same sparse pattern so
        # differences come from signature signs and null annihilation only.
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=dtype,
            leaf_n=6,
            product_chunk_size=8 if p + q + r >= 16 else 16,
        )
        entries_a, entries_b = _signature_sweep_entries(p, q, r)
        A = _make_sparse_multivector(algebra, entries_a, dtype)
        B = _make_sparse_multivector(algebra, entries_b, dtype)

        actual = algebra.geometric_product(A, B)
        expected_sparse = _sparse_product_reference(entries_a, entries_b, p, q, r)
        expected = _make_expected_multivector(algebra, expected_sparse, dtype)

        assert torch.allclose(actual, expected, atol=atol, rtol=atol)

    @pytest.mark.parametrize(
        ("p", "q", "r", "dtype", "atol"),
        [
            pytest.param(8, 0, 0, torch.float64, 1e-12, id="n8-positive"),
            pytest.param(0, 9, 0, torch.float64, 1e-12, id="n9-negative"),
            pytest.param(0, 0, 10, torch.float64, 1e-12, id="n10-null"),
            pytest.param(6, 3, 2, torch.float64, 1e-12, id="n11-mixed-null"),
            pytest.param(8, 3, 1, torch.float64, 1e-12, id="n12-mixed-null"),
            pytest.param(7, 4, 2, torch.float64, 1e-12, id="n13-mixed-null"),
            pytest.param(8, 4, 2, torch.float32, 1e-6, id="n14-mixed-null"),
            pytest.param(9, 4, 2, torch.float32, 1e-6, id="n15-mixed-null"),
            pytest.param(10, 4, 2, torch.float32, 1e-6, id="n16-mixed-null"),
        ],
    )
    def test_sparse_multivector_products_cover_every_dimension_from_8_to_16(self, p, q, r, dtype, atol):
        # This is the main 8D-16D product sweep. It deliberately uses one
        # signature per dimension so changes in split depth, odd/even splits,
        # null compaction, and the 16D cap are all exercised by a single test
        # family. The bitmask oracle makes this independent from dense support.
        n = p + q + r
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=dtype,
            leaf_n=6,
            product_chunk_size=8 if n >= 14 else 16,
        )
        entries_a, entries_b = _signature_sweep_entries(p, q, r)
        A = _make_sparse_multivector(algebra, entries_a, dtype)
        B = _make_sparse_multivector(algebra, entries_b, dtype)

        actual = algebra.geometric_product(A, B)
        expected_sparse = _sparse_product_reference(entries_a, entries_b, p, q, r)
        expected = _make_expected_multivector(algebra, expected_sparse, dtype)

        assert torch.allclose(actual, expected, atol=atol, rtol=atol)

    def test_automatic_tiled_cl12_product_matches_bitmask_reference(self):
        # Cl(6,3,3) has signature gcd 3, so the automatic planner can reorder
        # dimensions into repeated Cl(2,1,1) tiles and share child modules. The
        # bitmask reference stays in public canonical order, making this a direct
        # check that permutation signs are restored correctly at the boundary.
        p, q, r = 6, 3, 3
        dtype = torch.float64
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=dtype,
            leaf_n=4,
            product_chunk_size=16,
        )
        entries_a, entries_b = _signature_sweep_entries(p, q, r)
        A = _make_sparse_multivector(algebra, entries_a, dtype)
        B = _make_sparse_multivector(algebra, entries_b, dtype)

        actual = algebra.geometric_product(A, B)
        expected_sparse = _sparse_product_reference(entries_a, entries_b, p, q, r)
        expected = _make_expected_multivector(algebra, expected_sparse, dtype)

        assert algebra.basis_permutation.uses_permutation
        assert algebra.left_sub.left_sub is algebra.right_sub
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_cl12_sparse_multivectors_satisfy_numerical_identities(self):
        # This test avoids any dense reference. Instead, it checks identities
        # that must hold for the product regardless of partition shape:
        # distributivity, associativity, reverse anti-automorphism, grade
        # involution automorphism, scalar multiplication, and vector
        # anticommutation.
        p, q, r = 7, 3, 2
        dtype = torch.float64
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=dtype,
            leaf_n=6,
            product_chunk_size=64,
        )
        deep_algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=dtype,
            leaf_n=3,
            product_chunk_size=17,
        )
        entries_a, entries_b = _signature_sweep_entries(p, q, r)
        entries_c = [(0, -0.25), (5, 0.5), ((1 << 4) | (1 << 9), -0.75), ((1 << 10) | 3, 1.125)]

        A = _make_sparse_multivector(algebra, entries_a, dtype)
        B = _make_sparse_multivector(algebra, entries_b, dtype)
        C = _make_sparse_multivector(algebra, entries_c, dtype)

        AB = algebra.geometric_product(A, B)
        BC = algebra.geometric_product(B, C)
        AC = algebra.geometric_product(A, C)

        assert torch.allclose(AB, deep_algebra.geometric_product(A, B), atol=1e-10, rtol=1e-10)
        assert torch.allclose(A + B, B + A, atol=0.0, rtol=0.0)
        assert torch.allclose(algebra.geometric_product(A, B + C), AB + AC, atol=1e-10, rtol=1e-10)
        assert torch.allclose(algebra.geometric_product(A + B, C), AC + BC, atol=1e-10, rtol=1e-10)

        assert torch.allclose(
            algebra.geometric_product(AB, C),
            algebra.geometric_product(A, BC),
            atol=1e-10,
            rtol=1e-10,
        )
        assert torch.allclose(
            algebra.reverse(AB),
            algebra.geometric_product(algebra.reverse(B), algebra.reverse(A)),
            atol=1e-10,
            rtol=1e-10,
        )
        assert torch.allclose(
            algebra.grade_involution(AB),
            algebra.geometric_product(algebra.grade_involution(A), algebra.grade_involution(B)),
            atol=1e-10,
            rtol=1e-10,
        )

        scalar = torch.zeros_like(A)
        scalar[..., 0] = -1.75
        assert torch.allclose(algebra.geometric_product(scalar, A), -1.75 * A, atol=1e-12, rtol=1e-12)
        assert torch.allclose(algebra.geometric_product(A, scalar), -1.75 * A, atol=1e-12, rtol=1e-12)

        e0 = _make_sparse_multivector(algebra, [(1 << 0, 1.0)], dtype)
        e2 = _make_sparse_multivector(algebra, [(1 << 2, 1.0)], dtype)
        e0e2 = algebra.geometric_product(e0, e2)
        e2e0 = algebra.geometric_product(e2, e0)
        assert torch.allclose(e0e2, -e2e0, atol=1e-12, rtol=1e-12)
        assert not torch.allclose(e0e2, e2e0, atol=1e-12, rtol=1e-12)

    def test_cl12_backward_matches_finite_difference_directional_derivative(self):
        # Autograd is checked against a directional finite difference using
        # sparse perturbations. Keeping A, B, dA, dB, and the loss weight sparse
        # makes the reference direction easy to inspect while still exercising
        # the recursive product backward path.
        p, q, r = 7, 3, 2
        dtype = torch.float64
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=dtype,
            leaf_n=6,
            product_chunk_size=16,
            accumulation_dtype=torch.float64,
        )

        entries_a, entries_b = _signature_sweep_entries(p, q, r)
        entries_da = [(1, 0.125), ((1 << 5) | (1 << 8), -0.25), ((1 << 10) | 3, 0.375)]
        entries_db = [(2, -0.2), ((1 << 6) | (1 << 11), 0.15), ((1 << 7) | (1 << 9), -0.3)]
        entries_w = [(0, -0.5), ((1 << 1) | (1 << 7), 0.75), ((1 << 8) | (1 << 11), -1.25)]

        A = _make_sparse_multivector(algebra, entries_a, dtype).requires_grad_(True)
        B = _make_sparse_multivector(algebra, entries_b, dtype).requires_grad_(True)
        dA = _make_sparse_multivector(algebra, entries_da, dtype)
        dB = _make_sparse_multivector(algebra, entries_db, dtype)
        weight = _make_sparse_multivector(algebra, entries_w, dtype)

        loss = (algebra.geometric_product(A, B) * weight).sum()
        loss.backward()
        directional_grad = (A.grad * dA).sum() + (B.grad * dB).sum()

        eps = 1e-6
        loss_plus = (algebra.geometric_product(A.detach() + eps * dA, B.detach() + eps * dB) * weight).sum()
        loss_minus = (algebra.geometric_product(A.detach() - eps * dA, B.detach() - eps * dB) * weight).sum()
        finite_difference = (loss_plus - loss_minus) / (2.0 * eps)

        assert torch.allclose(directional_grad, finite_difference, atol=1e-9, rtol=1e-9)

    def test_cl16_dense_basis_product_matches_direct_bitmask_reference(self):
        # A single basis-product case at the 16D cap catches indexing mistakes
        # that are invisible in smaller dimensions. The binary literals make it
        # easy to see which public bits are active in each operand.
        p, q, r = 10, 4, 2
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=6,
            product_chunk_size=8,
        )
        index_a = 0b1001_0010_0110_1011
        index_b = 0b0110_1101_1000_1110
        result_index, sign = _basis_product_reference(index_a, index_b, p, q, r)

        A = torch.zeros(1, algebra.dim)
        B = torch.zeros(1, algebra.dim)
        A[0, index_a] = 1.25
        B[0, index_b] = -0.5

        actual = algebra.geometric_product(A, B)
        expected = torch.zeros_like(actual)
        expected[0, result_index] = 1.25 * -0.5 * sign

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_cl12_embedded_subalgebra_product_matches_local_isomorphism(self):
        # A contiguous 6D public slice should behave exactly like a standalone
        # local algebra with the induced signature. ``_shift_index`` embeds each
        # local basis blade into the global bit positions, then the global output
        # is compared against the shifted local product.
        p, q, r = 6, 4, 2
        offset, width = 3, 6
        local_p, local_q, local_r = _signature_for_range_reference(p, q, r, offset, width)

        global_algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=6,
            product_chunk_size=16,
        )
        local_algebra = PartitionedCliffordAlgebra(
            local_p,
            local_q,
            local_r,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=4,
            product_chunk_size=16,
        )

        entries_a = [(0, 0.5), (3, -1.25), (17, 0.75), (63, -0.2)]
        entries_b = [(1, 2.0), (10, -0.5), (48, 1.5)]
        local_a = torch.zeros(1, local_algebra.dim, dtype=torch.float64)
        local_b = torch.zeros(1, local_algebra.dim, dtype=torch.float64)
        global_a = torch.zeros(1, global_algebra.dim, dtype=torch.float64)
        global_b = torch.zeros(1, global_algebra.dim, dtype=torch.float64)

        for index, value in entries_a:
            local_a[0, index] = value
            global_a[0, _shift_index(index, offset)] = value
        for index, value in entries_b:
            local_b[0, index] = value
            global_b[0, _shift_index(index, offset)] = value

        local_product = local_algebra.geometric_product(local_a, local_b)
        actual = global_algebra.geometric_product(global_a, global_b)

        expected = torch.zeros_like(actual)
        shifted_indices = torch.tensor(
            [_shift_index(index, offset) for index in range(local_algebra.dim)],
            dtype=torch.long,
        )
        expected[..., shifted_indices] = local_product

        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_cl12_random_three_dimensional_subspace_projects_to_n3_engine(self):
        # This is a basis-independent isomorphism check. A random orthonormal
        # 3-frame defines an embedded Cl(3,0) subalgebra inside Cl(12,0). After
        # multiplying embedded multivectors globally, projecting back onto the
        # constructed basis must recover the local Cl(3,0) product.
        dtype = torch.float64
        global_algebra = PartitionedCliffordAlgebra(
            12,
            0,
            0,
            device=DEVICE,
            dtype=dtype,
            leaf_n=6,
            product_chunk_size=64,
        )
        local_algebra = CliffordAlgebra(3, 0, 0, device=DEVICE, dtype=dtype)

        generator = torch.Generator(device=DEVICE).manual_seed(211)
        frame, _ = torch.linalg.qr(torch.randn(global_algebra.n, 3, dtype=dtype, generator=generator))
        basis = _make_orthonormal_subspace_basis(global_algebra, frame.contiguous())

        gram = basis @ basis.transpose(0, 1)
        assert torch.allclose(gram, torch.eye(local_algebra.dim, dtype=dtype), atol=1e-12, rtol=1e-12)

        local_a = torch.randn(2, local_algebra.dim, dtype=dtype, generator=generator)
        local_b = torch.randn(2, local_algebra.dim, dtype=dtype, generator=generator)
        global_a = local_a @ basis
        global_b = local_b @ basis

        local_expected = local_algebra.geometric_product(local_a, local_b)
        global_product = global_algebra.geometric_product(global_a, global_b)
        projected = global_product @ basis.transpose(0, 1)
        embedded_expected = local_expected @ basis

        assert torch.allclose(projected, local_expected, atol=1e-10, rtol=1e-10)
        assert torch.allclose(global_product, embedded_expected, atol=1e-10, rtol=1e-10)

    def test_cl16_recursive_sign_merge_matches_direct_bitmask_reference(self):
        # This helper-level test walks the partition tree one basis product at a
        # time. It is faster than materializing full multivectors and directly
        # targets the recursive split/bridge/permutation sign merge.
        p, q, r = 10, 4, 2
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=6,
            product_chunk_size=4,
        )
        pairs = [
            (0, 0),
            (1, 1 << 15),
            (0xABCD, 0x1357),
            (0xFFFF, 0x0011),
            (0x2222, 0xDDDD),
            (0x7A5C, 0xC3A5),
            ((1 << 14) | 7, (1 << 14) | 3),
        ]

        for index_a, index_b in pairs:
            expected = _basis_product_reference(index_a, index_b, p, q, r)
            actual = _partitioned_basis_product(algebra, index_a, index_b)
            assert actual == expected

    def test_cl16_forced_unbalanced_tree_matches_bitmask_reference(self):
        # Explicit tree expressions are how callers force a non-default split.
        # This tree peels off four low dimensions, then recursively splits the
        # remaining high dimensions. It crosses positive, negative, and null
        # signature blocks, so both public-basis permutation and bridge signs
        # must be correct.
        p, q, r = 10, 4, 2
        dtype = torch.float32
        partition_tree = "R=0-3; L.R=4-7; L.L.R=8-11; L.L.L=12-15"
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=dtype,
            leaf_n=6,
            product_chunk_size=4,
            partition_tree=partition_tree,
        )
        entries_a, entries_b = _signature_sweep_entries(p, q, r)
        A = _make_sparse_multivector(algebra, entries_a, dtype)
        B = _make_sparse_multivector(algebra, entries_b, dtype)

        actual = algebra.geometric_product(A, B)
        expected_sparse = _sparse_product_reference(entries_a, entries_b, p, q, r)
        expected = _make_expected_multivector(algebra, expected_sparse, dtype)

        tree = algebra.describe_tree()
        assert "root: Cl(10,4,2)" in tree
        assert "root.L.L" in tree
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_cl16_basis_products_satisfy_algebraic_identities(self):
        # Tuple-valued basis products make exact identity checks possible at the
        # 16D cap. Associativity validates recursive merge consistency; reverse
        # and grade-involution identities validate the sign formulas used by the
        # structural buffers.
        p, q, r = 10, 4, 2
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=6,
            product_chunk_size=4,
        )
        triples = [
            (0x1234, 0x00F0, 0xABCD),
            (0x7001, 0x02A8, 0x1111),
            ((1 << 14) | 0x35, 0x4440, 0x2101),
            (0x7A5C, (1 << 15) | 0x81, 0x0013),
        ]

        for index_a, index_b, index_c in triples:
            left = _multiply_signed_basis(
                algebra,
                _multiply_signed_basis(algebra, (index_a, 1.0), (index_b, 1.0)),
                (index_c, 1.0),
            )
            right = _multiply_signed_basis(
                algebra,
                (index_a, 1.0),
                _multiply_signed_basis(algebra, (index_b, 1.0), (index_c, 1.0)),
            )
            assert left == right

        pairs = [
            (0x1234, 0x00F0),
            (0x7A5C, 0xC3A5),
            ((1 << 14) | 0x101, (1 << 14) | 0x077),
            ((1 << 15) | 0x222, 0x1357),
        ]
        for index_a, index_b in pairs:
            ab = _multiply_signed_basis(algebra, (index_a, 1.0), (index_b, 1.0))
            reverse_ab = _canonical_basis_term(ab[0], ab[1] * _reverse_sign(ab[0]))
            reverse_ba = _multiply_signed_basis(
                algebra,
                (index_b, _reverse_sign(index_b)),
                (index_a, _reverse_sign(index_a)),
            )
            assert reverse_ab == reverse_ba

            involution_ab = _canonical_basis_term(ab[0], ab[1] * _grade_involution_sign(ab[0]))
            involution_product = _multiply_signed_basis(
                algebra,
                (index_a, _grade_involution_sign(index_a)),
                (index_b, _grade_involution_sign(index_b)),
            )
            assert involution_ab == involution_product

    def test_cl16_simple_bivector_exp_matches_long_taylor_reference(self):
        # Simple bivector exponentials have a closed two-term form. The long
        # Taylor reference is intentionally scalar-only and independent of the
        # algebra implementation, so it catches coefficient/sign regressions in
        # the high-dimensional closed-form path.
        p, q, r = 16, 0, 0
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=6,
            product_chunk_size=4,
        )
        bivector_index = (1 << 0) | (1 << 13)
        theta = 0.375
        square = -1.0

        B = torch.zeros(1, algebra.dim, dtype=torch.float64)
        B[0, bivector_index] = theta
        actual = algebra._exp_bivector_closed(B)

        scalar_ref, bivector_ref = _long_taylor_simple_bivector(theta, square, order=80)

        assert torch.allclose(actual[0, 0], torch.tensor(scalar_ref, dtype=torch.float64), atol=1e-14, rtol=1e-14)
        assert torch.allclose(
            actual[0, bivector_index],
            torch.tensor(bivector_ref, dtype=torch.float64),
            atol=1e-14,
            rtol=1e-14,
        )
        assert torch.count_nonzero(actual).item() == 2

    @pytest.mark.parametrize(
        ("p", "q", "r", "bivector_index", "theta", "scalar_ref", "bivector_ref"),
        [
            (16, 0, 0, (1 << 0) | (1 << 13), 0.375, math.cos(0.375), math.sin(0.375)),
            (1, 15, 0, (1 << 0) | (1 << 1), 0.25, math.cosh(0.25), math.sinh(0.25)),
            (14, 0, 2, (1 << 0) | (1 << 14), 0.5, 1.0, 0.5),
        ],
    )
    def test_cl16_simple_bivector_exp_matches_closed_form(
        self,
        p,
        q,
        r,
        bivector_index,
        theta,
        scalar_ref,
        bivector_ref,
    ):
        # Parameterized closed-form cases cover the three regimes:
        # Euclidean-like elliptic, Lorentzian hyperbolic, and degenerate
        # parabolic. The selected bivector index always contains exactly two
        # bits so the expected output has only scalar and bivector components.
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=6,
            product_chunk_size=4,
        )

        B = torch.zeros(1, algebra.dim, dtype=torch.float64)
        B[0, bivector_index] = theta
        actual = algebra._exp_bivector_closed(B)

        assert torch.allclose(actual[0, 0], torch.tensor(scalar_ref, dtype=torch.float64), atol=1e-14, rtol=1e-14)
        assert torch.allclose(
            actual[0, bivector_index],
            torch.tensor(bivector_ref, dtype=torch.float64),
            atol=1e-14,
            rtol=1e-14,
        )
        assert torch.count_nonzero(actual).item() == 2

    def test_cl16_lorentzian_bivector_exp_matches_long_taylor_reference(self):
        # Separate long-Taylor check for the hyperbolic sign regime. In
        # Cl(1,15), e0 has positive square and e1 has negative square, so
        # (e0e1)^2 = +1 and exp(theta e0e1) uses cosh/sinh.
        p, q, r = 1, 15, 0
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=6,
            product_chunk_size=4,
        )
        bivector_index = (1 << 0) | (1 << 1)
        theta = 0.25
        square = 1.0

        B = torch.zeros(1, algebra.dim, dtype=torch.float64)
        B[0, bivector_index] = theta
        actual = algebra._exp_bivector_closed(B)

        scalar_ref, bivector_ref = _long_taylor_simple_bivector(theta, square, order=80)

        assert torch.allclose(actual[0, 0], torch.tensor(scalar_ref, dtype=torch.float64), atol=1e-14, rtol=1e-14)
        assert torch.allclose(
            actual[0, bivector_index],
            torch.tensor(bivector_ref, dtype=torch.float64),
            atol=1e-14,
            rtol=1e-14,
        )
        assert torch.count_nonzero(actual).item() == 2

    def test_cl16_degenerate_repeated_null_factor_annihilates_product(self):
        # In a degenerate signature, any product that repeats the same null
        # basis vector must vanish. These cases keep additional non-null bits in
        # each operand to ensure the annihilation is detected before other sign
        # details can mask the bug.
        p, q, r = 10, 4, 2
        algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=6,
            product_chunk_size=4,
        )
        null_bit = p + q
        pairs = [
            ((1 << null_bit) | 0x35, (1 << null_bit) | 0xC0),
            ((1 << (null_bit + 1)) | 0x1A5, (1 << (null_bit + 1)) | 0x21),
        ]

        for index_a, index_b in pairs:
            expected = _basis_product_reference(index_a, index_b, p, q, r)
            actual = _partitioned_basis_product(algebra, index_a, index_b)
            assert expected[1] == 0.0
            assert actual == expected
