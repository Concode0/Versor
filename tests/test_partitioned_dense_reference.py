# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Dense-reference checks for partitioned Clifford algebra.

This file covers the region where both kernels are valid, so regressions show
up as direct numerical error instead of only algebraic identity failures. The
slow Cl12 case is intentionally separated from the regular unit sweep because
the dense reference allocates a monolithic Cayley table.
"""

import pytest
import torch

from core.config import PartitionConfig, make_algebra

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def _make_dense_reference_pair(p: int, q: int, r: int, leaf_n: int, product_chunk_size: int = 32):
    reference = make_algebra(p, q, r, kernel="dense", device=DEVICE, dtype=torch.float64)
    partitioned = make_algebra(
        p,
        q,
        r,
        kernel="partitioned",
        device=DEVICE,
        dtype=torch.float64,
        partition=PartitionConfig(leaf_n=leaf_n, product_chunk_size=product_chunk_size),
    )
    return reference, partitioned


def _dense_inputs(dim: int, seed: int, batch_shape=(1,)):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    scale = 0.125
    A = torch.randn(*batch_shape, dim, dtype=torch.float64, generator=generator) * scale
    B = torch.randn(*batch_shape, dim, dtype=torch.float64, generator=generator) * scale
    return A, B


def _assert_bounded_error(actual: torch.Tensor, expected: torch.Tensor, label: str, *, atol=2e-9, rtol=2e-9):
    diff = actual - expected
    max_abs = diff.abs().max().item()
    denominator = expected.norm().clamp_min(torch.finfo(expected.dtype).eps)
    relative = (diff.norm() / denominator).item()
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{label} exceeded dense-reference error bounds: max_abs={max_abs:.3e}, relative={relative:.3e}"
    )
    if rtol == 0.0:
        assert max_abs == 0.0, f"{label} expected exact agreement, got max_abs={max_abs:.3e}"
    else:
        assert relative < rtol * 10.0, f"{label} relative error is excessive: {relative:.3e}"


@pytest.mark.parametrize(
    ("p", "q", "r", "leaf_n"),
    [
        pytest.param(4, 0, 0, 2, id="cl4_forced_recursive"),
        pytest.param(5, 2, 1, 4, id="cl8_mixed"),
        pytest.param(8, 0, 0, 4, id="cl8_euclidean"),
        pytest.param(7, 2, 1, 5, id="cl10_mixed"),
    ],
)
def test_dense_comparable_binary_operations_have_bounded_error(p, q, r, leaf_n):
    reference, partitioned = _make_dense_reference_pair(p, q, r, leaf_n)
    A, B = _dense_inputs(partitioned.dim, seed=503 + p * 17 + q * 11 + r)

    for method_name in [
        "geometric_product",
        "wedge",
        "inner_product",
        "commutator",
        "anti_commutator",
        "left_contraction",
    ]:
        expected = getattr(reference, method_name)(A, B)
        actual = getattr(partitioned, method_name)(A, B)
        _assert_bounded_error(actual, expected, f"Cl({p},{q},{r}).{method_name}")


@pytest.mark.parametrize(
    ("p", "q", "r", "leaf_n"),
    [
        pytest.param(8, 0, 0, 4, id="cl8_euclidean"),
        pytest.param(7, 2, 1, 5, id="cl10_mixed"),
    ],
)
def test_dense_comparable_unary_operations_have_bounded_error(p, q, r, leaf_n):
    reference, partitioned = _make_dense_reference_pair(p, q, r, leaf_n)
    mv, _ = _dense_inputs(partitioned.dim, seed=719 + p * 17 + q * 11 + r, batch_shape=(2,))

    for grade in range(partitioned.num_grades):
        expected = reference.grade_projection(mv, grade)
        actual = partitioned.grade_projection(mv, grade)
        _assert_bounded_error(actual, expected, f"Cl({p},{q},{r}).grade_projection({grade})", atol=0.0, rtol=0.0)

    for method_name in [
        "reverse",
        "pseudoscalar_product",
        "dual",
        "grade_involution",
        "clifford_conjugation",
        "norm_sq",
    ]:
        expected = getattr(reference, method_name)(mv)
        actual = getattr(partitioned, method_name)(mv)
        _assert_bounded_error(actual, expected, f"Cl({p},{q},{r}).{method_name}")

    vectors = torch.randn(2, partitioned.n, dtype=torch.float64, generator=torch.Generator().manual_seed(907))
    _assert_bounded_error(partitioned.embed_vector(vectors), reference.embed_vector(vectors), "embed_vector")


def test_partitioned_8d_fixture_matches_dense_reference(partitioned_algebra_8d):
    reference = make_algebra(8, 0, 0, kernel="dense", device=DEVICE, dtype=torch.float64)
    A, B = _dense_inputs(partitioned_algebra_8d.dim, seed=1013, batch_shape=(2,))

    expected = reference.geometric_product(A, B)
    actual = partitioned_algebra_8d.geometric_product(A, B)

    _assert_bounded_error(actual, expected, "partitioned_algebra_8d.geometric_product")


@pytest.mark.slow
def test_partitioned_12d_fixture_dense_reference_error_is_bounded(partitioned_algebra_12d):
    reference = make_algebra(12, 0, 0, kernel="dense", device=DEVICE, dtype=torch.float64)
    A, B = _dense_inputs(partitioned_algebra_12d.dim, seed=1207)

    expected = reference.geometric_product(A, B)
    actual = partitioned_algebra_12d.geometric_product(A, B)

    _assert_bounded_error(actual, expected, "partitioned_algebra_12d.geometric_product", atol=5e-9, rtol=5e-9)


@pytest.mark.slow
def test_partitioned_12d_mixed_fixture_matches_bitmask_reference(partitioned_algebra_12d_mixed):
    entries_a = [(0, 0.25), (3, -1.5), (257, 0.75), (2049, 2.0)]
    entries_b = [(1, -0.5), (384, 1.25), (1025, -2.0), (4095, 0.5)]
    A = _make_sparse_multivector(partitioned_algebra_12d_mixed, entries_a, torch.float64)
    B = _make_sparse_multivector(partitioned_algebra_12d_mixed, entries_b, torch.float64)

    expected = torch.zeros_like(A)
    expected_sparse = _sparse_product_reference(entries_a, entries_b, 8, 3, 1)
    for index, value in expected_sparse.items():
        expected[0, index] = value

    actual = partitioned_algebra_12d_mixed.geometric_product(A, B)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.slow
def test_partitioned_16d_fixture_basis_product_matches_bitmask_reference(partitioned_algebra_16d):
    p, q, r = 10, 4, 2
    index_a = 0b1001_0010_0110_1011
    index_b = 0b0110_1101_1000_1110
    result_index, sign = _basis_product_reference(index_a, index_b, p, q, r)

    A = torch.zeros(1, partitioned_algebra_16d.dim, dtype=torch.float32)
    B = torch.zeros(1, partitioned_algebra_16d.dim, dtype=torch.float32)
    A[0, index_a] = 1.25
    B[0, index_b] = -0.5

    expected = torch.zeros_like(A)
    expected[0, result_index] = 1.25 * -0.5 * sign
    actual = partitioned_algebra_16d.geometric_product(A, B)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def _make_sparse_multivector(algebra, entries, dtype: torch.dtype) -> torch.Tensor:
    mv = torch.zeros(1, algebra.dim, dtype=dtype)
    for index, value in entries:
        mv[0, index] += value
    return mv


def _sparse_product_reference(
    entries_a: list[tuple[int, float]],
    entries_b: list[tuple[int, float]],
    p: int,
    q: int,
    r: int,
) -> dict[int, float]:
    result = {}
    for index_a, value_a in entries_a:
        for index_b, value_b in entries_b:
            result_index, sign = _basis_product_reference(index_a, index_b, p, q, r)
            if sign == 0.0:
                continue
            result[result_index] = result.get(result_index, 0.0) + value_a * value_b * sign
    return result


def _basis_product_reference(index_a: int, index_b: int, p: int, q: int, r: int) -> tuple[int, float]:
    n = p + q + r
    swap_count = 0
    for bit in range(n):
        if index_a & (1 << bit):
            swap_count += (index_b & ((1 << bit) - 1)).bit_count()

    sign = -1.0 if swap_count % 2 else 1.0

    negative_mask = sum(1 << bit for bit in range(p, p + q))
    if ((index_a & index_b & negative_mask).bit_count() % 2) == 1:
        sign = -sign

    null_mask = sum(1 << bit for bit in range(p + q, n))
    if (index_a & index_b & null_mask) != 0:
        sign = 0.0

    return index_a ^ index_b, sign
