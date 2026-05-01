# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

import pytest
import torch
import torch.nn as nn

from core.algebra import CliffordAlgebra
from core.partitioned_algebra import PartitionedCliffordAlgebra

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def _dtype_tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return 5e-3
    if dtype == torch.bfloat16:
        return 5e-2
    if dtype == torch.float32:
        return 2e-5
    return 1e-10


def _make_pair(p=3, q=1, r=0, *, leaf_n=2, product_chunk_size=None, dtype=torch.float64):
    reference = CliffordAlgebra(p, q, r, device=DEVICE, dtype=dtype)
    algebra = PartitionedCliffordAlgebra(
        p,
        q,
        r,
        device=DEVICE,
        dtype=dtype,
        leaf_n=leaf_n,
        product_chunk_size=product_chunk_size,
    )
    return reference, algebra


def _assert_matches_monolithic(p, q=0, r=0, *, leaf_n=6, shape=(3,), dtype=torch.float64):
    torch.manual_seed(17)
    reference = CliffordAlgebra(p, q, r, device=DEVICE, dtype=dtype)
    algebra = PartitionedCliffordAlgebra(p, q, r, device=DEVICE, dtype=dtype, leaf_n=leaf_n)

    dim = 2 ** (p + q + r)
    A = torch.randn(*shape, dim, dtype=dtype)
    B = torch.randn(*shape, dim, dtype=dtype)

    expected = reference.geometric_product(A, B)
    actual = algebra.geometric_product(A, B)

    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)


class _PartitionedProductLayer(nn.Module):
    def __init__(self, p, q=0, r=0):
        super().__init__()
        self.algebra = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=2,
            product_chunk_size=4,
        )
        self.weight = nn.Parameter(torch.randn(self.algebra.dim))

    def forward(self, x):
        weight = self.weight.expand_as(x)
        return self.algebra.geometric_product(x, weight)


class TestPartitionedCliffordAlgebra:
    def test_leaf_matches_core_kernel(self):
        _assert_matches_monolithic(3, 1, 0, leaf_n=6, shape=(2,))

    def test_forced_recursive_euclidean_matches_core_kernel(self):
        _assert_matches_monolithic(4, 0, 0, leaf_n=2, shape=(4,))

    def test_default_recursive_cl8_matches_core_kernel(self):
        _assert_matches_monolithic(8, 0, 0, leaf_n=6, shape=(2,))

    def test_recursive_tree_uses_balanced_binary_splits(self):
        algebra = PartitionedCliffordAlgebra(20, 0, 0, device=DEVICE, leaf_n=6)

        assert algebra.left_n == 10
        assert algebra.right_n == 10
        assert algebra.left_sub.left_n == 5
        assert algebra.left_sub.right_n == 5
        assert algebra.right_sub.left_n == 5
        assert algebra.right_sub.right_n == 5

        assert not hasattr(algebra.left_sub, "cayley_indices")
        assert not hasattr(algebra.right_sub, "cayley_indices")

    def test_describe_tree_reports_split_layout_and_shared_nodes(self, capsys):
        algebra = PartitionedCliffordAlgebra(8, 0, 0, device=DEVICE, leaf_n=2)

        tree = algebra.describe_tree()
        lines = tree.splitlines()

        assert lines[0] == (
            "root: Cl(8,0,0), n=8, dim=256, bits=[0, 8), "
            "split left=4 bits=[4, 8), right=4 bits=[0, 4), pairs=256, chunk=64"
        )
        assert "root.L: Cl(4,0,0), n=4, dim=16, bits=[4, 8)" in lines[1]
        assert "root.R: Cl(4,0,0), n=4, dim=16, bits=[0, 4)" in tree
        assert "shared_with=root.L" in tree

        algebra.print_tree()
        assert capsys.readouterr().out.strip() == tree

    def test_repeated_signature_tiles_share_subalgebras_automatically(self):
        algebra = PartitionedCliffordAlgebra(
            8,
            4,
            4,
            device=DEVICE,
            leaf_n=4,
        )

        assert (algebra.left_sub.p, algebra.left_sub.q, algebra.left_sub.r) == (4, 2, 2)
        assert (algebra.right_sub.p, algebra.right_sub.q, algebra.right_sub.r) == (4, 2, 2)
        assert algebra.left_sub is algebra.right_sub
        assert (algebra.left_sub.left_sub.p, algebra.left_sub.left_sub.q, algebra.left_sub.left_sub.r) == (2, 1, 1)
        assert algebra.left_sub.left_sub is algebra.left_sub.right_sub
        assert algebra.basis_permutation.uses_permutation

        tree = algebra.describe_tree()
        assert "root: Cl(8,4,4)" in tree
        assert "root.L: Cl(4,2,2)" in tree
        assert "root.L.L: Cl(2,1,1)" in tree
        assert "shared_with=root.L" in tree
        assert "shared_with=root.L.L" in tree

    def test_repeated_signature_tile_product_matches_core_kernel_with_basis_permutation(self):
        torch.manual_seed(107)
        reference = CliffordAlgebra(4, 2, 2, device=DEVICE, dtype=torch.float64)
        algebra = PartitionedCliffordAlgebra(
            4,
            2,
            2,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=4,
        )
        A = torch.randn(2, algebra.dim, dtype=torch.float64)
        B = torch.randn(2, algebra.dim, dtype=torch.float64)

        assert algebra.basis_permutation.uses_permutation
        assert torch.allclose(
            algebra.geometric_product(A, B), reference.geometric_product(A, B), atol=1e-10, rtol=1e-10
        )

    def test_repeated_signature_tile_product_gradients_match_core_kernel(self):
        torch.manual_seed(109)
        reference = CliffordAlgebra(4, 2, 2, device=DEVICE, dtype=torch.float64)
        algebra = PartitionedCliffordAlgebra(
            4,
            2,
            2,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=4,
        )

        A_ref = torch.randn(2, algebra.dim, dtype=torch.float64, requires_grad=True)
        B_ref = torch.randn(2, algebra.dim, dtype=torch.float64, requires_grad=True)
        A_partitioned = A_ref.detach().clone().requires_grad_(True)
        B_partitioned = B_ref.detach().clone().requires_grad_(True)

        reference.geometric_product(A_ref, B_ref).square().sum().backward()
        algebra.geometric_product(A_partitioned, B_partitioned).square().sum().backward()

        assert torch.allclose(A_partitioned.grad, A_ref.grad, atol=1e-9, rtol=1e-9)
        assert torch.allclose(B_partitioned.grad, B_ref.grad, atol=1e-9, rtol=1e-9)

    def test_identical_recursive_subalgebras_are_shared(self):
        algebra = PartitionedCliffordAlgebra(8, 0, 0, device=DEVICE, leaf_n=2)

        assert algebra.left_sub is algebra.right_sub
        assert algebra.left_sub.left_sub is algebra.left_sub.right_sub

    def test_recursive_node_uses_compact_memory_layout(self):
        algebra = PartitionedCliffordAlgebra(4, 0, 0, device=DEVICE, leaf_n=2)

        assert "_grade_masks_float" not in algebra._buffers
        assert "_grade_masks_float_T" not in algebra._buffers
        assert torch.equal(algebra.grade_masks[2], algebra.grade_index == 2)
        assert algebra.grade_masks_float.dtype == algebra.dtype

        assert algebra._right_pair_count == algebra.right_dim * algebra.right_dim
        assert not hasattr(algebra, "_right_pair_full")
        assert not hasattr(algebra, "_right_pair_a")
        assert not hasattr(algebra, "_right_pair_b")
        assert not hasattr(algebra, "_right_pair_result")
        assert not hasattr(algebra, "_right_pair_signs")
        assert not hasattr(algebra, "_right_interaction")
        assert not hasattr(algebra, "bridge_signs")
        assert not hasattr(algebra, "_uses_basis_permutation")
        assert not hasattr(algebra, "_to_split_basis")
        assert not hasattr(algebra, "_to_public_basis")
        assert not hasattr(algebra, "_split_basis_signs")
        _, _, pair_result, pair_signs = algebra._right_product_slice(0, algebra._right_pair_count)
        assert pair_result.shape == (algebra._right_pair_count,)
        assert pair_signs.dtype == torch.int8
        assert not algebra.basis_permutation.uses_permutation
        assert algebra.basis_permutation.split_to_public.numel() == 0
        assert algebra.basis_permutation.public_to_split.numel() == 0
        assert algebra.basis_permutation.split_signs.numel() == 0

    def test_default_recursive_mixed_signature_matches_core_kernel(self):
        _assert_matches_monolithic(5, 2, 1, leaf_n=6, shape=(2,))

    def test_recursive_product_supports_extra_batch_axes(self):
        _assert_matches_monolithic(4, 1, 0, leaf_n=3, shape=(2, 3))

    def test_recursive_product_gradients_match_core_kernel(self):
        torch.manual_seed(23)
        reference = CliffordAlgebra(4, 0, 0, device=DEVICE, dtype=torch.float64)
        algebra = PartitionedCliffordAlgebra(4, 0, 0, device=DEVICE, dtype=torch.float64, leaf_n=2)

        A_ref = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
        B_ref = torch.randn(2, 16, dtype=torch.float64, requires_grad=True)
        A_partitioned = A_ref.detach().clone().requires_grad_(True)
        B_partitioned = B_ref.detach().clone().requires_grad_(True)

        reference.geometric_product(A_ref, B_ref).square().sum().backward()
        algebra.geometric_product(A_partitioned, B_partitioned).square().sum().backward()

        assert torch.allclose(A_partitioned.grad, A_ref.grad, atol=1e-9, rtol=1e-9)
        assert torch.allclose(B_partitioned.grad, B_ref.grad, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_recursive_operations_support_generic_floating_dtypes(self, dtype):
        torch.manual_seed(67)
        reference = CliffordAlgebra(4, 0, 0, device=DEVICE, dtype=dtype)
        algebra = PartitionedCliffordAlgebra(4, 0, 0, device=DEVICE, dtype=dtype, leaf_n=2)
        atol = _dtype_tolerance(dtype)

        A = torch.randn(2, algebra.dim, dtype=dtype) * 0.25
        B = torch.randn(2, algebra.dim, dtype=dtype) * 0.25

        for method_name in [
            "geometric_product",
            "wedge",
            "inner_product",
            "commutator",
            "anti_commutator",
            "left_contraction",
        ]:
            expected = getattr(reference, method_name)(A, B)
            actual = getattr(algebra, method_name)(A, B)
            assert actual.dtype == dtype
            assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=atol), method_name

        mv = torch.randn(2, algebra.dim, dtype=dtype) * 0.25
        assert algebra.norm_sq(mv).dtype == dtype
        assert torch.allclose(algebra.norm_sq(mv).float(), reference.norm_sq(mv).float(), atol=atol, rtol=atol)

        bivector = torch.zeros(1, algebra.dim, dtype=dtype)
        bivector[0, 3] = 0.125
        actual_exp = algebra.exp(bivector)
        expected_exp = reference.exp(bivector)
        assert actual_exp.dtype == dtype
        assert torch.allclose(actual_exp.float(), expected_exp.float(), atol=atol, rtol=atol)

    @pytest.mark.parametrize(
        ("algebra_dtype", "input_dtype", "expected_dtype"),
        [
            (torch.float64, torch.float32, torch.float64),
            (torch.float32, torch.float16, torch.float32),
            (torch.float16, torch.bfloat16, torch.float32),
            (torch.bfloat16, torch.float16, torch.float32),
        ],
    )
    @pytest.mark.parametrize("leaf_n", [2, 6])
    def test_geometric_product_promotes_inputs_with_algebra_dtype(
        self,
        algebra_dtype,
        input_dtype,
        expected_dtype,
        leaf_n,
    ):
        torch.manual_seed(71)
        algebra = PartitionedCliffordAlgebra(4, 0, 0, device=DEVICE, dtype=algebra_dtype, leaf_n=leaf_n)
        reference = CliffordAlgebra(4, 0, 0, device=DEVICE, dtype=expected_dtype)
        atol = _dtype_tolerance(expected_dtype)

        A = torch.randn(2, algebra.dim, dtype=input_dtype) * 0.25
        B = torch.randn(2, algebra.dim, dtype=input_dtype) * 0.25

        actual = algebra.geometric_product(A, B)
        expected = reference.geometric_product(A.to(dtype=expected_dtype), B.to(dtype=expected_dtype))

        assert actual.dtype == expected_dtype
        assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=atol)

        bivector = torch.zeros(1, algebra.dim, dtype=input_dtype)
        bivector[0, 3] = 0.125
        actual_exp = algebra.exp(bivector)
        expected_exp = reference.exp(bivector.to(dtype=expected_dtype))

        assert actual_exp.dtype == expected_dtype
        assert torch.allclose(actual_exp.float(), expected_exp.float(), atol=atol, rtol=atol)

    def test_stable_accumulation_reduces_cumulative_forward_error(self):
        torch.manual_seed(73)
        reference = CliffordAlgebra(8, 0, 0, device=DEVICE, dtype=torch.float64)
        standard = PartitionedCliffordAlgebra(
            8, 0, 0, device=DEVICE, dtype=torch.float32, leaf_n=4, product_chunk_size=1
        )
        stable = PartitionedCliffordAlgebra(
            8,
            0,
            0,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=4,
            product_chunk_size=1,
            accumulation_dtype=torch.float64,
        )
        assert stable.left_sub.accumulation_dtype == torch.float64

        factors = [torch.randn(1, standard.dim, dtype=torch.float32) * 0.02 for _ in range(5)]
        expected = factors[0].double()
        actual_standard = factors[0]
        actual_stable = factors[0]
        for factor in factors[1:]:
            expected = reference.geometric_product(expected, factor.double())
            actual_standard = standard.geometric_product(actual_standard, factor)
            actual_stable = stable.geometric_product(actual_stable, factor)

        standard_error = (actual_standard.double() - expected).norm()
        stable_error = (actual_stable.double() - expected).norm()

        assert actual_stable.dtype == torch.float32
        assert stable_error < standard_error * 0.5

    def test_stable_accumulation_reduces_cumulative_backward_error(self):
        torch.manual_seed(79)
        reference = CliffordAlgebra(8, 0, 0, device=DEVICE, dtype=torch.float64)
        standard = PartitionedCliffordAlgebra(
            8, 0, 0, device=DEVICE, dtype=torch.float32, leaf_n=4, product_chunk_size=1
        )
        stable = PartitionedCliffordAlgebra(
            8,
            0,
            0,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=4,
            product_chunk_size=1,
            accumulation_dtype=torch.float64,
        )

        A = torch.randn(2, standard.dim, dtype=torch.float32) * 0.02
        B = torch.randn(2, standard.dim, dtype=torch.float32) * 0.02
        A_ref = A.double().requires_grad_(True)
        B_ref = B.double().requires_grad_(True)
        A_standard = A.detach().clone().requires_grad_(True)
        B_standard = B.detach().clone().requires_grad_(True)
        A_stable = A.detach().clone().requires_grad_(True)
        B_stable = B.detach().clone().requires_grad_(True)

        reference.geometric_product(reference.geometric_product(A_ref, B_ref), B_ref).square().sum().backward()
        standard.geometric_product(
            standard.geometric_product(A_standard, B_standard), B_standard
        ).square().sum().backward()
        stable.geometric_product(stable.geometric_product(A_stable, B_stable), B_stable).square().sum().backward()

        standard_error = (A_standard.grad.double() - A_ref.grad).norm() + (B_standard.grad.double() - B_ref.grad).norm()
        stable_error = (A_stable.grad.double() - A_ref.grad).norm() + (B_stable.grad.double() - B_ref.grad).norm()

        assert A_stable.grad.dtype == torch.float32
        assert B_stable.grad.dtype == torch.float32
        assert stable_error < standard_error * 0.7

    def test_recursive_product_chunked_pair_merge_matches_core_kernel(self):
        torch.manual_seed(29)
        reference, algebra = _make_pair(5, 1, 0, leaf_n=2, product_chunk_size=3)
        A = torch.randn(2, 1, algebra.dim, dtype=torch.float64)
        B = torch.randn(1, 3, algebra.dim, dtype=torch.float64)

        expected = reference.geometric_product(A, B)
        actual = algebra.geometric_product(A, B)

        assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)

    @pytest.mark.parametrize("pair_range", ["full", "chunk"])
    def test_indexed_right_interaction_merge_matches_reference(self, pair_range):
        torch.manual_seed(31)
        product_chunk_size = None if pair_range == "full" else 3
        algebra = PartitionedCliffordAlgebra(
            5,
            1,
            0,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=2,
            product_chunk_size=product_chunk_size,
        )
        if pair_range == "full":
            start, end = 0, algebra._right_pair_count
        else:
            start, end = algebra._product_chunk_size, 2 * algebra._product_chunk_size

        _, _, pair_result, pair_signs = algebra._right_product_slice(start, end)
        pair_count = int(pair_signs.numel())
        left_products = torch.randn(2, pair_count, algebra.left_dim, dtype=torch.float64)
        merged_terms = left_products.clone().requires_grad_(True)
        index_terms = left_products.clone().requires_grad_(True)

        actual = algebra._merge_right_interactions(merged_terms, pair_result, pair_signs)
        expected = algebra._merge_right_interactions_index_add(index_terms, pair_result, pair_signs)

        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

        weight = torch.randn_like(actual)
        (actual * weight).sum().backward()
        (expected * weight).sum().backward()

        assert torch.allclose(merged_terms.grad, index_terms.grad, atol=1e-12, rtol=1e-12)

    @pytest.mark.parametrize(("p", "q", "r"), [(5, 1, 0), (1, 1, 4)])
    def test_vectorized_full_pair_product_matches_chunked_with_gradients(self, p, q, r):
        torch.manual_seed(83 + p * 13 + q * 7 + r)
        reference = CliffordAlgebra(p, q, r, device=DEVICE, dtype=torch.float64)
        vectorized = PartitionedCliffordAlgebra(p, q, r, device=DEVICE, dtype=torch.float64, leaf_n=3)
        chunked = PartitionedCliffordAlgebra(
            p,
            q,
            r,
            device=DEVICE,
            dtype=torch.float64,
            leaf_n=3,
            product_chunk_size=5,
        )

        assert vectorized._product_chunk_size >= vectorized._right_pair_count
        assert chunked._product_chunk_size < chunked._right_pair_count

        A = torch.randn(2, vectorized.dim, dtype=torch.float64)
        B = torch.randn(2, vectorized.dim, dtype=torch.float64)
        A_ref = A.clone().requires_grad_(True)
        B_ref = B.clone().requires_grad_(True)
        A_vectorized = A.clone().requires_grad_(True)
        B_vectorized = B.clone().requires_grad_(True)
        A_chunked = A.clone().requires_grad_(True)
        B_chunked = B.clone().requires_grad_(True)

        expected = reference.geometric_product(A_ref, B_ref)
        actual_vectorized = vectorized.geometric_product(A_vectorized, B_vectorized)
        actual_chunked = chunked.geometric_product(A_chunked, B_chunked)

        assert torch.allclose(actual_vectorized, expected, atol=1e-10, rtol=1e-10)
        assert torch.allclose(actual_chunked, expected, atol=1e-10, rtol=1e-10)

        weight = torch.linspace(-0.3, 0.4, vectorized.dim, dtype=torch.float64)
        (expected * weight).sum().backward()
        (actual_vectorized * weight).sum().backward()
        (actual_chunked * weight).sum().backward()

        assert torch.allclose(A_vectorized.grad, A_ref.grad, atol=1e-10, rtol=1e-10)
        assert torch.allclose(B_vectorized.grad, B_ref.grad, atol=1e-10, rtol=1e-10)
        assert torch.allclose(A_chunked.grad, A_ref.grad, atol=1e-10, rtol=1e-10)
        assert torch.allclose(B_chunked.grad, B_ref.grad, atol=1e-10, rtol=1e-10)

    def test_unit_rotor_chain_maintains_normalization_beyond_depth_threshold(self):
        algebra = PartitionedCliffordAlgebra(
            8,
            0,
            0,
            device=DEVICE,
            dtype=torch.float32,
            leaf_n=4,
            product_chunk_size=16,
            accumulation_dtype=torch.float64,
        )
        bivector = torch.zeros(1, algebra.dim, dtype=torch.float32)
        bivector[0, (1 << 0) | (1 << 6)] = 0.03125
        step = algebra._exp_bivector_closed(bivector)

        rotor = torch.zeros_like(step)
        rotor[0, 0] = 1.0
        identity = rotor.clone()

        max_error = 0.0
        for depth in range(1, 129):
            rotor = algebra.geometric_product(rotor, step)
            if depth % 16 == 0:
                rotor_norm = algebra.geometric_product(rotor, algebra.reverse(rotor))
                max_error = max(max_error, (rotor_norm - identity).abs().max().item())

        assert max_error < 5e-5

    def test_bridge_sign_for_high_times_low_vector(self):
        reference = CliffordAlgebra(5, 0, 0, device=DEVICE, dtype=torch.float64)
        algebra = PartitionedCliffordAlgebra(5, 0, 0, device=DEVICE, dtype=torch.float64, leaf_n=4)

        A = torch.zeros(1, 32, dtype=torch.float64)
        B = torch.zeros(1, 32, dtype=torch.float64)
        A[0, 16] = 1.0  # e5, the high block's first vector after a 4D low split
        B[0, 1] = 1.0  # e1, a low-block vector

        actual = algebra.geometric_product(A, B)
        expected = reference.geometric_product(A, B)

        assert actual[0, 17].item() == -1.0
        assert torch.equal(actual, expected)

    def test_null_cross_split_matches_core_kernel(self):
        _assert_matches_monolithic(4, 2, 2, leaf_n=4, shape=(2,))

    def test_minkowski_signature_matches_core_kernel(self):
        _assert_matches_monolithic(1, 3, 0, leaf_n=2, shape=(3,))

    def test_degenerate_signature_matches_core_kernel(self):
        _assert_matches_monolithic(2, 1, 2, leaf_n=2, shape=(2,))

    @pytest.mark.parametrize(
        ("p", "q", "r"),
        [
            (0, 3, 0),
            (0, 0, 4),
            (2, 0, 2),
            (0, 2, 2),
            (2, 2, 1),
        ],
    )
    def test_general_signature_sweep_matches_core_kernel(self, p, q, r):
        torch.manual_seed(19 + p * 11 + q * 7 + r)
        reference, algebra = _make_pair(p, q, r, leaf_n=2, product_chunk_size=3)
        A = torch.randn(2, algebra.dim, dtype=torch.float64)
        B = torch.randn(2, algebra.dim, dtype=torch.float64)

        for method_name in [
            "geometric_product",
            "wedge",
            "inner_product",
            "commutator",
            "anti_commutator",
            "left_contraction",
        ]:
            expected = getattr(reference, method_name)(A, B)
            actual = getattr(algebra, method_name)(A, B)
            assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10), method_name

        for method_name in [
            "reverse",
            "pseudoscalar_product",
            "dual",
            "grade_involution",
            "clifford_conjugation",
            "norm_sq",
        ]:
            expected = getattr(reference, method_name)(A)
            actual = getattr(algebra, method_name)(A)
            assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10), method_name

    def test_recursive_node_does_not_allocate_global_cayley_table(self):
        algebra = PartitionedCliffordAlgebra(8, 0, 0, device=DEVICE, leaf_n=6)

        assert not hasattr(algebra, "cayley_indices")
        assert not hasattr(algebra, "cayley_signs")
        assert algebra.left_sub.dim == 16
        assert algebra.right_sub.dim == 16

    def test_static_structural_sign_buffers_match_core_kernel(self):
        reference, algebra = _make_pair(3, 1, 1, leaf_n=2)

        for name in [
            "grade_index",
            "rev_signs",
            "_involution_signs",
            "conj_signs",
            "_cayley_diag",
            "_norm_sq_signs",
            "_hermitian_signs",
            "_ps_source",
            "_ps_signs",
            "_bv_indices",
            "bv_sq_scalar",
            "rc_action",
        ]:
            expected = getattr(reference, name)
            actual = getattr(algebra, name)
            if expected.dtype.is_floating_point:
                assert torch.allclose(actual, expected)
            else:
                assert torch.equal(actual, expected)

    def test_unary_operations_match_core_kernel(self):
        torch.manual_seed(31)
        reference, algebra = _make_pair(3, 1, 0, leaf_n=2)
        mv = torch.randn(2, algebra.dim, dtype=torch.float64)
        vectors = torch.randn(2, algebra.n, dtype=torch.float64)

        assert torch.allclose(algebra.embed_vector(vectors), reference.embed_vector(vectors))
        assert torch.allclose(algebra.get_grade_norms(mv), reference.get_grade_norms(mv), atol=1e-12, rtol=1e-12)
        for grade in range(algebra.num_grades):
            assert torch.allclose(algebra.grade_projection(mv, grade), reference.grade_projection(mv, grade))

        for method_name in [
            "reverse",
            "pseudoscalar_product",
            "dual",
            "grade_involution",
            "clifford_conjugation",
            "norm_sq",
        ]:
            expected = getattr(reference, method_name)(mv)
            actual = getattr(algebra, method_name)(mv)
            assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_binary_operations_match_core_kernel(self):
        torch.manual_seed(37)
        reference, algebra = _make_pair(3, 1, 0, leaf_n=2, product_chunk_size=3)
        A = torch.randn(2, 1, algebra.dim, dtype=torch.float64)
        B = torch.randn(1, 3, algebra.dim, dtype=torch.float64)

        for method_name in [
            "wedge",
            "inner_product",
            "commutator",
            "anti_commutator",
            "left_contraction",
        ]:
            expected = getattr(reference, method_name)(A, B)
            actual = getattr(algebra, method_name)(A, B)
            assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_bivector_vector_right_contraction_matches_core_kernel(self):
        torch.manual_seed(41)
        reference, algebra = _make_pair(4, 0, 0, leaf_n=2)
        A = reference.grade_projection(torch.randn(3, algebra.dim, dtype=torch.float64), 2)
        B = reference.grade_projection(torch.randn(3, algebra.dim, dtype=torch.float64), 1)

        expected = reference.right_contraction(A, B)
        actual = algebra.right_contraction(A, B)

        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_blade_and_versor_operations_match_core_kernel(self):
        torch.manual_seed(43)
        reference, algebra = _make_pair(3, 1, 0, leaf_n=2)
        mv = torch.randn(2, algebra.dim, dtype=torch.float64)
        blade = torch.zeros(2, algebra.dim, dtype=torch.float64)
        blade[:, 1] = 1.0
        blade[:, 2] = 0.25

        for method_name in ["blade_inverse", "blade_project", "blade_reject", "reflect", "versor_product"]:
            if method_name == "blade_inverse":
                expected = reference.blade_inverse(blade)
                actual = algebra.blade_inverse(blade)
            elif method_name in {"blade_project", "blade_reject"}:
                expected = getattr(reference, method_name)(mv, blade)
                actual = getattr(algebra, method_name)(mv, blade)
            elif method_name == "versor_product":
                expected = reference.versor_product(blade, mv)
                actual = algebra.versor_product(blade, mv)
            else:
                expected = getattr(reference, method_name)(mv, blade)
                actual = getattr(algebra, method_name)(mv, blade)
            assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_sandwich_variants_match_core_kernel(self):
        torch.manual_seed(47)
        reference, algebra = _make_pair(3, 0, 0, leaf_n=2)
        bivector = torch.zeros(4, algebra.dim, dtype=torch.float64)
        bivector[:, 3] = torch.linspace(0.05, 0.2, 4, dtype=torch.float64)
        rotors = reference.exp(bivector)

        x_batch_channel = torch.randn(2, 4, algebra.dim, dtype=torch.float64)
        x_same_batch = torch.randn(4, 3, algebra.dim, dtype=torch.float64)
        x_multi = torch.randn(2, 3, algebra.dim, dtype=torch.float64)

        expected = reference.per_channel_sandwich(rotors, x_batch_channel)
        actual = algebra.per_channel_sandwich(rotors, x_batch_channel)
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

        expected = reference.sandwich_product(rotors, x_same_batch)
        actual = algebra.sandwich_product(rotors, x_same_batch)
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

        expected = reference.multi_rotor_sandwich(rotors, x_multi)
        actual = algebra.multi_rotor_sandwich(rotors, x_multi)
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_exp_paths_match_core_kernel(self):
        reference, algebra = _make_pair(4, 0, 0, leaf_n=2)
        B = torch.zeros(2, algebra.dim, dtype=torch.float64)
        B[:, 3] = torch.tensor([0.125, -0.25], dtype=torch.float64)

        assert torch.allclose(
            algebra._exp_bivector_closed(B), reference._exp_bivector_closed(B), atol=1e-12, rtol=1e-12
        )
        assert torch.allclose(algebra.exp(B), reference.exp(B), atol=1e-12, rtol=1e-12)

        mv = torch.zeros(2, algebra.dim, dtype=torch.float64)
        mv[:, 0] = 0.1
        mv[:, 1] = torch.tensor([0.02, -0.03], dtype=torch.float64)
        mv[:, 3] = torch.tensor([0.04, 0.05], dtype=torch.float64)
        assert torch.allclose(
            algebra._exp_taylor(mv, order=6), reference._exp_taylor(mv, order=6), atol=1e-12, rtol=1e-12
        )

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_geometric_product_matches_eager(self):
        torch.manual_seed(53)
        algebra = PartitionedCliffordAlgebra(4, 1, 0, device=DEVICE, dtype=torch.float32, leaf_n=2)
        A = torch.randn(3, algebra.dim)
        B = torch.randn(3, algebra.dim)

        def product(x, y):
            return algebra.geometric_product(x, y)

        compiled_product = torch.compile(product, backend="aot_eager")

        assert torch.allclose(compiled_product(A, B), product(A, B), atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_training_backward_matches_eager(self):
        torch.manual_seed(59)
        eager_layer = _PartitionedProductLayer(4, 0, 0)
        compiled_layer = _PartitionedProductLayer(4, 0, 0)
        compiled_layer.weight.data.copy_(eager_layer.weight.data)

        x_eager = torch.randn(3, eager_layer.algebra.dim, requires_grad=True)
        x_compiled = x_eager.detach().clone().requires_grad_(True)

        eager_loss = eager_layer(x_eager).square().sum()
        eager_loss.backward()

        compiled_forward = torch.compile(compiled_layer, backend="aot_eager")
        compiled_loss = compiled_forward(x_compiled).square().sum()
        compiled_loss.backward()

        assert torch.allclose(compiled_loss, eager_loss, atol=1e-5, rtol=1e-5)
        assert torch.allclose(x_compiled.grad, x_eager.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(compiled_layer.weight.grad, eager_layer.weight.grad, atol=1e-4, rtol=1e-4)
