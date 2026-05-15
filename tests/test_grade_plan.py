import pytest
import torch

from core.config import make_algebra
from core.foundation.basis import (
    basis_count_for_grades,
    basis_index_tuple_for_grades,
    basis_indices_for_grades,
    expand_output_grades,
    geometric_product_output_grades,
    operation_coefficient,
    product_output_grades,
)
from core.foundation.layout import AlgebraSpec
from core.planning.flow import GradeFlow
from core.planning.layouts import build_product_request
from core.planning.planner import GradePlanner
from core.planning.policy import PlanningLimits
from core.planning.product import (
    GradeProductExecutor,
    build_grade_product_plan,
)
from core.planning.tree import build_grade_plan_tree
from core.planning.unary import build_unary_request
from core.runtime.algebra import CliffordAlgebra
from core.runtime.context import AlgebraContext
from core.runtime.multivector import Multivector

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def _project_to_grades(algebra, mv: torch.Tensor, grades: tuple[int, ...]) -> torch.Tensor:
    result = torch.zeros_like(mv)
    for grade in grades:
        result = result + algebra.grade_projection(mv, grade)
    return result


def _grade_only_input(algebra, batch: int, grades: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    mv = torch.zeros(batch, algebra.dim, dtype=torch.float64)
    indices = basis_indices_for_grades(algebra.n, grades, device=DEVICE)
    mv[:, indices] = torch.randn(batch, indices.numel(), dtype=torch.float64, generator=generator) * 0.1
    return mv


def test_grade_expansion_for_common_high_dim_paths():
    assert geometric_product_output_grades(1, 1, 16) == (0, 2)
    assert geometric_product_output_grades(2, 1, 16) == (1, 3)
    assert product_output_grades(2, 1, 16, op="wedge") == (3,)
    assert product_output_grades(2, 1, 16, op="commutator") == (1,)
    assert product_output_grades(2, 1, 16, op="anti_commutator") == (3,)
    assert expand_output_grades((0, 2), (1,), 16, op="gp") == (1, 3)
    assert expand_output_grades((1,), (1,), 16, op="wedge") == (2,)
    assert expand_output_grades((1,), (1,), 16, op="gp", project_grades=(0,)) == (0,)


def test_operation_coefficients_keep_wedge_as_exterior_product():
    # e12 and e3 commute, so the antisymmetric formula would vanish.  The
    # exterior product is instead the grade-sum part of the geometric product.
    assert operation_coefficient(3, 4, 3, 0, 0, "wedge") == 1.0
    assert operation_coefficient(3, 4, 3, 0, 0, "commutator") == 0.0
    assert operation_coefficient(3, 4, 3, 0, 0, "anti_commutator") == 2.0


def test_basis_indices_for_grades_are_combinatorial_and_high_dimensional():
    assert basis_index_tuple_for_grades(4, (1, 2)) == tuple(
        index for index in range(1 << 4) if index.bit_count() in {1, 2}
    )
    high = basis_index_tuple_for_grades(32, (1, 2))
    assert len(high) == basis_count_for_grades(32, (1, 2))
    assert high[0] == 1
    assert high[-1] == (1 << 31) | (1 << 30)


def test_basis_tensorization_reports_int64_bitmask_boundary():
    with pytest.raises(ValueError, match="torch.long basis bitmasks"):
        basis_indices_for_grades(64, (1,))


def test_grade_plan_tree_groups_routes_without_runtime_partition_backend():
    spec = AlgebraSpec(10, 4, 2)
    tree = build_grade_plan_tree(
        spec,
        left_grades=(1, 2),
        right_grades=(1,),
        output_grades=(0, 2),
        op="gp",
        chunk_pair_limit=128,
    )

    assert tree.output_grades == (0, 2)
    assert [(path.left_grade, path.right_grade, path.output_grades) for path in tree.paths] == [
        (1, 1, (0, 2)),
    ]
    assert tree.path_count == 1
    assert tree.estimated_pairs == 16 * 16
    assert tree.estimated_chunks == 2
    assert tree.path_for_grades(1, 1) is tree.paths[0]
    assert tree.path_for_grades(2, 1) is None


def test_product_request_infers_declared_layouts_and_output_grades():
    spec = AlgebraSpec(10, 4, 2)
    left = torch.zeros(2, spec.dim)
    right = torch.zeros(2, spec.dim)

    request = build_product_request(
        spec,
        left,
        right,
        left_grades=(1,),
        right_grades=(1,),
        op="gp",
        full_layout_allowed=False,
    )

    assert request.left_grades == (1,)
    assert request.right_grades == (1,)
    assert request.output_grades == (0, 2)
    assert not request.left_compact
    assert not request.right_compact


def test_product_request_detects_compact_tensors_from_layout_shape():
    spec = AlgebraSpec(6, 0, 0)
    layout = spec.layout((1,))
    left = torch.zeros(2, layout.dim)
    right = torch.zeros(2, layout.dim)

    request = build_product_request(
        spec,
        left,
        right,
        left_layout=layout,
        right_layout=layout,
        output_grades=(0, 2),
        op="gp",
    )

    assert request.left_compact
    assert request.right_compact


def test_unary_request_infers_projection_layout_without_full_layout():
    spec = AlgebraSpec(10, 4, 2)
    values = torch.zeros(2, spec.dim)

    request = build_unary_request(
        spec,
        values,
        op="grade_projection",
        output_grades=(1,),
        full_layout_allowed=False,
    )

    assert request.input_grades == (1,)
    assert request.output_grades == (1,)
    assert not request.input_compact


def test_grade_flow_propagates_embedding_unary_product_and_merge():
    spec = AlgebraSpec(8, 0, 0)
    vector = GradeFlow.vector(spec)
    scalar = GradeFlow.scalar(spec)

    product = vector.product(vector)
    projected = product.project((2,))
    merged = scalar.merge(projected)

    assert vector.grades == (1,)
    assert vector.unary("reverse").grades == (1,)
    assert product.grades == (0, 2)
    assert projected.grades == (2,)
    assert merged.grades == (0, 2)


def test_grade_layout_compact_dense_round_trip():
    spec = AlgebraSpec(4, 1, 1)
    layout = spec.layout((0, 2))
    dense = torch.randn(2, spec.dim, dtype=torch.float64, generator=torch.Generator().manual_seed(97))

    values = layout.compact(dense)
    materialized = layout.dense(values)

    assert values.shape[-1] == layout.dim
    assert torch.allclose(materialized[..., layout.indices_tensor(device=dense.device)], values)
    outside = torch.ones(spec.dim, dtype=torch.bool)
    outside[layout.indices_tensor()] = False
    assert materialized[..., outside].abs().sum().item() == 0.0


@pytest.mark.parametrize("op", ["gp", "wedge", "inner", "commutator", "anti_commutator"])
def test_static_grade_product_matches_dense_kernel_for_selected_grade_paths(op):
    algebra = CliffordAlgebra(4, 1, 1, device=DEVICE, dtype=torch.float64)
    left_grades = (1,)
    right_grades = (1, 2)
    output_grades = expand_output_grades(left_grades, right_grades, algebra.n, op=op)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=left_grades,
        right_grades=right_grades,
        output_grades=output_grades,
        op=op,
        device=DEVICE,
        dtype=torch.float64,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 3, left_grades, seed=101)
    B = _grade_only_input(algebra, 3, right_grades, seed=103)

    expected = _project_to_grades(algebra, getattr(algebra, _dense_method_name(op))(A, B), output_grades)
    actual = product.forward_dense(A, B)

    assert product.pair_count < algebra.dim * algebra.dim
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_wedge_dense_and_planned_paths_are_exterior_product_for_higher_grades():
    algebra = CliffordAlgebra(3, 0, 0, device=DEVICE, dtype=torch.float64)
    spec = AlgebraSpec.from_algebra(algebra)
    layout_2 = spec.layout((2,))
    layout_1 = spec.layout((1,))
    layout_3 = spec.layout((3,))

    e12 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e12[0, 3] = 1.0
    e3 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e3[0, 4] = 1.0
    expected = torch.zeros_like(e12)
    expected[0, 7] = 1.0

    dense = algebra.wedge(e12, e3)
    compact = algebra.wedge(
        layout_2.compact(e12),
        layout_1.compact(e3),
        left_layout=layout_2,
        right_layout=layout_1,
        output_grades=(3,),
        left_compact=True,
        right_compact=True,
        compact_output=True,
    )

    assert torch.allclose(dense, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(compact, layout_3.compact(expected), atol=1e-12, rtol=1e-12)


def test_wedge_chains_as_iterative_exterior_product():
    algebra = CliffordAlgebra(4, 0, 0, device=DEVICE, dtype=torch.float64)
    e1 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e2 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e3 = torch.zeros(1, algebra.dim, dtype=torch.float64)
    e1[0, 1] = 1.0
    e2[0, 2] = 1.0
    e3[0, 4] = 1.0
    expected = torch.zeros_like(e1)
    expected[0, 7] = 1.0

    e12 = algebra.wedge(e1, e2, left_grades=(1,), right_grades=(1,), output_grades=(2,))
    actual = algebra.wedge(e12, e3, left_grades=(2,), right_grades=(1,), output_grades=(3,))

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_wedge_plan_prunes_grade_route_pairs_before_coefficients():
    algebra = CliffordAlgebra(6, 0, 0, device=DEVICE, dtype=torch.float64)
    broad = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(1, 3),
        op="wedge",
        device=DEVICE,
        dtype=torch.float64,
    )
    exterior_only = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        op="wedge",
        device=DEVICE,
        dtype=torch.float64,
    )

    assert broad.pair_count == exterior_only.pair_count
    assert all(int(index).bit_count() == 3 for index in broad.output_indices.tolist())


def test_product_plan_owns_compact_position_buffers():
    algebra = CliffordAlgebra(4, 1, 0, device=DEVICE, dtype=torch.float64)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        op="gp",
        device=DEVICE,
        dtype=torch.float64,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 2, (1,), seed=109)
    B = _grade_only_input(algebra, 2, (1,), seed=111)

    left_positions = {index: position for position, index in enumerate(plan.left_layout.basis_indices)}
    right_positions = {index: position for position, index in enumerate(plan.right_layout.basis_indices)}
    expected_left = torch.tensor([left_positions[int(index)] for index in plan.left_indices], dtype=torch.long)
    expected_right = torch.tensor([right_positions[int(index)] for index in plan.right_indices], dtype=torch.long)

    assert torch.equal(plan.left_compact_positions.cpu(), expected_left)
    assert torch.equal(plan.right_compact_positions.cpu(), expected_right)
    assert torch.allclose(
        product.forward_compact(plan.left_layout.compact(A), plan.right_layout.compact(B)),
        product(A, B),
        atol=1e-12,
        rtol=1e-12,
    )


def test_product_executor_compact_forward_supports_different_layout_widths():
    algebra = CliffordAlgebra(4, 1, 0, device=DEVICE, dtype=torch.float64)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(1,),
        right_grades=(1, 2),
        output_grades=(0, 1, 2, 3),
        op="gp",
        device=DEVICE,
        dtype=torch.float64,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 2, (1,), seed=115)
    B = _grade_only_input(algebra, 2, (1, 2), seed=117)

    compact = product.forward_compact(plan.left_layout.compact(A), plan.right_layout.compact(B))
    dense = product(A, B)

    assert plan.left_layout.dim != plan.right_layout.dim
    assert torch.allclose(compact, dense, atol=1e-12, rtol=1e-12)


def test_algebra_projected_product_matches_dense_kernel_and_compact_output():
    algebra = CliffordAlgebra(4, 1, 1, device=DEVICE, dtype=torch.float64)
    A = _grade_only_input(algebra, 2, (1,), seed=113)
    B = _grade_only_input(algebra, 2, (1,), seed=127)

    dense_expected = _project_to_grades(algebra, algebra.geometric_product(A, B), (0, 2))
    dense_actual = algebra.projected_geometric_product(
        A,
        B,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
    )
    compact_actual = algebra.projected_geometric_product(
        A,
        B,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        compact_output=True,
    )

    assert torch.allclose(dense_actual, dense_expected, atol=1e-12, rtol=1e-12)
    assert compact_actual.shape[-1] == AlgebraSpec.from_algebra(algebra).layout((0, 2)).dim


def test_grade_planner_reuses_projected_product_executor():
    algebra = CliffordAlgebra(4, 1, 1, device=DEVICE, dtype=torch.float64)
    planner = GradePlanner(algebra)

    first = planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=torch.float64,
        device=DEVICE,
    )
    second = planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=torch.float64,
        device=DEVICE,
    )

    assert first is second


def test_algebra_product_executor_returns_preplanned_runtime_handle():
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)

    first = algebra.product_executor(
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
    )
    second = algebra.product_executor(
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
    )

    assert first is second
    assert first.left_grades == (1,)
    assert first.right_grades == (1,)
    assert first.output_grades == (0, 2)
    assert first.coefficients.dtype == algebra.dtype


def test_grade_planner_rekeys_cached_executor_after_dtype_move():
    algebra = CliffordAlgebra(4, 1, 1, device=DEVICE, dtype=torch.float64)
    executor = algebra.planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=algebra.dtype,
        device=DEVICE,
    )

    algebra.to(dtype=torch.float32)
    moved = algebra.planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=algebra.dtype,
        device=DEVICE,
    )

    assert moved is executor
    assert moved.coefficients.dtype == torch.float32


def test_multivector_compact_projected_product_keeps_dense_tensor_compatibility():
    algebra = CliffordAlgebra(4, 1, 1, device=DEVICE, dtype=torch.float64)
    A = Multivector(algebra, _grade_only_input(algebra, 2, (1,), seed=131)).compact((1,))
    B = Multivector(algebra, _grade_only_input(algebra, 2, (1,), seed=137)).compact((1,))

    result = A.projected_product(B, output_grades=(0, 2))
    expected = _project_to_grades(algebra, algebra.geometric_product(A.tensor, B.tensor), (0, 2))

    assert result.is_compact
    assert result.values.shape[-1] == result.layout.dim
    assert result.tensor.shape[-1] == algebra.dim
    assert torch.allclose(result.tensor, expected, atol=1e-12, rtol=1e-12)


def test_multivector_compact_projected_product_supports_mixed_dense_operand():
    algebra = CliffordAlgebra(4, 1, 1, device=DEVICE, dtype=torch.float64)
    A = Multivector(algebra, _grade_only_input(algebra, 2, (1,), seed=139)).compact((1,))
    B = Multivector(algebra, _grade_only_input(algebra, 2, (1,), seed=149))

    result = A.projected_product(B, output_grades=(0, 2), right_grades=(1,))
    expected = _project_to_grades(algebra, algebra.geometric_product(A.tensor, B.tensor), (0, 2))

    assert result.is_compact
    assert torch.allclose(result.tensor, expected, atol=1e-12, rtol=1e-12)


def test_make_algebra_returns_context_above_dense_threshold():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)

    assert isinstance(algebra, AlgebraContext)
    assert algebra.n == 16
    assert not algebra.allow_full_layout_products


def test_dense_policy_uses_context_by_default_and_explicit_dense_up_to_twelve():
    auto_dense = make_algebra(8, 0, 0, device=DEVICE, dtype=torch.float32)
    auto_context = make_algebra(9, 0, 0, device=DEVICE, dtype=torch.float32)
    explicit_dense = make_algebra(9, 0, 0, kernel="dense", device=DEVICE, dtype=torch.float32)

    assert isinstance(auto_dense, CliffordAlgebra)
    assert isinstance(auto_context, AlgebraContext)
    assert isinstance(explicit_dense, CliffordAlgebra)
    with pytest.raises(AssertionError):
        CliffordAlgebra(9, 0, 0, device=DEVICE, dtype=torch.float32)
    with pytest.raises(AssertionError):
        make_algebra(13, 0, 0, kernel="dense", device=DEVICE, dtype=torch.float32)


def test_dense_kernel_accepts_shared_planned_operation_kwargs():
    algebra = CliffordAlgebra(4, 1, 1, device=DEVICE, dtype=torch.float64)
    A = _grade_only_input(algebra, 2, (1,), seed=191)
    B = _grade_only_input(algebra, 2, (1,), seed=193)

    actual = algebra.geometric_product(
        A,
        B,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        compact_output=True,
    )
    expected = algebra.projected_geometric_product(
        A,
        B,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        compact_output=True,
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_dense_and_context_share_layout_indices_and_bivector_metric_signs():
    dense = CliffordAlgebra(3, 1, 0, device=DEVICE, dtype=torch.float64)
    context = make_algebra(3, 1, 0, kernel="context", device=DEVICE, dtype=torch.float64)

    assert torch.equal(dense.grade_indices((2,)), context.grade_indices((2,)))
    assert torch.allclose(dense.bivector_squared_signs(), context.bivector_squared_signs())


def test_context_projected_product_handles_high_dim_vector_product():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    A = torch.zeros(1, algebra.dim)
    B = torch.zeros(1, algebra.dim)
    A[0, 1] = 1.0
    B[0, 1] = 1.0
    B[0, 2] = 1.0

    values, layout = algebra.projected_geometric_product(
        A,
        B,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        compact_output=True,
        return_layout=True,
    )

    scalar_pos = layout.basis_indices.index(0)
    bivector_pos = layout.basis_indices.index(3)
    assert values.shape[-1] == layout.dim
    assert torch.allclose(values[0, scalar_pos], torch.tensor(1.0))
    assert torch.allclose(values[0, bivector_pos], torch.tensor(1.0))


def test_context_planned_unary_projection_and_reverse_avoid_full_layout():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    mv = torch.zeros(1, algebra.dim)
    mv[0, 1] = 2.0
    mv[0, 3] = 5.0

    projected, projected_layout = algebra.grade_projection(mv, 1, compact_output=True, return_layout=True)
    reversed_bivector = algebra.reverse(
        mv,
        input_grades=(2,),
        compact_output=True,
    )
    vector_pos = projected_layout.basis_indices.index(1)
    bivector_layout = algebra.layout((2,))
    bivector_pos = bivector_layout.basis_indices.index(3)

    assert torch.allclose(projected[0, vector_pos], torch.tensor(2.0))
    assert torch.allclose(reversed_bivector[0, bivector_pos], torch.tensor(-5.0))


def test_context_planned_unary_compact_reverse():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    values = torch.arange(layout.dim, dtype=torch.float32).unsqueeze(0)

    actual, output_layout = algebra.reverse(
        values,
        input_layout=layout,
        input_compact=True,
        compact_output=True,
        return_layout=True,
    )

    assert output_layout == layout
    assert torch.allclose(actual, -values)


def test_dense_kernel_planned_unary_handles_compact_layouts():
    algebra = CliffordAlgebra(6, 0, 0, device=DEVICE, dtype=torch.float32)
    layout = algebra.layout((2,))
    values = torch.arange(layout.dim, dtype=torch.float32).unsqueeze(0)

    actual, output_layout = algebra.reverse(
        values,
        input_layout=layout,
        input_compact=True,
        compact_output=True,
        return_layout=True,
    )

    assert output_layout == layout
    assert torch.allclose(actual, -values)


def test_multivector_compact_geometric_product_stays_compact_in_high_dimensions():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    result = Multivector(algebra, values=left, layout=vector_layout) * Multivector(
        algebra,
        values=right,
        layout=vector_layout,
    )

    assert result.is_compact
    assert result.layout.grades == (0, 2)
    scalar_pos = result.layout.basis_indices.index(0)
    assert torch.allclose(result.values[0, scalar_pos], torch.tensor(1.0))


def test_multivector_compact_binary_wrappers_do_not_unwrap_dense_tensors():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    bivector_layout = algebra.layout((2,))
    vector_layout = algebra.layout((1,))
    bivector = Multivector(algebra, values=torch.randn(2, bivector_layout.dim), layout=bivector_layout)
    vector = Multivector(algebra, values=torch.randn(2, vector_layout.dim), layout=vector_layout)

    results = [
        (bivector.geometric_product(vector), (1, 3)),
        (bivector.wedge(vector), (3,)),
        (bivector.inner(vector), (3,)),
        (bivector.commutator(vector), (1,)),
        (bivector.anti_commutator(vector), (3,)),
    ]

    for result, expected_grades in results:
        assert result.is_compact
        assert result.layout.grades == expected_grades


def test_multivector_compact_addition_merges_layouts_without_dense_materialization():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    bivector_layout = algebra.layout((2,))
    vector = Multivector(algebra, values=torch.ones(1, vector_layout.dim), layout=vector_layout)
    bivector = Multivector(algebra, values=2.0 * torch.ones(1, bivector_layout.dim), layout=bivector_layout)

    result = vector + bivector

    assert result.is_compact
    assert result.layout.grades == (1, 2)
    vector_values = vector.with_layout(result.layout).values
    bivector_values = bivector.with_layout(result.layout).values
    assert torch.allclose(result.values, vector_values + bivector_values)


def test_context_default_grades_drive_compact_product_without_callsite_metadata():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, default_grades=(1,))
    vector_layout = algebra.layout()
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    values, output_layout = algebra.geometric_product(
        left,
        right,
        compact_output=True,
        return_layout=True,
    )

    assert vector_layout.grades == (1,)
    assert output_layout.grades == (0, 2)
    assert torch.allclose(values[0, output_layout.basis_indices.index(0)], torch.tensor(1.0))


def test_context_declared_grades_infer_compact_operand_shapes():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    left[0, 0] = 1.0
    right[0, 0] = 1.0

    values, output_layout = algebra.projected_geometric_product(
        left,
        right,
        left_grades=(1,),
        right_grades=(1,),
        compact_output=True,
        return_layout=True,
    )

    assert output_layout.grades == (0, 2)
    assert torch.allclose(values[0, output_layout.basis_indices.index(0)], torch.tensor(1.0))


def test_context_projected_product_pairwise_mixed_compact_widths():
    algebra = make_algebra(16, 0, 0, kernel="context", device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))
    left = torch.randn(3, left_layout.dim)
    right = torch.randn(4, right_layout.dim)

    values, output_layout = algebra.projected_wedge(
        left,
        right,
        left_layout=left_layout,
        right_layout=right_layout,
        output_grades=(3,),
        pairwise=True,
        compact_output=True,
        return_layout=True,
    )
    executor = algebra.product_executor(
        op="wedge",
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        dtype=torch.float32,
        device=DEVICE,
    )

    assert output_layout.grades == (3,)
    assert values.shape == (3, 4, output_layout.dim)
    assert torch.allclose(values, executor.forward_pairwise_compact(left, right), atol=1e-6, rtol=1e-6)


def test_context_projected_product_suggests_pairwise_for_mismatched_item_axes():
    algebra = make_algebra(16, 0, 0, kernel="context", device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))
    left = torch.randn(3, left_layout.dim)
    right = torch.randn(4, right_layout.dim)

    with pytest.raises(ValueError, match="Use pairwise=True"):
        algebra.projected_wedge(
            left,
            right,
            left_layout=left_layout,
            right_layout=right_layout,
            output_grades=(3,),
            compact_output=True,
        )


def test_context_pairwise_projected_product_requires_item_axes():
    algebra = make_algebra(16, 0, 0, kernel="context", device=DEVICE, dtype=torch.float32)
    left_layout = algebra.layout((2,))
    right_layout = algebra.layout((1,))

    with pytest.raises(ValueError, match="explicit item axes"):
        algebra.projected_wedge(
            torch.randn(left_layout.dim),
            torch.randn(right_layout.dim),
            left_layout=left_layout,
            right_layout=right_layout,
            output_grades=(3,),
            pairwise=True,
            compact_output=True,
        )


def test_context_declared_product_requires_compact_output_without_dense_materialization():
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32)
    vector_layout = algebra.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)

    with pytest.raises(ValueError, match="Dense materialization is disabled"):
        algebra.projected_geometric_product(
            left,
            right,
            left_layout=vector_layout,
            right_layout=vector_layout,
        )


def test_high_dim_context_requires_declared_layout_for_products():
    algebra = make_algebra(13, 0, 0, device=DEVICE, dtype=torch.float32)
    A = torch.zeros(1, algebra.dim)
    B = torch.zeros(1, algebra.dim)

    with pytest.raises(ValueError, match="Declare active grades"):
        algebra.geometric_product(A, B)

    with pytest.raises(ValueError, match="Declare active grades"):
        algebra.reverse(A)


def test_context_requires_declared_grades_by_default_even_low_dimensional():
    context = make_algebra(4, 0, 0, kernel="context", device=DEVICE, dtype=torch.float64)

    with pytest.raises(ValueError, match="Declare active grades"):
        context.layout()


def test_context_warns_for_explicit_implicit_full_layout_fallback_between_eight_and_twelve():
    context = make_algebra(
        9,
        0,
        0,
        kernel="context",
        device=DEVICE,
        dtype=torch.float32,
        allow_full_layout_products=True,
    )

    with pytest.warns(RuntimeWarning, match="implicit full Cl\\(9,0,0\\) layout"):
        layout = context.layout()

    assert layout.grades == tuple(range(context.n + 1))
    assert context.allow_full_layout_products


def test_low_dim_context_can_use_full_layout_fallback():
    context = make_algebra(
        4,
        0,
        0,
        kernel="context",
        device=DEVICE,
        dtype=torch.float64,
        allow_full_layout_products=True,
    )
    dense = CliffordAlgebra(4, 0, 0, device=DEVICE, dtype=torch.float64)
    A = _grade_only_input(dense, 2, (1,), seed=163)
    B = _grade_only_input(dense, 2, (1,), seed=167)

    actual = context.geometric_product(A, B)
    expected = dense.geometric_product(A, B)

    assert context.allow_full_layout_products
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_context_static_product_cost_limits_raise_before_executor_build():
    limits = PlanningLimits(warn_lanes=512, max_lanes=512, warn_pairs=512, max_pairs=64)
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, planning_limits=limits)
    layout = algebra.layout((1,))
    left = torch.zeros(1, layout.dim)
    right = torch.zeros(1, layout.dim)

    with pytest.raises(ValueError, match="basis interactions"):
        algebra.projected_geometric_product(
            left,
            right,
            left_layout=layout,
            right_layout=layout,
            compact_output=True,
        )


def test_direct_product_executor_obeys_static_pair_limits():
    limits = PlanningLimits(warn_lanes=512, max_lanes=512, warn_pairs=512, max_pairs=64)
    algebra = make_algebra(16, 0, 0, device=DEVICE, dtype=torch.float32, planning_limits=limits)

    with pytest.raises(ValueError, match="basis interactions"):
        algebra.planner.product_executor(
            op="gp",
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            dtype=torch.float32,
            device=DEVICE,
        )


def test_context_static_layout_cost_limit_raises_before_basis_materialization():
    limits = PlanningLimits(warn_lanes=32, max_lanes=64, warn_pairs=512, max_pairs=1024)
    algebra = make_algebra(32, 0, 0, device=DEVICE, dtype=torch.float32, planning_limits=limits)

    with pytest.raises(ValueError, match="active lanes"):
        algebra.layout((1, 2))


def test_high_dimensional_vector_product_plan_avoids_full_basis_enumeration():
    algebra = make_algebra(
        32,
        0,
        0,
        device=DEVICE,
        dtype=torch.float32,
        planning_limits=PlanningLimits(max_lanes=4096, max_pairs=100_000),
    )
    vector_layout = algebra.layout((1,))
    executor = algebra.planner.product_executor(
        op="gp",
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        dtype=torch.float32,
        device=DEVICE,
    )

    assert vector_layout.dim == 32
    assert executor.output_dim == 1 + 32 * 31 // 2
    assert executor.pair_count == 32 * 32


def test_high_dimensional_product_plan_reports_int64_bitmask_boundary():
    algebra = make_algebra(
        64,
        0,
        0,
        device=DEVICE,
        dtype=torch.float32,
        planning_limits=PlanningLimits(max_lanes=4096, max_pairs=100_000),
    )

    with pytest.raises(ValueError, match="Current Torch-backed executors support bitmask tensorization up to n=63"):
        algebra.planner.product_executor(
            op="gp",
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            dtype=torch.float32,
            device=DEVICE,
        )


def test_context_static_product_cost_warns_near_configured_limits():
    limits = PlanningLimits(warn_lanes=512, max_lanes=512, warn_pairs=128, max_pairs=512)
    algebra = make_algebra(10, 4, 2, device=DEVICE, dtype=torch.float32, planning_limits=limits)
    layout = algebra.layout((1,))
    left = torch.zeros(1, layout.dim)
    right = torch.zeros(1, layout.dim)

    with pytest.warns(RuntimeWarning, match="basis interactions"):
        values = algebra.projected_geometric_product(
            left,
            right,
            left_layout=layout,
            right_layout=layout,
            compact_output=True,
        )

    assert values.shape[-1] == algebra.layout((0, 2)).dim


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_static_grade_product_compiles_fullgraph_with_aot_eager():
    algebra = CliffordAlgebra(5, 1, 0, device=DEVICE, dtype=torch.float32)
    plan = build_grade_product_plan(
        algebra.p,
        algebra.q,
        algebra.r,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        op="gp",
        device=DEVICE,
        dtype=torch.float32,
    )
    product = GradeProductExecutor(plan)
    A = _grade_only_input(algebra, 2, (1,), seed=107).to(dtype=torch.float32)
    B = _grade_only_input(algebra, 2, (1,), seed=109).to(dtype=torch.float32)

    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)

    expected = product(A, B)
    actual = compiled(A, B)

    assert actual.shape[-1] == product.output_dim
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_planned_unary_compiles_fullgraph_with_aot_eager():
    algebra = make_algebra(6, 0, 0, kernel="context", device=DEVICE, dtype=torch.float32)
    executor = algebra.planner.unary_executor(
        op="reverse",
        input_grades=(2,),
        dtype=torch.float32,
        device=DEVICE,
    )
    values = _grade_only_input(CliffordAlgebra(6, 0, 0, device=DEVICE), 2, (2,), seed=173).to(dtype=torch.float32)

    compiled = torch.compile(executor, backend="aot_eager", fullgraph=True)

    expected = executor(values)
    actual = compiled(values)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_algebra_projected_product_compiles_fullgraph_after_cache_warm():
    algebra = CliffordAlgebra(5, 1, 0, device=DEVICE, dtype=torch.float32)
    A = _grade_only_input(algebra, 2, (1,), seed=151).to(dtype=torch.float32)
    B = _grade_only_input(algebra, 2, (1,), seed=157).to(dtype=torch.float32)

    def product(x, y):
        return algebra.projected_geometric_product(
            x,
            y,
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            compact_output=True,
        )

    expected = product(A, B)
    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)
    actual = compiled(A, B)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_context_projected_product_compiles_fullgraph_from_cold_planner_cache():
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    algebra = AlgebraContext(6, 0, device=DEVICE, dtype=torch.float32)
    generator = torch.Generator(device=DEVICE).manual_seed(181)
    left = torch.randn(2, algebra.layout((1,)).dim, dtype=torch.float32, generator=generator)
    right = torch.randn(2, algebra.layout((1,)).dim, dtype=torch.float32, generator=generator)

    def product(x, y):
        return algebra.geometric_product(
            x,
            y,
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(0, 2),
            left_compact=True,
            right_compact=True,
            compact_output=True,
        )

    assert not algebra.planner._product_executors
    compiled = torch.compile(product, backend="aot_eager", fullgraph=True)
    actual = compiled(left, right)
    expected = product(left, right)

    assert len(algebra.planner._product_executors) == 1
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def _dense_method_name(op: str) -> str:
    if op == "gp":
        return "geometric_product"
    if op == "inner":
        return "inner_product"
    if op == "anti_commutator":
        return "anti_commutator"
    return op
