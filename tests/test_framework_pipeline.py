import pytest
import torch
import torch.nn as nn

from core.planning import PlanningLimits
from core.runtime.algebra import CliffordAlgebra
from core.runtime.context import AlgebraContext
from layers import ProductLayer, WedgeLayer
from optimizers import make_riemannian_optimizer

pytestmark = pytest.mark.unit


def test_product_layer_dense_matches_algebra(algebra_3d):
    left = torch.randn(4, 5, algebra_3d.dim)
    right = torch.randn(4, 5, algebra_3d.dim)
    layer = ProductLayer(algebra_3d)

    actual = layer(left, right)
    expected = algebra_3d.geometric_product(left, right)

    assert torch.allclose(actual, expected)


def test_wedge_layer_declared_grades_match_planned_algebra(algebra_3d):
    left = algebra_3d.embed_vector(torch.randn(4, 5, algebra_3d.n))
    right = algebra_3d.embed_vector(torch.randn(4, 5, algebra_3d.n))
    layer = WedgeLayer(
        algebra_3d,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(2,),
    )

    actual = layer(left, right)
    expected = algebra_3d.wedge(left, right, left_grades=(1,), right_grades=(1,), output_grades=(2,))

    assert torch.allclose(actual, expected)


def test_product_layer_pairwise_compact_widths_match_dense_reference():
    context = AlgebraContext(p=5, q=0, device="cpu")
    dense = CliffordAlgebra(p=5, q=0, device="cpu")
    left_layout = context.layout((2,))
    right_layout = context.layout((1,))
    output_layout = context.layout((3,))

    left = torch.randn(2, 3, left_layout.dim)
    right = torch.randn(2, 4, right_layout.dim)
    layer = WedgeLayer(
        context,
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
        left_compact=True,
        right_compact=True,
        compact_output=True,
        pairwise=True,
    )

    actual = layer(left, right)
    cache_size = len(context.planner._product_executors)
    repeated = layer(left, right)
    expected_dense = dense.wedge(
        left_layout.dense(left).unsqueeze(2),
        right_layout.dense(right).unsqueeze(1),
        left_grades=(2,),
        right_grades=(1,),
        output_grades=(3,),
    )
    expected = output_layout.compact(expected_dense)

    assert actual.shape == (2, 3, 4, output_layout.dim)
    assert torch.allclose(actual, expected)
    assert torch.allclose(repeated, actual)
    assert cache_size == 1
    assert len(context.planner._product_executors) == cache_size


def test_compact_layer_pipeline_trains_with_riemannian_optimizer_factory():
    context = AlgebraContext(p=6, q=0, device="cpu")
    vector_layout = context.layout((1,))
    trivector_layout = context.layout((3,))

    class CompactPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.wedge_vectors = WedgeLayer(
                context,
                left_grades=(1,),
                right_grades=(1,),
                output_grades=(2,),
                compact_output=True,
            )
            self.wedge_trivector = WedgeLayer(
                context,
                left_grades=(2,),
                right_grades=(1,),
                output_grades=(3,),
                left_compact=True,
                right_compact=True,
                compact_output=True,
            )
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, left_vector, right_vector, third_vector):
            bivector = self.wedge_vectors(left_vector, right_vector)
            return self.scale * self.wedge_trivector(bivector, third_vector)

    dense_left = context.embed_vector(torch.randn(8, context.n))
    dense_right = context.embed_vector(torch.randn(8, context.n))
    compact_third = vector_layout.compact(context.embed_vector(torch.randn(8, context.n)))
    model = CompactPipeline()
    optimizer = make_riemannian_optimizer(model, context, optimizer="adam", lr=0.01)

    output = model(dense_left, dense_right, compact_third)
    assert output.shape == (8, trivector_layout.dim)
    assert len(context.planner._product_executors) == 2

    loss = output.square().mean()
    loss.backward()
    optimizer.step()

    assert model.scale.grad is not None
    assert torch.isfinite(model.scale).all()


def test_product_layer_uses_context_planning_limits():
    limits = PlanningLimits(warn_lanes=32, max_lanes=512, warn_pairs=32, max_pairs=64)
    context = AlgebraContext(p=16, q=0, device="cpu", planning_limits=limits)
    vector_layout = context.layout((1,))
    left = torch.zeros(1, vector_layout.dim)
    right = torch.zeros(1, vector_layout.dim)
    layer = ProductLayer(
        context,
        left_grades=(1,),
        right_grades=(1,),
        output_grades=(0, 2),
        left_compact=True,
        right_compact=True,
        compact_output=True,
    )

    with pytest.raises(ValueError, match="basis interactions"):
        layer(left, right)
