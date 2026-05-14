import math

import pytest
import torch

from core.runtime.algebra import CliffordAlgebra
from layers.blocks.attention import GeometricProductAttention

pytestmark = pytest.mark.unit

DEVICE = "cpu"


def _reference_attention_score(algebra, q_head, k_head, bivector_weight):
    product = algebra.geometric_product(q_head.unsqueeze(3), algebra.reverse(k_head).unsqueeze(2))
    score_g0 = product[..., 0].sum(-1)

    g2_idx = algebra.grade_masks[2].nonzero(as_tuple=False).squeeze(-1)
    if g2_idx.numel() > 0:
        g2 = torch.index_select(product, -1, g2_idx)
        score_g2 = g2.pow(2).sum(dim=(-1, -2)).sqrt()
    else:
        score_g2 = torch.zeros_like(score_g0)

    scale = math.sqrt(q_head.shape[3] * algebra.dim)
    return (score_g0 + bivector_weight * score_g2) / scale


def test_attention_dense_chunked_score_matches_direct_product():
    algebra = CliffordAlgebra(3, 0, 0, device=DEVICE, dtype=torch.float64)
    attn = GeometricProductAttention(
        algebra,
        channels=4,
        num_heads=2,
        causal=False,
        bivector_weight=0.25,
        score_blade_chunk_size=1,
        score_precompute_limit=0,
    )
    q_head = torch.randn(2, 2, 3, 2, algebra.dim, dtype=torch.float64)
    k_head = torch.randn(2, 2, 4, 2, algebra.dim, dtype=torch.float64)

    actual = attn._compute_score(q_head, k_head)
    expected = _reference_attention_score(algebra, q_head, k_head, attn.bivector_weight)

    assert not hasattr(attn, "_g2_b_idx")
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_attention_forward_shape_after_score_refactor():
    algebra = CliffordAlgebra(3, 0, 0, device=DEVICE, dtype=torch.float32)
    attn = GeometricProductAttention(algebra, channels=4, num_heads=2, causal=False)
    x = torch.randn(2, 5, 4, algebra.dim)

    y = attn(x)

    assert y.shape == x.shape


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_attention_dense_score_compiles_fullgraph():
    algebra = CliffordAlgebra(4, 0, 0, device=DEVICE, dtype=torch.float32)
    attn = GeometricProductAttention(algebra, channels=4, num_heads=2, causal=False)
    q_head = torch.randn(1, 2, 3, 2, algebra.dim)
    k_head = torch.randn(1, 2, 4, 2, algebra.dim)

    def score(q, k):
        return attn._compute_score(q, k)

    expected = score(q_head, k_head)
    compiled = torch.compile(score, backend="aot_eager", fullgraph=True)
    actual = compiled(q_head, k_head)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
