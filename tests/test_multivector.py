# Tests for core/multivector.py operator overloading

import pytest
import torch

from core.algebra import CliffordAlgebra
from core.multivector import Multivector


@pytest.fixture
def alg():
    return CliffordAlgebra(3, 0, device="cpu")


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)


def rand_mv(alg, rng, batch=4):
    dim = 2**alg.n
    return Multivector(alg, torch.randn(batch, dim, generator=rng))


# ---- constructors ----


def test_from_vectors(alg):
    v = torch.randn(4, 3)
    mv = Multivector.from_vectors(alg, v)
    assert mv.shape[-1] == 2**alg.n


def test_scalar(alg):
    mv = Multivector.scalar(alg, 3.0, batch_shape=(2,))
    assert mv.tensor[0, 0].item() == 3.0
    assert mv.tensor[0, 1:].abs().sum().item() == 0.0


# ---- arithmetic operators ----


def test_add_multivector(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a + b
    assert torch.allclose(c.tensor, a.tensor + b.tensor)


def test_add_scalar(alg, rng):
    a = rand_mv(alg, rng)
    c = a + 1.0
    assert torch.allclose(c.tensor, a.tensor + 1.0)


def test_radd(alg, rng):
    a = rand_mv(alg, rng)
    c = 2.0 + a
    assert torch.allclose(c.tensor, a.tensor + 2.0)


def test_sub_multivector(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a - b
    assert torch.allclose(c.tensor, a.tensor - b.tensor)


def test_rsub(alg, rng):
    a = rand_mv(alg, rng)
    c = 1.0 - a
    assert torch.allclose(c.tensor, 1.0 - a.tensor)


def test_neg(alg, rng):
    a = rand_mv(alg, rng)
    c = -a
    assert torch.allclose(c.tensor, -a.tensor)


def test_mul_scalar(alg, rng):
    a = rand_mv(alg, rng)
    c = a * 3.0
    assert torch.allclose(c.tensor, a.tensor * 3.0)


def test_rmul_scalar(alg, rng):
    a = rand_mv(alg, rng)
    c = 3.0 * a
    assert torch.allclose(c.tensor, a.tensor * 3.0)


def test_truediv(alg, rng):
    a = rand_mv(alg, rng)
    c = a / 2.0
    assert torch.allclose(c.tensor, a.tensor / 2.0)


# ---- geometric product ----


def test_mul_geometric(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a * b
    expected = alg.geometric_product(a.tensor, b.tensor)
    assert torch.allclose(c.tensor, expected)


def test_geometric_product_method(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    assert torch.allclose((a * b).tensor, a.geometric_product(b).tensor)


# ---- wedge / inner ----


def test_xor_wedge(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a ^ b
    expected = alg.wedge(a.tensor, b.tensor)
    assert torch.allclose(c.tensor, expected)


def test_wedge_method(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    assert torch.allclose((a ^ b).tensor, a.wedge(b).tensor)


def test_or_inner(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a | b
    expected = alg.inner_product(a.tensor, b.tensor)
    assert torch.allclose(c.tensor, expected)


def test_inner_method(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    assert torch.allclose((a | b).tensor, a.inner(b).tensor)


# ---- reversion ----


def test_invert_reversion(alg, rng):
    a = rand_mv(alg, rng)
    c = ~a
    expected = alg.reverse(a.tensor)
    assert torch.allclose(c.tensor, expected)


def test_reverse_method(alg, rng):
    a = rand_mv(alg, rng)
    assert torch.allclose((~a).tensor, a.reverse().tensor)


# ---- grade projection ----


def test_grade(alg, rng):
    a = rand_mv(alg, rng)
    g0 = a.grade(0)
    expected = alg.grade_projection(a.tensor, 0)
    assert torch.allclose(g0.tensor, expected)


# ---- involutions ----


def test_grade_involution(alg, rng):
    a = rand_mv(alg, rng)
    c = a.grade_involution()
    expected = alg.grade_involution(a.tensor)
    assert torch.allclose(c.tensor, expected)


def test_clifford_conjugation(alg, rng):
    a = rand_mv(alg, rng)
    c = a.clifford_conjugation()
    expected = alg.clifford_conjugation(a.tensor)
    assert torch.allclose(c.tensor, expected)


# ---- dual ----


def test_dual(alg, rng):
    a = rand_mv(alg, rng)
    c = a.dual()
    expected = alg.dual(a.tensor)
    assert torch.allclose(c.tensor, expected)


# ---- norms ----


def test_norm_sq(alg, rng):
    a = rand_mv(alg, rng)
    ns = a.norm_sq()
    expected = alg.norm_sq(a.tensor)
    assert torch.allclose(ns, expected)


def test_norm(alg, rng):
    a = rand_mv(alg, rng)
    n = a.norm()
    assert n.shape[0] == a.shape[0]


def test_get_grade_norms(alg, rng):
    a = rand_mv(alg, rng)
    gn = a.get_grade_norms()
    expected = alg.get_grade_norms(a.tensor)
    assert torch.allclose(gn, expected)


# ---- contractions ----


def test_left_contraction(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a.left_contraction(b)
    expected = alg.left_contraction(a.tensor, b.tensor)
    assert torch.allclose(c.tensor, expected)


def test_right_contraction(alg, rng):
    # right_contraction expects bivector _| vector shapes
    bv = alg.grade_projection(rand_mv(alg, rng).tensor, 2)
    vec = alg.embed_vector(torch.randn(4, 3))
    a = Multivector(alg, bv)
    b = Multivector(alg, vec)
    c = a.right_contraction(b)
    expected = alg.right_contraction(bv, vec)
    assert torch.allclose(c.tensor, expected)


# ---- commutators ----


def test_commutator(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a.commutator(b)
    expected = alg.commutator(a.tensor, b.tensor)
    assert torch.allclose(c.tensor, expected)


def test_anti_commutator(alg, rng):
    a, b = rand_mv(alg, rng), rand_mv(alg, rng)
    c = a.anti_commutator(b)
    expected = alg.anti_commutator(a.tensor, b.tensor)
    assert torch.allclose(c.tensor, expected)


# ---- inverse ----


def test_inverse(alg):
    # A unit vector should have a well-defined inverse
    v = torch.zeros(8)
    v[1] = 1.0  # e1 in Cl(3,0)
    mv = Multivector(alg, v)
    inv = mv.inverse()
    product = mv * inv
    # Should be ~1 (scalar)
    assert abs(product.tensor[0].item() - 1.0) < 1e-5


# ---- exp ----


def test_exp(alg):
    # exp of zero bivector = 1
    bv = torch.zeros(8)
    mv = Multivector(alg, bv)
    r = mv.exp()
    assert abs(r.tensor[0].item() - 1.0) < 1e-5


# ---- sandwich / versor / reflect ----


def test_sandwich(alg):
    # Identity rotor (scalar=1) should leave x unchanged
    rotor = Multivector.scalar(alg, 1.0)
    x = Multivector(alg, alg.embed_vector(torch.randn(3)))
    result = rotor.sandwich(x)
    assert torch.allclose(result.tensor, x.tensor, atol=1e-5)


def test_versor_product(alg):
    # Identity versor should leave x unchanged
    V = Multivector.scalar(alg, 1.0)
    x = Multivector(alg, alg.embed_vector(torch.randn(3)))
    result = V.versor_product(x)
    assert torch.allclose(result.tensor, x.tensor, atol=1e-5)


def test_reflect(alg):
    # Reflect e1 through plane ⊥ to e1 → should flip sign of e1 component
    x = Multivector(alg, alg.embed_vector(torch.tensor([1.0, 0.0, 0.0])))
    n = Multivector(alg, alg.embed_vector(torch.tensor([1.0, 0.0, 0.0])))
    r = x.reflect(n)
    expected = alg.reflect(x.tensor, n.tensor)
    assert torch.allclose(r.tensor, expected, atol=1e-5)


# ---- projections ----


def test_blade_project_reject(alg):
    x = Multivector(alg, alg.embed_vector(torch.tensor([1.0, 2.0, 3.0])))
    b = Multivector(alg, alg.embed_vector(torch.tensor([1.0, 0.0, 0.0])))
    proj = x.blade_project(b)
    rej = x.blade_reject(b)
    # proj + rej ≈ x
    recon = proj + rej
    assert torch.allclose(recon.tensor, x.tensor, atol=1e-5)


# ---- torch interop ----


def test_to(alg, rng):
    a = rand_mv(alg, rng)
    b = a.to(dtype=torch.float64)
    assert b.dtype == torch.float64


def test_detach(alg, rng):
    a = rand_mv(alg, rng).requires_grad_()
    b = a.detach()
    assert not b.tensor.requires_grad


def test_clone(alg, rng):
    a = rand_mv(alg, rng)
    b = a.clone()
    assert torch.allclose(a.tensor, b.tensor)
    b.tensor[0, 0] = 999.0
    assert a.tensor[0, 0] != 999.0


def test_requires_grad(alg, rng):
    a = rand_mv(alg, rng)
    a.requires_grad_()
    assert a.tensor.requires_grad


def test_shape_device_dtype(alg, rng):
    a = rand_mv(alg, rng)
    assert a.shape == a.tensor.shape
    assert a.device == a.tensor.device
    assert a.dtype == a.tensor.dtype


# ---- algebra mismatch ----


def test_algebra_mismatch_raises():
    a1 = CliffordAlgebra(3, 0, device="cpu")
    a2 = CliffordAlgebra(2, 1, device="cpu")
    m1 = Multivector(a1, torch.randn(8))
    m2 = Multivector(a2, torch.randn(8))
    with pytest.raises(ValueError, match="Algebra mismatch"):
        m1 + m2


# ---- NotImplemented for bad types ----


def test_notimplemented():
    alg = CliffordAlgebra(2, 0, device="cpu")
    mv = Multivector(alg, torch.randn(4))
    with pytest.raises(TypeError):
        mv + "string"
    with pytest.raises(TypeError):
        mv * [1, 2]
    with pytest.raises(TypeError):
        mv ^ 3.0
    with pytest.raises(TypeError):
        mv | 3.0


# ---- repr ----


def test_repr(alg, rng):
    a = rand_mv(alg, rng)
    r = repr(a)
    assert "Cl(3,0,0)" in r
    assert "Multivector" in r
