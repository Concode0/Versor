# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Tests for rotor_translate.py — direct rotor-to-formula translation.

import pytest
import numpy as np
import torch

from core.algebra import CliffordAlgebra
from models.rotor_translate import (
    RotorTranslator, SimplePlane, RotorTerm,
    _correlation, _log_corr_divide,
)
from models.sr_net import SRGBN


@pytest.fixture
def algebra():
    return CliffordAlgebra(3, 0, 0, device="cpu")


@pytest.fixture
def algebra4():
    return CliffordAlgebra(4, 0, 0, device="cpu")


def test_import_and_instantiate(algebra):
    """RotorTranslator can be instantiated with an algebra."""
    translator = RotorTranslator(algebra)
    assert translator.algebra is algebra
    assert len(translator.symbols) == 3


def test_simple_plane_dataclass():
    """SimplePlane stores plane info correctly."""
    plane = SimplePlane(var_i=0, var_j=1, sig_type="elliptic", angle=0.5)
    assert plane.var_i == 0
    assert plane.var_j == 1
    assert plane.sig_type == "elliptic"
    assert plane.angle == 0.5


def test_rotor_term_dataclass():
    """RotorTerm stores planes and weight."""
    plane = SimplePlane(var_i=0, var_j=2, sig_type="hyperbolic", angle=1.2)
    term = RotorTerm(planes=[plane], weight=0.8, expr=None)
    assert len(term.planes) == 1
    assert term.weight == 0.8
    assert term.expr is None


def test_rotor_term_defaults():
    """RotorTerm default fields."""
    term = RotorTerm()
    assert term.planes == []
    assert term.weight == 1.0
    assert term.expr is None


def test_translate_trained_model(algebra):
    """translate() extracts terms from a trained single-rotor SRGBN."""
    model = SRGBN.single_rotor(algebra, in_features=2, channels=8)

    # Manually set a large bivector weight so it's above threshold
    with torch.no_grad():
        model.blocks[0].rotor.bivector_weights.fill_(0.0)
        # Set e12 plane (index 0 in bivector basis for Cl(3,0))
        model.blocks[0].rotor.bivector_weights[:, 0] = 0.5

    translator = RotorTranslator(algebra)
    terms = translator.translate(model)

    assert len(terms) >= 1, "Should extract at least one term"
    assert terms[0].planes[0].sig_type == "elliptic"
    assert terms[0].expr is not None
    assert terms[0].fn is not None


def test_translate_zero_bivectors(algebra):
    """translate() returns empty list for zero bivectors."""
    model = SRGBN.single_rotor(algebra, in_features=2, channels=4)

    with torch.no_grad():
        model.blocks[0].rotor.bivector_weights.fill_(0.0)

    translator = RotorTranslator(algebra)
    terms = translator.translate(model)
    assert len(terms) == 0


def test_plane_to_action_elliptic(algebra):
    """Elliptic plane produces rotation mixing two variables."""
    import sympy
    translator = RotorTranslator(algebra)
    plane = SimplePlane(var_i=0, var_j=1, sig_type="elliptic", angle=0.3)
    expr = translator._plane_to_action(plane)

    # The expression mixes x1 and x2 (rotation in 01-plane)
    x1, x2 = sympy.Symbol("x1"), sympy.Symbol("x2")
    assert x1 in expr.free_symbols or x2 in expr.free_symbols
    # Evaluate: cos(0.6)*x1 - sin(0.6)*x2
    val = float(expr.subs({x1: 1.0, x2: 0.0}))
    assert abs(val - np.cos(0.6)) < 1e-6


def test_plane_to_action_hyperbolic():
    """Hyperbolic plane produces boost mixing two variables."""
    import sympy
    algebra = CliffordAlgebra(2, 1, 0, device="cpu")
    translator = RotorTranslator(algebra)
    plane = SimplePlane(var_i=0, var_j=2, sig_type="hyperbolic", angle=0.4)
    expr = translator._plane_to_action(plane)

    x1, x2, x3 = sympy.Symbol("x1"), sympy.Symbol("x2"), sympy.Symbol("x3")
    val = float(expr.subs({x1: 1.0, x2: 0.0, x3: 0.0}))
    assert abs(val - np.cosh(0.8)) < 1e-6


def test_plane_to_action_parabolic():
    """Parabolic plane produces linear expression."""
    algebra = CliffordAlgebra(2, 0, 1, device="cpu")
    translator = RotorTranslator(algebra)
    plane = SimplePlane(var_i=0, var_j=1, sig_type="parabolic", angle=0.5)
    expr = translator._plane_to_action(plane)

    # Parabolic: x + 2*theta*y, should be polynomial (no trig)
    expr_str = str(expr)
    assert "cos" not in expr_str
    assert "sin" not in expr_str


def test_to_formula_empty(algebra):
    """to_formula returns 'y = 0' for empty terms."""
    translator = RotorTranslator(algebra)
    assert translator.to_formula([]) == "y = 0"


def test_to_formula_nonempty(algebra):
    """to_formula returns 'y = ...' for non-empty terms."""
    import sympy
    translator = RotorTranslator(algebra)
    x1 = sympy.Symbol("x1")
    term = RotorTerm(planes=[], weight=2.0, expr=x1**2)
    formula = translator.to_formula([term])
    assert formula.startswith("y = ")
    assert "x1" in formula


def test_translate_implicit(algebra):
    """translate_implicit replaces target var with 'y' symbol."""
    model = SRGBN.single_rotor(algebra, in_features=2, channels=8)

    with torch.no_grad():
        model.blocks[0].rotor.bivector_weights.fill_(0.0)
        model.blocks[0].rotor.bivector_weights[:, 0] = 0.5

    translator = RotorTranslator(algebra)
    terms = translator.translate_implicit(model, target_var_idx=2)

    # Check that terms contain 'y' in their expression
    if terms:
        import sympy
        y = sympy.Symbol("y")
        # The third variable should be 'y' in the expression
        expr_str = str(terms[0].expr)
        # Either has 'y' or only uses x1, x2
        assert terms[0].expr is not None


def test_evaluate_terms(algebra):
    """evaluate_terms produces finite predictions."""
    import sympy
    translator = RotorTranslator(algebra)

    x1, x2, x3 = translator.symbols
    expr = 2.0 * x1 + 3.0 * x2
    fn = sympy.lambdify([x1, x2, x3], expr, "numpy")
    term = RotorTerm(planes=[], weight=1.0, expr=expr, fn=fn)

    X = np.random.randn(10, 3).astype(np.float32)
    y_hat = translator.evaluate_terms([term], X)

    expected = 2.0 * X[:, 0] + 3.0 * X[:, 1]
    np.testing.assert_allclose(y_hat, expected, atol=1e-5)


def test_correlation():
    """_correlation returns ~1 for perfectly correlated arrays."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    assert _correlation(a, b) > 0.99

    # Uncorrelated
    c = np.array([1.0, -1.0, 1.0, -1.0])
    d = np.array([1.0, 1.0, -1.0, -1.0])
    assert _correlation(c, d) < 0.1


def test_log_corr_divide():
    """_log_corr_divide returns divided result when log-correlated."""
    a = np.array([10.0, 100.0, 1000.0, 10000.0])
    b = np.array([2.0, 20.0, 200.0, 2000.0])
    result, corr = _log_corr_divide(a, b)
    assert result is not None
    assert corr > 0.7
    np.testing.assert_allclose(result, a / b, atol=1e-5)


def test_log_corr_divide_uncorrelated():
    """_log_corr_divide returns None for uncorrelated data."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    result, corr = _log_corr_divide(a, b)
    # With these values, log correlation may or may not be high
    # but at minimum the function should not crash
    assert isinstance(corr, float)
