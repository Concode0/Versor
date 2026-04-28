import pytest
from core.algebra import CliffordAlgebra

DEVICE = "cpu"


# -- Function-scoped (default) ------------------------------------------
@pytest.fixture
def algebra_2d():
    return CliffordAlgebra(p=2, q=0, device=DEVICE)


@pytest.fixture
def algebra_3d():
    return CliffordAlgebra(p=3, q=0, device=DEVICE)


@pytest.fixture
def algebra_4d():
    return CliffordAlgebra(p=4, q=0, device=DEVICE)


@pytest.fixture
def algebra_spacetime():
    return CliffordAlgebra(p=1, q=3, device=DEVICE)


@pytest.fixture
def algebra_minkowski():
    return CliffordAlgebra(p=2, q=1, device=DEVICE)


@pytest.fixture
def algebra_conformal():
    return CliffordAlgebra(p=4, q=1, device=DEVICE)


# -- Module-scoped (used by test_geodesic.py - exact name match) ----------
@pytest.fixture(scope="module")
def alg2():
    return CliffordAlgebra(p=2, q=0, device=DEVICE)


@pytest.fixture(scope="module")
def alg3():
    return CliffordAlgebra(p=3, q=0, device=DEVICE)


@pytest.fixture(scope="module")
def alg31():
    return CliffordAlgebra(p=3, q=1, device=DEVICE)
