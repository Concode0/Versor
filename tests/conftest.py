import pytest
import torch

from core.algebra import CliffordAlgebra
from core.config import PartitionConfig, make_algebra

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


# -- High-dimensional partitioned algebras ------------------------------
@pytest.fixture
def partitioned_algebra_8d():
    return make_algebra(
        p=8,
        q=0,
        r=0,
        kernel="partitioned",
        device=DEVICE,
        dtype=torch.float64,
        partition=PartitionConfig(leaf_n=6, product_chunk_size=32),
    )


@pytest.fixture
def partitioned_algebra_12d():
    return make_algebra(
        p=12,
        q=0,
        r=0,
        kernel="partitioned",
        device=DEVICE,
        dtype=torch.float64,
        partition=PartitionConfig(leaf_n=6, product_chunk_size=64),
    )


@pytest.fixture
def partitioned_algebra_12d_mixed():
    return make_algebra(
        p=8,
        q=3,
        r=1,
        kernel="partitioned",
        device=DEVICE,
        dtype=torch.float64,
        partition=PartitionConfig(leaf_n=6, product_chunk_size=32),
    )


@pytest.fixture
def partitioned_algebra_16d():
    return make_algebra(
        p=10,
        q=4,
        r=2,
        kernel="partitioned",
        device=DEVICE,
        dtype=torch.float32,
        partition=PartitionConfig(leaf_n=6, product_chunk_size=8),
    )


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
