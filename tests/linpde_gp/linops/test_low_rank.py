import numpy as np
import probnum as pn
import linpde_gp
import pytest


@pytest.fixture
def dim() -> int:
    return 100


@pytest.fixture
def rank() -> int:
    return 20


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(31)


@pytest.fixture
def operator(dim: int, rank: int, rng: np.random.Generator) -> pn.linops.LinearOperator:
    A = pn.linops.Scaling(rng.uniform(0.9, 1.1, size=dim))
    C = pn.linops.Scaling(rng.uniform(1.0, 2.0, size=rank))

    U = rng.uniform(low=0.0, high=1.0, size=(dim, rank))
    V = rng.uniform(low=0.0, high=1.0, size=(rank, dim))

    return linpde_gp.linops.LowRankUpdate(A, U, C, V)


def test_inv(operator: pn.linops.LinearOperator):
    np.testing.assert_allclose(
        operator.inv().todense(), np.linalg.inv(operator.todense())
    )


def test_det(operator: pn.linops.LinearOperator):
    np.testing.assert_allclose(operator.det(), np.linalg.det(operator.todense()))
