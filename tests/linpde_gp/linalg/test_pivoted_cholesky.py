import numpy as np
import probnum as pn
import probnum.problems.zoo.linalg
import pytest

import linpde_gp


@pytest.fixture
def dim() -> int:
    return 40


@pytest.fixture
def A(dim: int) -> np.ndarray:
    return probnum.problems.zoo.linalg.random_spd_matrix(
        np.random.default_rng(), dim=dim
    )


def test_valid_matrix_square_root(dim: int, A: np.ndarray):
    L = linpde_gp.linalg.pivoted_cholesky(A, k=dim)

    np.testing.assert_almost_equal(L @ L.T, A)
