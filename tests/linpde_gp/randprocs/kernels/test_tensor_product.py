import numpy as np
import scipy.stats

import pytest
from pytest_cases import fixture

from linpde_gp.randprocs import covfuncs


@fixture
@pytest.mark.parametrize("d", [1, 2, 3])
def lengthscales(d: int) -> np.ndarray:
    rng = np.random.default_rng(390435023409 + d)
    return rng.gamma(1.0, size=(d,)) + 0.2


@fixture
def k_tprod(lengthscales: np.ndarray) -> covfuncs.TensorProduct:
    return covfuncs.TensorProduct(
        *(
            covfuncs.ExpQuad((), lengthscales=lengthscale)
            for lengthscale in lengthscales
        )
    )


@fixture
def k_expquad(lengthscales: np.ndarray) -> covfuncs.ExpQuad:
    return covfuncs.ExpQuad((lengthscales.size,), lengthscales=lengthscales)


@fixture
def xs(lengthscales: np.ndarray) -> np.ndarray:
    sampler = scipy.stats.qmc.Sobol(lengthscales.size, seed=109134809)
    xs01 = sampler.random_base2(7)  # 2^7 = 128 points
    return scipy.stats.qmc.scale(xs01, -3.0 * lengthscales, 3.0 * lengthscales)


def test_tprod_expquad_equal(
    k_tprod: covfuncs.TensorProduct, k_expquad: covfuncs.ExpQuad, xs: np.ndarray
):
    """Test whether the tensor product of `ExpQuad` covariance functions produces the
    same result as a multivariate `ExpQuad` covariance function."""
    Kxx_tprod = k_tprod.matrix(xs)
    Kxx_expquad = k_expquad.matrix(xs)

    np.testing.assert_allclose(Kxx_tprod, Kxx_expquad)
