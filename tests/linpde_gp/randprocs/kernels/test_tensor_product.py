import numpy as np
import scipy.stats

import pytest
from pytest_cases import fixture

from linpde_gp.randprocs import kernels


@fixture
@pytest.mark.parametrize("d", [1, 2, 3])
def lengthscales(d: int) -> np.ndarray:
    rng = np.random.default_rng(390435023409 + d)
    return rng.gamma(1.0, size=(d,)) + 0.2


@fixture
def k_tprod(lengthscales: np.ndarray) -> kernels.TensorProductKernel:
    return kernels.TensorProductKernel(
        *(kernels.ExpQuad((), lengthscales=lengthscale) for lengthscale in lengthscales)
    )


@fixture
def k_expquad(lengthscales: np.ndarray) -> kernels.ExpQuad:
    return kernels.ExpQuad((lengthscales.size,), lengthscales=lengthscales)


@fixture
def xs(lengthscales: np.ndarray) -> np.ndarray:
    sampler = scipy.stats.qmc.Sobol(lengthscales.size, seed=109134809)
    xs01 = sampler.random_base2(7)  # 2^7 = 128 points
    return scipy.stats.qmc.scale(xs01, -3.0 * lengthscales, 3.0 * lengthscales)


def test_tprod_expquad_equal(
    k_tprod: kernels.TensorProductKernel, k_expquad: kernels.ExpQuad, xs: np.ndarray
):
    """Test whether the tensor product of `ExpQuad` covariance function produces the
    same result as a multivariate `ExpQuad` kernel."""
    Kxx_tprod = k_tprod.matrix(xs)
    Kxx_expquad = k_expquad.matrix(xs)

    np.testing.assert_allclose(Kxx_tprod, Kxx_expquad)
