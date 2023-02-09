import numpy as np

import pytest

from linpde_gp.randprocs import kernels


@pytest.fixture
def k():
    return kernels.Matern((), nu=3.5)


# def test_numpy_jax_equal(k):
#     xs = np.linspace(-2.0, 2.0, 100)

#     kxx = k(xs[:, None], xs[None, :])
#     kxx_jax = k.jax(xs[:, None], xs[None, :])

#     np.testing.assert_allclose(kxx, kxx_jax)
