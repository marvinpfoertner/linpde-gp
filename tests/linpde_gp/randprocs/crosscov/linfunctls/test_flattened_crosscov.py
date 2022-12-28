import pytest

import numpy as np

import linpde_gp
from linpde_gp.linfunctls import FlattenedLinearFunctional

@pytest.fixture
def matern_1d():
    return linpde_gp.randprocs.covfuncs.Matern(())

@pytest.fixture
def linfunctl_1d():
    X = np.array([1., 2., 3.])
    return linpde_gp.linfunctls.DiracFunctional((), (), X)

@pytest.fixture
def input_vec_1d():
    rng = np.random.default_rng(2196001)
    return rng.normal(size=(10))

@pytest.mark.parametrize("argnum", [0, 1])
def test_flat_stays_unchanged(matern_1d, linfunctl_1d, input_vec_1d, argnum):
    linfunctl_flat = FlattenedLinearFunctional(linfunctl_1d)

    crosscov = linfunctl_1d(matern_1d, argnum=argnum)
    crosscov_flat = linfunctl_flat(matern_1d, argnum=argnum)

    res = crosscov(input_vec_1d)
    res_flat = crosscov_flat(input_vec_1d)

    assert res.shape == res_flat.shape
    np.testing.assert_allclose(res, res_flat)

    cov = linfunctl_1d(crosscov)
    cov_flat = linfunctl_flat(crosscov_flat)

    assert cov.shape == cov_flat.shape
    np.testing.assert_allclose(cov, cov_flat)

@pytest.fixture
def matern_multi():
    m1 = linpde_gp.randprocs.covfuncs.Matern((2,), nu=1.5, lengthscales=0.1)
    m2 = linpde_gp.randprocs.covfuncs.Matern((2,), nu=2.5, lengthscales=0.2)
    m3 = linpde_gp.randprocs.covfuncs.Matern((2,), nu=3.5, lengthscales=0.3)
    return linpde_gp.randprocs.covfuncs.IndependentMultiOutputCovarianceFunction(m1, m2, m3)

@pytest.fixture
def linfunctl_multi():
    rng = np.random.default_rng(7942198)
    X = rng.normal(size=(100, 2))
    return linpde_gp.linfunctls.DiracFunctional((2,), (3,), X)

@pytest.fixture
def input_vec_multi():
    rng = np.random.default_rng(389437)
    return rng.normal(size=(77, 2))

def test_multi(matern_multi, linfunctl_multi, input_vec_multi):
    linfunctl_flat = FlattenedLinearFunctional(linfunctl_multi)

    crosscov = linfunctl_multi(matern_multi, argnum=1)
    crosscov_flat = linfunctl_flat(matern_multi, argnum=1)

    res = crosscov(input_vec_multi)
    res_flat = crosscov_flat(input_vec_multi)

    assert res.shape == (77, 3, 100, 3)
    assert res_flat.shape == (77, 3, 300)
    np.testing.assert_allclose(res.reshape((77, 3, 300), order="C"), res_flat)

    cov = linfunctl_multi(crosscov)
    cov_flat = linfunctl_flat(crosscov_flat)

    assert cov.shape == (100, 3, 100, 3)
    assert cov_flat.shape == (300, 300)
    np.testing.assert_allclose(cov.reshape((100, 3, 300), order="C").reshape((300, 300), order="C"), cov_flat)