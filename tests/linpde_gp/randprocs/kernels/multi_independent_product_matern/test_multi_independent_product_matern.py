import pytest
from pytest_cases import parametrize_with_cases
import numpy as np

from linpde_gp.randprocs.kernels import MultiIndependentProductMatern

@parametrize_with_cases('input_dim,output_dim,lengthscales,p')
def test_illegal_parameters(input_dim, output_dim, lengthscales, p):
    with pytest.raises(ValueError):
        mip = MultiIndependentProductMatern(input_dim, output_dim, lengthscales, p)

@pytest.fixture
def random_lengthscales():
    rng = np.random.default_rng(12938422)
    return rng.random(size=(3, 2))

@pytest.fixture
def inputs_unbatched():
    rng = np.random.default_rng(9238134)
    return rng.random(size=(2,)), rng.random(size=(2,))

def test_independence(random_lengthscales, inputs_unbatched):
    mip = MultiIndependentProductMatern(2, 3, random_lengthscales)
    x0, x1, = inputs_unbatched
    res = mip(x0, x1)
    assert res.shape == (3, 3)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[i, j], 0.)

def test_same_input(random_lengthscales, inputs_unbatched):
    mip = MultiIndependentProductMatern(2, 3, random_lengthscales)
    x0, _, = inputs_unbatched
    res = mip(x0, x0)
    assert res.shape == (3, 3)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[i, j], 0.) 
        if i == j:
            np.testing.assert_allclose(res[i, j], 1.)

@pytest.fixture
def inputs_batched():
    rng = np.random.default_rng(9238134)
    return rng.random(size=(42, 1, 2,)), rng.random(size=(1, 85, 2,))

def test_batched_input(random_lengthscales, inputs_batched):
    mip = MultiIndependentProductMatern(2, 3, random_lengthscales)
    x0, x1, = inputs_batched
    res = mip(x0, x1)
    assert res.shape == (42, 85, 3, 3)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[..., i, j], np.zeros((42, 85))) 