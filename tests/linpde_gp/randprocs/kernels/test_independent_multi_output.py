import numpy as np

import pytest

from linpde_gp.randprocs.covfuncs import (
    IndependentMultiOutputCovarianceFunction,
    Matern,
    TensorProduct,
)


@pytest.fixture
def random_lengthscales():
    rng = np.random.default_rng(12938422)
    return rng.random(size=(3, 2))


@pytest.fixture
def inputs_unbatched():
    rng = np.random.default_rng(9238134)
    return rng.random(size=(2,)), rng.random(size=(2,))


@pytest.fixture
def random_product_materns(random_lengthscales):
    return [
        TensorProduct(
            *(
                Matern((), nu=2.5, lengthscales=lengthscale)
                for lengthscale in cur_lengthscales
            )
        )
        for cur_lengthscales in random_lengthscales
    ]


def test_independence(random_product_materns, inputs_unbatched):
    mo = IndependentMultiOutputCovarianceFunction(*random_product_materns)
    (
        x0,
        x1,
    ) = inputs_unbatched
    res = mo(x0, x1)
    assert res.shape == (3, 3)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[i, j], 0.0)


def test_same_input(random_product_materns, inputs_unbatched):
    mo = IndependentMultiOutputCovarianceFunction(*random_product_materns)
    (
        x0,
        _,
    ) = inputs_unbatched
    res = mo(x0, x0)
    assert res.shape == (3, 3)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[i, j], 0.0)
        if i == j:
            np.testing.assert_allclose(res[i, j], 1.0)


@pytest.fixture
def inputs_batched():
    rng = np.random.default_rng(9238134)
    return rng.random(size=(42, 1, 2,)), rng.random(
        size=(
            1,
            85,
            2,
        )
    )


def test_batched_input(random_product_materns, inputs_batched):
    mo = IndependentMultiOutputCovarianceFunction(*random_product_materns)
    (
        x0,
        x1,
    ) = inputs_batched
    res = mo(x0, x1)
    assert res.shape == (42, 85, 3, 3)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[..., i, j], np.zeros((42, 85)))
