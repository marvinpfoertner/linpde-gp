import numpy as np
from scipy.linalg import block_diag

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
    num_outputs = mo.output_size_0
    res = mo(x0, x1)
    assert res.shape == (num_outputs, num_outputs)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[i, j], 0.0)


def test_same_input(random_product_materns, inputs_unbatched):
    mo = IndependentMultiOutputCovarianceFunction(*random_product_materns)
    (
        x0,
        _,
    ) = inputs_unbatched
    num_outputs = mo.output_size_0
    res = mo(x0, x0)
    assert res.shape == (num_outputs, num_outputs)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[i, j], 0.0)
        if i == j:
            np.testing.assert_allclose(res[i, j], 1.0)


@pytest.fixture
def inputs_batched():
    rng = np.random.default_rng(9238134)
    return rng.random(size=(10, 1, 2,)), rng.random(
        size=(
            1,
            15,
            2,
        )
    )


def test_batched_input(random_product_materns, inputs_batched):
    mo = IndependentMultiOutputCovarianceFunction(*random_product_materns)
    (
        x0,
        x1,
    ) = inputs_batched
    num_outputs = mo.output_size_0
    num_input_0 = np.prod(x0.shape[:-1])
    num_input_1 = np.prod(x1.shape[:-1])
    res = mo(x0, x1)
    assert res.shape == (num_input_0, num_input_1, num_outputs, num_outputs)
    for (i, j) in np.ndindex(3, 3):
        if i != j:
            np.testing.assert_allclose(res[..., i, j], np.zeros((10, 15)))

def test_linop_same_input(random_product_materns, inputs_batched):
    mo = IndependentMultiOutputCovarianceFunction(*random_product_materns)
    (
        x0,
        _
    ) = inputs_batched
    num_outputs = mo.output_size_0
    num_input = np.prod(x0.shape[:-1])
    res = mo.linop(x0, x0)
    assert res.shape == (num_outputs * num_input, num_outputs * num_input)
    res = res @ np.eye(res.shape[1])
    manual_block_diag = block_diag(*[rpm.matrix(x0, x0) for rpm in random_product_materns])
    np.testing.assert_allclose(res, manual_block_diag)

def test_linop_different_inputs(random_product_materns, inputs_batched):
    """In particular, this test case will have non-square blocks on the 
    diagonal."""
    mo = IndependentMultiOutputCovarianceFunction(*random_product_materns)
    (
        x0,
        x1
    ) = inputs_batched
    num_outputs = mo.output_size_0
    num_input_0 = np.prod(x0.shape[:-1])
    num_input_1 = np.prod(x1.shape[:-1])
    res = mo.linop(x0, x1)
    assert res.shape == (num_outputs * num_input_0, num_outputs * num_input_1)
    res = res @ np.eye(res.shape[1])
    manual_block_diag = block_diag(*[rpm.matrix(x0, x1) for rpm in random_product_materns])
    np.testing.assert_allclose(res, manual_block_diag)
