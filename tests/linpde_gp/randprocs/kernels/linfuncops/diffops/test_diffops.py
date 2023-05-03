import functools
import operator

import jax
import numpy as np
from probnum.typing import ShapeType
import scipy.stats

import pytest
from pytest_cases import parametrize_with_cases

from .cases import CovarianceFunctionDiffOpTestCase, case_modules


def X(input_shape: ShapeType) -> np.ndarray:
    d = functools.reduce(operator.mul, input_shape, 1)
    sampler = scipy.stats.qmc.Sobol(d, seed=109134809 + d)
    xs01 = sampler.random_base2(7)  # 2^7 = 128 points
    xs = scipy.stats.qmc.scale(xs01, -3.0, 3.0)
    return xs.reshape((-1,) + input_shape)


@parametrize_with_cases("test_case", cases=case_modules)
def test_L0kL1_expected_type(test_case: CovarianceFunctionDiffOpTestCase):
    if test_case.expected_type is not None:
        # pylint: disable=unidiomatic-typecheck
        assert type(test_case.L0kL1) == test_case.expected_type


@parametrize_with_cases("test_case", cases=case_modules)
def test_L0kL1(test_case: CovarianceFunctionDiffOpTestCase):
    Xs = X(test_case.k.input_shape)

    L0kL1_XX = test_case.L0kL1(Xs[:, None], Xs[None, :])
    L0kL1_XX_jax = test_case.L0kL1_jax(Xs[:, None], Xs[None, :])

    nan_mask = np.isnan(L0kL1_XX_jax)

    if np.any(nan_mask):
        L0kL1_XX_jax[nan_mask] = L0kL1_XX[nan_mask]

    np.testing.assert_allclose(L0kL1_XX, L0kL1_XX_jax, atol=1e-14)


@parametrize_with_cases("test_case", cases=case_modules)
def test_L0kL1_call_equals_jax(test_case: CovarianceFunctionDiffOpTestCase):
    Xs = X(test_case.k.input_shape)

    L0kL1_jax = jax.jit(test_case.L0kL1.jax)

    np.testing.assert_allclose(
        test_case.L0kL1(Xs[:, None], Xs[None, :]),
        L0kL1_jax(Xs[:, None], Xs[None, :]),
    )


@parametrize_with_cases("test_case", cases=case_modules)
def test_L0kL1_linop_keops_equals_no_keops(test_case: CovarianceFunctionDiffOpTestCase):
    Xs = X(test_case.k.input_shape)

    keops_linop = test_case.L0kL1.linop(Xs, Xs)
    keops_mat = keops_linop @ np.eye(keops_linop.shape[1])

    no_keops_linop = test_case.L0kL1.linop(Xs, Xs)
    try:
        no_keops_linop._use_keops = False  # pylint: disable=protected-access
        no_keops_mat = no_keops_linop @ np.eye(no_keops_linop.shape[1])
    except AttributeError:
        # for _use_keops
        pytest.skip("No KeOps implementation available.")

    np.testing.assert_allclose(no_keops_mat, keops_mat)
