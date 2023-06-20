import numpy as np

import pytest
from pytest_cases import fixture

from linpde_gp.linfuncops.diffops import Derivative, DirectionalDerivative
from linpde_gp.randprocs.covfuncs import Matern


@fixture
def matern():
    return Matern((), nu=2.5)


@fixture
def derivative():
    return Derivative(1)


@fixture
def dir_deriv():
    return DirectionalDerivative(1.0)


@fixture
@pytest.mark.parametrize("seed", [0, 1, 2])
def x0(seed):
    rng = np.random.RandomState(seed=234909139 + seed)
    return rng.randn(10)


@fixture
@pytest.mark.parametrize("seed", [0, 1, 2])
def x1(seed):
    rng = np.random.RandomState(seed=456094377 + seed)
    return rng.randn(10)


def test_deriv_equals_dir_deriv_one_side(matern, derivative, dir_deriv, x0, x1):
    np.testing.assert_allclose(
        derivative(matern, argnum=1)(x0, x1), dir_deriv(matern, argnum=1)(x0, x1)
    )


def test_deriv_equals_dir_deriv_both_sides(matern, derivative, dir_deriv, x0, x1):
    np.testing.assert_allclose(
        derivative(derivative(matern, argnum=1), argnum=0)(x0, x1),
        dir_deriv(dir_deriv(matern, argnum=1), argnum=0)(x0, x1),
    )
