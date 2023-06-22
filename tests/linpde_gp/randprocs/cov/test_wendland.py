import numpy as np
from scipy.spatial.distance import pdist, squareform

import pytest
from pytest_cases import fixture

from linpde_gp.functions import RationalPolynomial
from linpde_gp.randprocs.covfuncs import (
    WendlandCovarianceFunction,
    WendlandFunction,
    WendlandPolynomial,
)


@fixture
@pytest.mark.parametrize(
    "poly_params",
    [
        (1, [1], 1, 0),
        (3, [1, 3], 1, 1),
        (5, [1, 5, 8], 1, 2),
        (2, [1], 3, 0),
        (4, [1, 4], 3, 1),
        (6, [3, 18, 35], 3, 2),
        (8, [1, 8, 25, 32], 3, 3),
        (3, [1], 5, 0),
        (5, [1, 5], 5, 1),
        (7, [1, 7, 16], 5, 2),
    ],
)
def true_poly_params(poly_params):
    prefactor_degree, other_coeffs, d, k = poly_params
    poly = RationalPolynomial(other_coeffs)
    prefactor = RationalPolynomial([1, -1])
    for _ in range(prefactor_degree):
        poly *= prefactor
    return poly, d, k


def test_wendland_polynomial(true_poly_params):
    true_polynomial, d, k = true_poly_params
    our_polynomial = WendlandPolynomial(d, k)

    # Equal up to constant factor
    div, rem = divmod(true_polynomial, our_polynomial)
    assert all(coeff == 0.0 for coeff in rem.coefficients)
    assert div.degree == 0


@fixture
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def r(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=100)


@pytest.mark.parametrize("d", [1, 3, 5])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_wendland_function(r, d, k):
    polynomial = WendlandPolynomial(d, k)
    func = WendlandFunction(d, k)

    true_result = np.where(r <= 1, polynomial(r), 0)

    np.testing.assert_allclose(func(r), true_result)


@fixture
@pytest.mark.parametrize("d", [1, 3, 5])
def x0(d):
    rng = np.random.default_rng(234923490)
    return rng.uniform(-1, 1, size=(100, d))


@pytest.mark.parametrize("k", [0, 1, 2])
def test_wendland_covariance_function(x0, k):
    cov = WendlandCovarianceFunction(x0.shape[1:], k)
    func = WendlandFunction(x0.shape[1], k)
    dist_mat = squareform(pdist(x0, metric="euclidean"))

    true_result = func(dist_mat)
    np.testing.assert_allclose(cov.matrix(x0), true_result)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_wendland_covariance_function_linop(x0, k):
    cov = WendlandCovarianceFunction(x0.shape[1:], k)
    linop_res = cov.linop(x0)

    # Need to be a bit more lenient with the comparison here because
    # KeOps does computations in floats, not Fractions
    # I don't think there's a way around this.
    np.testing.assert_allclose(
        linop_res @ np.eye(linop_res.shape[1]), cov.matrix(x0), rtol=0.0, atol=1e-12
    )
