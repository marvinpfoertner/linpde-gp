from typing import Union

import numpy as np
from probnum.typing import ShapeType

import pytest
import pytest_cases

import linpde_gp
from linpde_gp.linfuncops import diffops

JaxLinearOperatorPair = tuple[
    linpde_gp.linfuncops.JaxLinearOperator, linpde_gp.linfuncops.JaxLinearOperator
]


def case_diffops_directional_derivative_laplacian(
    input_shape: ShapeType,
) -> JaxLinearOperatorPair:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction ** 2))

    return (
        diffops.DirectionalDerivative(direction),
        diffops.ScaledLaplaceOperator(input_shape, alpha=-1.3),
    )


def case_diffops_directional_derivative_spatial_laplacian(
    input_shape: ShapeType,
) -> Union[JaxLinearOperatorPair, NotImplementedError]:
    if input_shape == () or input_shape == (1,):
        return NotImplementedError(
            "`ScaledSpatialLaplacian` needs at least two dimensional input vectors"
        )

    rng = np.random.default_rng(4343609)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction ** 2))

    return (
        diffops.DirectionalDerivative(direction),
        diffops.ScaledSpatialLaplacian(input_shape, alpha=-2.1),
    )


@pytest_cases.fixture("module")
@pytest_cases.parametrize_with_cases(
    "diffop_permutation",
    cases=pytest_cases.THIS_MODULE,
    glob="diffops_*",
    scope="module",
)
def _diffops_permutation0(
    diffop_permutation: Union[JaxLinearOperatorPair, NotImplementedError],
) -> JaxLinearOperatorPair:
    if isinstance(diffop_permutation, NotImplementedError):
        pytest.skip(diffop_permutation.args[0])

    return diffop_permutation


def case_permutation0(_diffops_permutation0):
    return _diffops_permutation0


def case_permutation1(_diffops_permutation0):
    return _diffops_permutation0[1], _diffops_permutation0[0]


@pytest_cases.fixture("module")
@pytest_cases.parametrize_with_cases(
    "diffop_permutation",
    cases=pytest_cases.THIS_MODULE,
    glob="permutation*",
    scope="module",
)
def diffop_pair(diffop_permutation):
    return diffop_permutation


def test_L0_k_L1adj(k, k_jax, diffop_pair, X):
    L0, L1 = diffop_pair

    L0_k_L1adj = L0(L1(k, argnum=1), argnum=0)
    L0_k_L1adj_jax = L0(L1(k_jax, argnum=1), argnum=0)

    np.testing.assert_allclose(
        L0_k_L1adj(X[:, None], X[None, :]),
        L0_k_L1adj_jax(X[:, None], X[None, :]),
    )
