from typing import Union

import numpy as np
from probnum.typing import ShapeType

import pytest
import pytest_cases

import linpde_gp
from linpde_gp.linfuncops import diffops

DifferentialOperatorPair = tuple[
    linpde_gp.linfuncops.JaxLinearOperator, linpde_gp.linfuncops.JaxLinearOperator
]


def case_diffops_directional_derivative_laplacian(
    input_shape: ShapeType,
) -> DifferentialOperatorPair:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction ** 2))

    return (
        diffops.DirectionalDerivative(direction),
        diffops.Laplacian(input_shape, alpha=-1.3),
    )


def case_diffops_directional_derivative_spatial_laplacian(
    input_shape: ShapeType,
) -> Union[DifferentialOperatorPair, NotImplementedError]:
    if input_shape in ((), (1,)):
        return NotImplementedError(
            "`SpatialLaplacian` needs at least two dimensional input vectors"
        )

    rng = np.random.default_rng(4343609)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction ** 2))

    return (
        diffops.DirectionalDerivative(direction),
        diffops.SpatialLaplacian(input_shape, alpha=-2.1),
    )


@pytest_cases.fixture("module")
@pytest_cases.parametrize_with_cases(
    "diffop_pair_",
    cases=pytest_cases.THIS_MODULE,
    glob="diffops_*",
    scope="module",
)
def _diffops(
    diffop_pair_: Union[DifferentialOperatorPair, NotImplementedError],
) -> DifferentialOperatorPair:
    if isinstance(diffop_pair_, NotImplementedError):
        pytest.skip(diffop_pair_.args[0])

    return diffop_pair_


def case_diffop_pair_permutation0(
    _diffops: DifferentialOperatorPair,
) -> DifferentialOperatorPair:
    return _diffops


def case_diffop_pair_permutation1(
    _diffops: DifferentialOperatorPair,
) -> DifferentialOperatorPair:
    return _diffops[1], _diffops[0]


@pytest_cases.fixture("module")
@pytest_cases.parametrize_with_cases(
    "diffop_pair_",
    cases=pytest_cases.THIS_MODULE,
    glob="diffop_pair_permutation*",
    scope="module",
)
def diffop_pair(diffop_pair_: DifferentialOperatorPair) -> DifferentialOperatorPair:
    return diffop_pair_


def test_expquad_diffop0_diffop1(
    expquad: linpde_gp.randprocs.kernels.ExpQuad,
    expquad_jax: linpde_gp.randprocs.kernels.JaxKernel,
    diffop_pair: DifferentialOperatorPair,
    X: np.ndarray,
):
    diffop0, diffop1 = diffop_pair

    expquad_diffop0_diffop1 = diffop0(diffop1(expquad, argnum=1), argnum=0)
    expquad_diffop0_diffop1_jax = diffop0(diffop1(expquad_jax, argnum=1), argnum=0)

    np.testing.assert_allclose(
        expquad_diffop0_diffop1(X[:, None], X[None, :]),
        expquad_diffop0_diffop1_jax(X[:, None], X[None, :]),
    )


def test_expquad_diffop0_diffop1_jax_equals_call(
    expquad: linpde_gp.randprocs.kernels.ExpQuad,
    diffop_pair: DifferentialOperatorPair,
    X: np.ndarray,
):
    diffop0, diffop1 = diffop_pair

    k_diffop0_diffop1 = diffop0(diffop1(expquad, argnum=1), argnum=0)

    np.testing.assert_allclose(
        k_diffop0_diffop1.jax(X[:, None], X[None, :]),
        k_diffop0_diffop1(X[:, None], X[None, :]),
    )
