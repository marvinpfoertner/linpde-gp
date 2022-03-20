from typing import Union

import numpy as np
from probnum.typing import ShapeType

import pytest
import pytest_cases

import linpde_gp
from linpde_gp.linfuncops import diffops


def case_diffop_laplacian(input_shape: ShapeType) -> diffops.Laplacian:
    return -diffops.Laplacian(domain_shape=input_shape)


def case_diffop_spatial_laplacian(
    input_shape: ShapeType,
) -> Union[diffops.SpatialLaplacian, NotImplementedError]:
    if input_shape in ((), (1,)):
        return NotImplementedError(
            "`SpatialLaplacian` needs at least two dimensional input vectors"
        )

    return -3.1 * diffops.SpatialLaplacian(domain_shape=input_shape)


def case_diffop_directional_derivative(
    input_shape: ShapeType,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    rng = np.random.default_rng(9871239)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction ** 2))

    return diffops.DirectionalDerivative(direction)


@pytest_cases.fixture(scope="module")
@pytest_cases.parametrize_with_cases(
    "diffop_",
    cases=pytest_cases.THIS_MODULE,
    glob="diffop_*",
    scope="module",
)
def diffop(
    diffop_: Union[linpde_gp.linfuncops.JaxLinearOperator, NotImplementedError],
) -> linpde_gp.linfuncops.JaxLinearOperator:
    if isinstance(diffop_, NotImplementedError):
        pytest.skip(diffop_.args[0])

    return diffop_


def test_expquad_diffop_first(
    expquad: linpde_gp.randprocs.kernels.ExpQuad,
    expquad_jax: linpde_gp.randprocs.kernels.JaxKernel,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    expquad_diffop_first = diffop(expquad, argnum=0)
    expquad_diffop_first_jax = diffop(expquad_jax, argnum=0)

    np.testing.assert_allclose(
        expquad_diffop_first(X, None),
        expquad_diffop_first_jax(X, None),
    )

    np.testing.assert_allclose(
        expquad_diffop_first(X[:, None], X[None, :]),
        expquad_diffop_first_jax(X[:, None], X[None, :]),
    )


def test_expquad_diffop_second(
    expquad: linpde_gp.randprocs.kernels.ExpQuad,
    expquad_jax: linpde_gp.randprocs.kernels.JaxKernel,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    expquad_diffop_second = diffop(expquad, argnum=1)
    expquad_diffop_second_jax = diffop(expquad_jax, argnum=1)

    np.testing.assert_allclose(
        expquad_diffop_second(X, None),
        expquad_diffop_second_jax(X, None),
    )

    np.testing.assert_allclose(
        expquad_diffop_second(X[:, None], X[None, :]),
        expquad_diffop_second_jax(X[:, None], X[None, :]),
    )


def test_expquad_diffop_both(
    expquad: linpde_gp.randprocs.kernels.ExpQuad,
    expquad_jax: linpde_gp.randprocs.kernels.JaxKernel,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    expquad_diffop_both = diffop(diffop(expquad, argnum=1), argnum=0)
    expquad_diffop_both_jax = diffop(diffop(expquad_jax, argnum=1), argnum=0)

    np.testing.assert_allclose(
        expquad_diffop_both(X, None),
        expquad_diffop_both_jax(X, None),
    )

    np.testing.assert_allclose(
        expquad_diffop_both(X[:, None], X[None, :]),
        expquad_diffop_both_jax(X[:, None], X[None, :]),
    )


def test_expquad_diffop_both_jax_equals_call(
    expquad: linpde_gp.randprocs.kernels.ExpQuad,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    expquad_diffop_both = diffop(diffop(expquad, argnum=1), argnum=0)

    np.testing.assert_allclose(
        expquad_diffop_both.jax(X, None),
        expquad_diffop_both(X, None),
    )

    np.testing.assert_allclose(
        expquad_diffop_both.jax(X[:, None], X[None, :]),
        expquad_diffop_both(X[:, None], X[None, :]),
    )
