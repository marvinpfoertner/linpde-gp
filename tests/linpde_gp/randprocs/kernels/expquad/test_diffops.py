from typing import Union

import numpy as np
from probnum.typing import ShapeType

import pytest
import pytest_cases

import linpde_gp
from linpde_gp.linfuncops import diffops


def case_diffop_scaled_laplace(
    input_shape: ShapeType,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    return diffops.Laplacian(domain_shape=input_shape, alpha=-1.0)


def case_diffop_scaled_spatial_laplacian(
    input_shape: ShapeType,
) -> Union[diffops.SpatialLaplacian, NotImplementedError]:
    if input_shape in ((), (1,)):
        return NotImplementedError(
            "`SpatialLaplacian` needs at least two dimensional input vectors"
        )

    return diffops.SpatialLaplacian(domain_shape=input_shape, alpha=-3.1)


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
    diffop_: linpde_gp.linfuncops.LinearFunctionOperator,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    if isinstance(diffop_, NotImplementedError):
        pytest.skip(diffop_.args[0])

    return diffop_


def test_kLa(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    k_jax: linpde_gp.randprocs.kernels.JaxKernel,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    kLa = diffop(k, argnum=1)
    kLa_jax = diffop(k_jax, argnum=1)

    np.testing.assert_allclose(
        kLa(X[:, None], X[None, :]),
        kLa_jax(X[:, None], X[None, :]),
    )


def test_Lk(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    k_jax: linpde_gp.randprocs.kernels.JaxKernel,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    Lk = diffop(k, argnum=0)
    Lk_jax = diffop(k_jax, argnum=0)

    np.testing.assert_allclose(
        Lk(X[:, None], X[None, :]),
        Lk_jax(X[:, None], X[None, :]),
    )


def test_LkLa(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    k_jax: linpde_gp.randprocs.kernels.JaxKernel,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    LkLa = diffop(diffop(k, argnum=1), argnum=0)
    LkLa_jax = diffop(diffop(k_jax, argnum=1), argnum=0)

    np.testing.assert_allclose(
        LkLa(X[:, None], X[None, :]),
        LkLa_jax(X[:, None], X[None, :]),
    )


def test_LkLa_jax_equals_call(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    diffop: linpde_gp.linfuncops.JaxLinearOperator,
    X: np.ndarray,
):
    LkLa = diffop(diffop(k, argnum=1), argnum=0)

    np.testing.assert_allclose(
        LkLa.jax(X[:, None], X[None, :]),
        LkLa(X[:, None], X[None, :]),
    )
