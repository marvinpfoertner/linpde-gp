from typing import Union

import numpy as np
from probnum.typing import ShapeType

import pytest
import pytest_cases

import linpde_gp


def case_diffop_scaled_laplace(
    input_shape: ShapeType,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    return linpde_gp.problems.pde.diffops.ScaledLaplaceOperator(
        domain_shape=input_shape, alpha=-1.0
    )


def case_diffop_scaled_spatial_laplacian(
    input_shape: ShapeType,
) -> Union[linpde_gp.problems.pde.diffops.ScaledSpatialLaplacian, NotImplementedError]:
    if input_shape == () or input_shape == (1,):
        return NotImplementedError(
            "`ScaledSpatialLaplacian` needs at least two dimensional input vectors"
        )

    return linpde_gp.problems.pde.diffops.ScaledSpatialLaplacian(
        domain_shape=input_shape, alpha=-1.0
    )


def case_diffop_directional_derivative(
    input_shape: ShapeType,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    rng = np.random.default_rng(9871239)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction ** 2))

    return linpde_gp.problems.pde.diffops.DirectionalDerivative(direction)


@pytest_cases.fixture(scope="module")
@pytest_cases.parametrize_with_cases(
    "diffop",
    cases=pytest_cases.THIS_MODULE,
    glob="diffop_*",
    scope="module",
)
def L(
    diffop: linpde_gp.linfuncops.LinearFunctionOperator,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    if isinstance(diffop, NotImplementedError):
        pytest.skip(diffop.args[0])

    return diffop


@pytest_cases.fixture(scope="module")
def Lk_jax(
    k_jax: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return L(k_jax, argnum=0)


@pytest_cases.fixture(scope="module")
def kLa_jax(
    k_jax: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return L(k_jax, argnum=1)


@pytest_cases.fixture(scope="module")
def LkLa_jax(
    kLa_jax: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return L(kLa_jax, argnum=0)


def test_Lk(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
    Lk_jax: linpde_gp.randprocs.kernels.JaxKernel,
    X: np.ndarray,
):
    Lk = L(k, argnum=0)

    np.testing.assert_allclose(
        Lk(X[:, None], X[None, :]),
        Lk_jax(X[:, None], X[None, :]),
    )


def test_kLa(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
    kLa_jax: linpde_gp.randprocs.kernels.JaxKernel,
    X: np.ndarray,
):
    kLa = L(k, argnum=1)

    np.testing.assert_allclose(
        kLa(X[:, None], X[None, :]),
        kLa_jax(X[:, None], X[None, :]),
    )


def test_LkLa(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
    LkLa_jax: linpde_gp.randprocs.kernels.JaxKernel,
    X: np.ndarray,
):
    LkLa = L(L(k, argnum=1), argnum=0)

    np.testing.assert_allclose(
        LkLa(X[:, None], X[None, :]),
        LkLa_jax(X[:, None], X[None, :]),
    )
