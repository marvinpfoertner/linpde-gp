import jax
import numpy as np
import pytest

import linpde_gp
from probnum.typing import ShapeType

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", params=(1, 2, 3), ids=("ndim-1", "ndim-2", "ndim-3"))
def input_dim(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def input_shape(input_dim: int) -> ShapeType:
    return (input_dim,)


@pytest.fixture(scope="module", params=(0.4, 1.0, 3.0), ids=("l-0.4", "l-1.0", "l-3.0"))
def lengthscale(request) -> float:
    return request.param


@pytest.fixture(scope="module", params=(0.1, 1.0, 2.8), ids=("o-0.1", "o-1.0", "o-2.8"))
def output_scale(request) -> float:
    return request.param


@pytest.fixture(scope="module")
def k(
    input_shape: ShapeType,
    lengthscale: float,
    output_scale: float,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return linpde_gp.randprocs.kernels.ExpQuad(
        input_shape=input_shape,
        lengthscales=lengthscale,
        output_scale=output_scale,
    )


@pytest.fixture(scope="module")
def k_jax(
    k: linpde_gp.randprocs.kernels.JaxKernel,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return linpde_gp.randprocs.kernels.JaxLambdaKernel(
        k=k.jax,
        input_shape=k.input_shape,
        vectorize=False,
    )


@pytest.fixture(scope="module")
def L(input_shape: ShapeType) -> linpde_gp.linfuncops.JaxLinearOperator:
    return linpde_gp.problems.pde.diffops.ScaledLaplaceOperator(
        domain_shape=input_shape, alpha=-1.0
    )


@pytest.fixture(scope="module")
def Lk_jax(
    k_jax: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return L(k_jax, argnum=0)


@pytest.fixture(scope="module")
def kLa_jax(
    k_jax: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return L(k_jax, argnum=1)


@pytest.fixture(scope="module")
def LkLa_jax(
    kLa_jax: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return L(kLa_jax, argnum=0)


@pytest.fixture(scope="module")
def X(input_dim: int) -> np.ndarray:
    return np.stack(
        np.meshgrid(
            *(np.linspace(-2.0, 2.0, 30 // input_dim) for _ in range(input_dim)),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, input_dim))


def test_Lk(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
    Lk_jax: linpde_gp.randprocs.kernels.JaxKernel,
    X: np.ndarray,
):
    Lk = L(k, argnum=0)

    np.testing.assert_allclose(
        Lk(X[:, None, :], X[None, :, :]),
        Lk_jax(X[:, None, :], X[None, :, :]),
    )


def test_kLa(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
    kLa_jax: linpde_gp.randprocs.kernels.JaxKernel,
    X: np.ndarray,
):
    kLa = L(k, argnum=1)

    np.testing.assert_allclose(
        kLa(X[:, None, :], X[None, :, :]),
        kLa_jax(X[:, None, :], X[None, :, :]),
    )


def test_LkLa(
    k: linpde_gp.randprocs.kernels.JaxKernel,
    L: linpde_gp.linfuncops.JaxLinearOperator,
    LkLa_jax: linpde_gp.randprocs.kernels.JaxKernel,
    X: np.ndarray,
):
    LkLa = L(L(k, argnum=1), argnum=0)

    np.testing.assert_allclose(
        LkLa(X[:, None, :], X[None, :, :]),
        LkLa_jax(X[:, None, :], X[None, :, :]),
    )
