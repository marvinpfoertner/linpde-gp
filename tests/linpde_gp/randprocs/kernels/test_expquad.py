from typing import Union

import jax
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeType
import scipy.stats

import pytest
import pytest_cases

import linpde_gp

jax.config.update("jax_enable_x64", True)


@pytest_cases.fixture(scope="module")
@pytest_cases.parametrize(
    "input_shape",
    ((), (1,), (2,), (3,)),
    ids=lambda input_shape: f"inshape={input_shape}",
)
def input_shape(input_shape: ShapeType) -> ShapeType:
    return input_shape


@pytest_cases.parametrize(lengthscale=(0.4, 1.0, 3.0))
def case_lengthscales_scalar(lengthscale) -> float:
    return lengthscale


def case_lengthscales_diagonal(input_shape: ShapeType) -> np.ndarray:
    seed = abs(hash(input_shape) + 34876)

    return np.random.default_rng(seed).uniform(0.4, 3.0, size=input_shape)


@pytest.mark.skip("slow")
def case_lengthscales_full(input_shape: ShapeType) -> np.ndarray:
    if input_shape == ():
        pytest.skip("`LinearOperator`s don't support scalar shapes")

    rng = np.random.default_rng(seed=abs(hash(input_shape) + 8934879))

    lengthscales = pn.linops.Scaling(rng.uniform(0.4, 3.0, size=input_shape))

    if input_shape == (1,):
        return lengthscales

    return (
        pn.linops.Matrix(
            scipy.stats.special_ortho_group.rvs(input_shape[0], random_state=rng)
        )
        @ lengthscales
    )


@pytest_cases.fixture(scope="module")
@pytest_cases.parametrize(output_scale=(0.1, 1.0, 2.8))
def output_scale(output_scale: float) -> float:
    return output_scale


@pytest_cases.fixture(scope="module")
@pytest_cases.parametrize_with_cases(
    "lengthscales",
    cases=pytest_cases.THIS_MODULE,
    glob="lengthscales_*",
    scope="module",
)
def k(
    input_shape: ShapeType,
    lengthscales: Union[ArrayLike, pn.linops.LinearOperatorLike],
    output_scale: float,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return linpde_gp.randprocs.kernels.ExpQuad(
        input_shape=input_shape,
        lengthscales=lengthscales,
        output_scale=output_scale,
    )


def case_linfuncop_scaled_laplace(
    input_shape: ShapeType,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    return linpde_gp.problems.pde.diffops.ScaledLaplaceOperator(
        domain_shape=input_shape, alpha=-1.0
    )


def case_linfuncop_directional_derivative(
    input_shape: ShapeType,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    return linpde_gp.problems.pde.diffops.TimeDerivative(
        domain_shape=input_shape,
    )


@pytest_cases.fixture(scope="module")
@pytest_cases.parametrize_with_cases(
    "linfuncop",
    cases=pytest_cases.THIS_MODULE,
    glob="linfuncop_*",
    scope="module",
)
def L(
    linfuncop: linpde_gp.linfuncops.LinearFunctionOperator,
) -> linpde_gp.linfuncops.JaxLinearOperator:
    return linfuncop


@pytest_cases.fixture(scope="module")
def k_jax(
    k: linpde_gp.randprocs.kernels.JaxKernel,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return linpde_gp.randprocs.kernels.JaxLambdaKernel(
        k=k.jax,
        input_shape=k.input_shape,
        vectorize=False,
    )


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


@pytest_cases.fixture(scope="module")
def X(input_shape: ShapeType) -> np.ndarray:
    input_dim = 1 if input_shape == () else input_shape[0]

    return np.stack(
        np.meshgrid(
            *(np.linspace(-2.0, 2.0, 30 // input_dim) for _ in range(input_dim)),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1,) + input_shape)


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
