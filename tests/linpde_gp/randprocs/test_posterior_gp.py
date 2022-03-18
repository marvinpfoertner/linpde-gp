from typing import Optional

import jax
import numpy as np
import probnum as pn
from probnum.typing import ShapeType
import scipy.linalg

import pytest

import linpde_gp

jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[1], scope="module")
def input_dim(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def input_shape(input_dim: int) -> ShapeType:
    return (input_dim,)


@pytest.fixture
def prior(input_shape: ShapeType) -> pn.randprocs.GaussianProcess:
    return pn.randprocs.GaussianProcess(
        mean=linpde_gp.randprocs.mean_fns.Zero(input_shape=input_shape),
        cov=linpde_gp.randprocs.kernels.ExpQuad(
            input_shape=input_shape,
            lengthscales=0.25,
            output_scale=2.0,
        ),
    )


@pytest.fixture
def batch_sizes() -> tuple[int]:
    return (2, 3, 2, 4)


@pytest.fixture
def num_points(batch_sizes: tuple[int]) -> int:
    return sum(batch_sizes)


@pytest.fixture
def Xs(num_points: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, num_points)[:, None]


@pytest.fixture
def Xs_test() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 50)[:, None]


@pytest.fixture
def Ys(Xs: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(np.pi * Xs[..., 0])


@pytest.fixture
def Xs_batched(Xs: np.ndarray, batch_sizes: tuple[int]) -> list[np.ndarray]:
    return np.array_split(
        Xs,
        np.cumsum(batch_sizes)[:-1],
        axis=0,
    )


@pytest.fixture
def Ys_batched(Ys: np.ndarray, batch_sizes: tuple[int]) -> list[np.ndarray]:
    return np.array_split(
        Ys,
        np.cumsum(batch_sizes)[:-1],
        axis=0,
    )


@pytest.fixture
def Y_errs_batched(batch_sizes: tuple[int]) -> tuple[pn.randvars.Normal]:
    return (
        pn.randvars.Normal(np.ones(batch_sizes[0]), 0.6 ** 2 * np.eye(batch_sizes[0])),
        None,
        None,
        pn.randvars.Normal(np.zeros(batch_sizes[3]), 0.3 ** 2 * np.eye(batch_sizes[3])),
    )


@pytest.fixture
def Ys_err(
    Y_errs_batched: tuple[pn.randvars.Normal], batch_sizes: tuple[int]
) -> pn.randvars.Normal:
    return pn.randvars.Normal(
        mean=np.concatenate(
            [
                batch_noise.mean if batch_noise is not None else np.zeros(batch_size)
                for batch_noise, batch_size in zip(Y_errs_batched, batch_sizes)
            ],
            axis=0,
        ),
        cov=scipy.linalg.block_diag(
            *(
                batch_noise.cov
                if batch_noise is not None
                else np.zeros((batch_size, batch_size))
                for batch_noise, batch_size in zip(Y_errs_batched, batch_sizes)
            )
        ),
    )


@pytest.fixture
def L(input_shape: ShapeType) -> linpde_gp.linfuncops.LinearFunctionOperator:
    return linpde_gp.problems.pde.diffops.ScaledLaplaceOperator(
        domain_shape=input_shape
    )


@pytest.fixture
def naive_posterior_gp(
    prior: pn.randprocs.GaussianProcess,
    Xs: np.ndarray,
    Ys: np.ndarray,
    Ys_err: pn.randvars.Normal,
) -> pn.randprocs.GaussianProcess:
    return condition_gp_on_observations(prior, Xs, Ys, noise=Ys_err)


@pytest.fixture
def naive_posterior_gp_linop(
    naive_posterior_gp: pn.randprocs.GaussianProcess,
    L: linpde_gp.linfuncops.LinearFunctionOperator,
) -> tuple[pn.randprocs.GaussianProcess, pn.randprocs.kernels.Kernel]:
    return apply_jax_linop_to_gp(naive_posterior_gp, L)


@pytest.fixture
def posterior_gp(
    prior: pn.randprocs.GaussianProcess,
    Xs_batched: tuple[np.ndarray],
    Ys_batched: tuple[np.ndarray],
    Y_errs_batched: tuple[pn.randvars.Normal],
) -> linpde_gp.randprocs.ConditionalGaussianProcess:
    posterior_gp = prior

    for X, Y, Y_err in zip(Xs_batched, Ys_batched, Y_errs_batched):
        posterior_gp = posterior_gp.condition_on_observations(X, Y, b=Y_err)

    return posterior_gp


def test_posterior_gp(
    posterior_gp: linpde_gp.randprocs.ConditionalGaussianProcess,
    naive_posterior_gp: pn.randprocs.GaussianProcess,
    Xs_test: np.ndarray,
):
    iter_X_test = posterior_gp(Xs_test)
    naive_X_test = naive_posterior_gp(Xs_test)

    np.testing.assert_allclose(iter_X_test.mean, naive_X_test.mean)
    np.testing.assert_allclose(iter_X_test.var, naive_X_test.var)
    np.testing.assert_allclose(iter_X_test.cov, naive_X_test.cov)


def test_posterior_gp_linop(
    posterior_gp: linpde_gp.randprocs.ConditionalGaussianProcess,
    L: linpde_gp.linfuncops.LinearFunctionOperator,
    naive_posterior_gp_linop: pn.randprocs.GaussianProcess,
    Xs_test: np.ndarray,
):
    posterior_gp_linop = L(posterior_gp)

    iter_X_test = posterior_gp_linop(Xs_test)
    naive_X_test = naive_posterior_gp_linop(Xs_test)

    np.testing.assert_allclose(iter_X_test.mean, naive_X_test.mean)
    np.testing.assert_allclose(iter_X_test.var, naive_X_test.var)
    np.testing.assert_allclose(iter_X_test.cov, naive_X_test.cov)


# Naive GP conditioning and transformation
def condition_gp_on_observations(
    gp: pn.randprocs.GaussianProcess,
    X: np.ndarray,
    fX: np.ndarray,
    noise: Optional[pn.randvars.Normal] = None,
) -> pn.randprocs.GaussianProcess:
    pred_mean = gp.mean(X)
    gram = gp.cov(X[:, None], X[None, :])

    if noise is not None:
        pred_mean += noise.mean
        gram += noise.cov

    gram_cho = scipy.linalg.cho_factor(gram)

    representer_weights = scipy.linalg.cho_solve(gram_cho, (fX - pred_mean))

    @jax.jit
    def cond_mean(x):
        mx = gp.mean.jax(x)
        kxX = gp.cov.jax(x, X)
        return mx + kxX @ representer_weights

    @jax.jit
    def cond_cov(x0, x1):
        kxx = gp.cov.jax(x0, x1)
        kxX = gp.cov.jax(x0, X)
        kXx = gp.cov.jax(X, x1)
        return kxx - kxX @ jax.scipy.linalg.cho_solve(gram_cho, kXx)

    cond_gp = pn.randprocs.GaussianProcess(
        mean=linpde_gp.function.JaxLambdaFunction(
            cond_mean, input_shape=gp.input_shape, vectorize=True
        ),
        cov=linpde_gp.randprocs.kernels.JaxLambdaKernel(
            cond_cov, input_shape=gp.input_shape, vectorize=True
        ),
    )

    return cond_gp


def apply_jax_linop_to_gp(
    gp: pn.randprocs.GaussianProcess,
    L: linpde_gp.linfuncops.JaxLinearOperator,
    **linop_kwargs,
) -> pn.randprocs.GaussianProcess:
    mean = L(gp.mean.jax, argnum=0, **linop_kwargs)
    crosscov = L(gp.cov.jax, argnum=1, **linop_kwargs)
    cov = L(crosscov, argnum=0, **linop_kwargs)

    gp_linop = pn.randprocs.GaussianProcess(
        mean=linpde_gp.function.JaxLambdaFunction(
            mean, input_shape=gp.input_shape, vectorize=True
        ),
        cov=linpde_gp.randprocs.kernels.JaxLambdaKernel(
            cov, input_shape=gp.input_shape, vectorize=True
        ),
    )

    return gp_linop
