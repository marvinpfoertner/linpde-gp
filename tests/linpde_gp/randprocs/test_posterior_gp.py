from typing import Optional

import jax
import linpde_gp
import numpy as np
import probnum as pn
import pytest
import scipy.linalg
from jax import numpy as jnp
from linpde_gp.typing import JaxLinearOperator

jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[1])
def input_dim(request) -> int:
    return request.param


@pytest.fixture
def prior(input_dim: int) -> pn.randprocs.GaussianProcess:
    lengthscale = 0.25
    output_scale = 2.0

    @jax.jit
    def prior_mean(x):
        return jnp.full_like(x[..., 0], 0.0)

    @jax.jit
    def prior_cov(x0, x1):
        sqnorms = jnp.sum((x0 - x1) ** 2, axis=-1)

        return output_scale ** 2 * jnp.exp(-(1.0 / (2.0 * lengthscale ** 2)) * sqnorms)

    return pn.randprocs.GaussianProcess(
        mean=linpde_gp.randprocs.mean_fns.JaxMean(prior_mean, vectorize=False),
        cov=linpde_gp.randprocs.kernels.JaxKernel(
            prior_cov,
            input_dim=input_dim,
            vectorize=True,
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
def X_test() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 50)[:, None]


@pytest.fixture
def Ys(Xs: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(np.pi * Xs[..., 0])


@pytest.fixture
def X_batches(Xs: np.ndarray, batch_sizes: tuple[int]) -> list[np.ndarray]:
    return np.array_split(
        Xs,
        np.cumsum(batch_sizes)[:-1],
        axis=0,
    )


@pytest.fixture
def Y_batches(Ys: np.ndarray, batch_sizes: tuple[int]) -> list[np.ndarray]:
    return np.array_split(
        Ys,
        np.cumsum(batch_sizes)[:-1],
        axis=0,
    )


@pytest.fixture
def batch_noise_models(batch_sizes: tuple[int]) -> tuple[pn.randvars.Normal]:
    return (
        pn.randvars.Normal(np.ones(batch_sizes[0]), 0.6 ** 2 * np.eye(batch_sizes[0])),
        None,
        None,
        pn.randvars.Normal(np.zeros(batch_sizes[3]), 0.3 ** 2 * np.eye(batch_sizes[3])),
    )


@pytest.fixture
def noise_model(
    batch_noise_models: tuple[pn.randvars.Normal], batch_sizes: tuple[int]
) -> pn.randvars.Normal:
    return pn.randvars.Normal(
        mean=np.concatenate(
            [
                batch_noise.mean if batch_noise is not None else np.zeros(batch_size)
                for batch_noise, batch_size in zip(batch_noise_models, batch_sizes)
            ],
            axis=0,
        ),
        cov=scipy.linalg.block_diag(
            *(
                batch_noise.cov
                if batch_noise is not None
                else np.zeros((batch_size, batch_size))
                for batch_noise, batch_size in zip(batch_noise_models, batch_sizes)
            )
        ),
    )


@pytest.fixture
def linop() -> linpde_gp.linfuncops.LinearFunctionOperator:
    return linpde_gp.problems.pde.diffops.LaplaceOperator()


@pytest.fixture
def naive_posterior_gp(
    prior: pn.randprocs.GaussianProcess,
    Xs: np.ndarray,
    Ys: np.ndarray,
    noise_model: pn.randvars.Normal,
) -> pn.randprocs.GaussianProcess:
    return condition_gp_on_observations(prior, Xs, Ys, noise=noise_model)


@pytest.fixture
def naive_posterior_gp_linop(
    naive_posterior_gp: pn.randprocs.GaussianProcess,
    linop: linpde_gp.linfuncops.LinearFunctionOperator,
) -> tuple[pn.randprocs.GaussianProcess, pn.randprocs.kernels.Kernel]:
    return apply_jax_linop_to_gp(naive_posterior_gp, linop)


@pytest.fixture
def posterior_gp(
    prior: pn.randprocs.GaussianProcess,
    X_batches: tuple[np.ndarray],
    Y_batches: tuple[np.ndarray],
    batch_noise_models: tuple[pn.randvars.Normal],
) -> linpde_gp.randprocs.PosteriorGaussianProcess:
    posterior_gp = prior

    for X, Y, noise in zip(X_batches, Y_batches, batch_noise_models):
        posterior_gp = posterior_gp.condition_on_observations(X, Y, noise)

    return posterior_gp


def test_posterior_gp(
    posterior_gp: linpde_gp.randprocs.PosteriorGaussianProcess,
    naive_posterior_gp: pn.randprocs.GaussianProcess,
    X_test: np.ndarray,
):
    iter_X_test = posterior_gp(X_test)
    naive_X_test = naive_posterior_gp(X_test)

    np.testing.assert_allclose(iter_X_test.mean, naive_X_test.mean)
    np.testing.assert_allclose(iter_X_test.var, naive_X_test.var)
    np.testing.assert_allclose(iter_X_test.cov, naive_X_test.cov)


def test_posterior_gp_linop(
    posterior_gp: linpde_gp.randprocs.PosteriorGaussianProcess,
    linop: linpde_gp.linfuncops.LinearFunctionOperator,
    naive_posterior_gp_linop: pn.randprocs.GaussianProcess,
    X_test: np.ndarray,
):
    posterior_gp_linop = linop(posterior_gp)

    iter_X_test = posterior_gp_linop(X_test)
    naive_X_test = naive_posterior_gp_linop(X_test)

    np.testing.assert_allclose(iter_X_test.mean, naive_X_test.mean)


# Naive GP conditioning and transformation
def condition_gp_on_observations(
    gp: pn.randprocs.GaussianProcess,
    X: np.ndarray,
    fX: np.ndarray,
    noise: Optional[pn.randvars.Normal] = None,
) -> pn.randprocs.GaussianProcess:
    pred_mean = gp._meanfun(X)
    gram = gp._covfun(X[:, None, :], X[None, :, :])

    if noise is not None:
        pred_mean += noise.mean
        gram += noise.cov

    gram_cho = scipy.linalg.cho_factor(gram)

    representer_weights = scipy.linalg.cho_solve(gram_cho, (fX - pred_mean))

    @jax.jit
    def cond_mean(x):
        mx = gp._meanfun.jax(x)
        kxX = gp._covfun.jax(x, X)
        return mx + kxX @ representer_weights

    @jax.jit
    def cond_cov(x0, x1):
        kxx = gp._covfun.jax(x0, x1)
        kxX = gp._covfun.jax(x0, X)
        kXx = gp._covfun.jax(X, x1)
        return kxx - kxX @ jax.scipy.linalg.cho_solve(gram_cho, kXx)

    cond_gp = pn.randprocs.GaussianProcess(
        mean=linpde_gp.randprocs.mean_fns.JaxMean(cond_mean, vectorize=True),
        cov=linpde_gp.randprocs.kernels.JaxKernel(
            cond_cov, input_dim=gp.input_dim, vectorize=True
        ),
    )

    return cond_gp


def apply_jax_linop_to_gp(
    gp: pn.randprocs.GaussianProcess,
    linop: JaxLinearOperator,
    **linop_kwargs,
) -> pn.randprocs.GaussianProcess:
    mean = linop(gp._meanfun.jax, argnum=0, **linop_kwargs)
    crosscov = linop(gp._covfun.jax, argnum=1, **linop_kwargs)
    cov = linop(crosscov, argnum=0, **linop_kwargs)

    gp_linop = pn.randprocs.GaussianProcess(
        mean=linpde_gp.randprocs.mean_fns.JaxMean(mean, vectorize=True),
        cov=linpde_gp.randprocs.kernels.JaxKernel(
            cov, input_dim=gp.input_dim, vectorize=True
        ),
    )

    return gp_linop
