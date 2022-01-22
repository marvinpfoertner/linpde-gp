import jax
import linpde_gp
import numpy as np
import probnum as pn
import pytest
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def prior() -> pn.randprocs.GaussianProcess:
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
            prior_cov, input_dim=1, vectorize=False
        ),
    )


@pytest.fixture
def group_sizes() -> tuple[int]:
    return (2, 3, 2, 4)


@pytest.fixture
def num_points(group_sizes: tuple[int]) -> int:
    return sum(group_sizes)


@pytest.fixture
def Xs(num_points: int, group_sizes: tuple[int]) -> list[np.ndarray]:
    return np.array_split(
        np.linspace(-1.0, 1.0, num_points)[:, None],
        np.cumsum(group_sizes)[:-1],
        axis=-2,
    )


@pytest.fixture
def Ys(Xs: list[np.ndarray]) -> list[np.ndarray]:
    return [2.0 * np.sin(np.pi * X[..., 0]) for X in Xs]


@pytest.fixture
def noise_models(group_sizes: tuple[int]) -> tuple[pn.randvars.Normal]:
    return (
        pn.randvars.Normal(np.ones(group_sizes[0]), 0.6 ** 2 * np.eye(group_sizes[0])),
        None,
        None,
        pn.randvars.Normal(np.zeros(group_sizes[3]), 0.3 ** 2 * np.eye(group_sizes[3])),
    )


@pytest.fixture
def posterior_gp(
    prior: pn.randprocs.GaussianProcess,
    Xs: tuple[np.ndarray],
    Ys: tuple[np.ndarray],
    noise_models: tuple[pn.randvars.Normal],
) -> linpde_gp.randprocs.PosteriorGaussianProcess:
    posterior_gp = prior

    for X, Y, noise_model in zip(Xs, Ys, noise_models):
        posterior_gp = posterior_gp.condition_on_observations(X, Y, noise_model)

    return posterior_gp


def test_posterior_gp(posterior_gp):
    pass
