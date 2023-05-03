import abc
import functools

import jax
import jax.numpy as jnp
import numpy as np

from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction


class DowndateCovarianceFunction(JaxCovarianceFunction):
    """
    Covariance function that is obtained by downdating a prior covariance function.
    """

    def __init__(self, prior_cov: JaxCovarianceFunction):
        self._prior_cov = prior_cov
        super().__init__(
            input_shape_0=prior_cov.input_shape_0,
            input_shape_1=prior_cov.input_shape_1,
            output_shape_0=prior_cov.output_shape_0,
            output_shape_1=prior_cov.output_shape_1,
        )

    @abc.abstractmethod
    def _downdate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def _downdate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        raise NotImplementedError

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        k_xx = self._prior_cov(x0, x1)
        return k_xx - self._downdate(x0, x1)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        k_xx = self._prior_cov.jax(x0, x1)
        return k_xx - self._downdate_jax(x0, x1)
