import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np

from ._jax import JaxKernel


class MultiIndependentProductMatern(JaxKernel):
    def __init__(
        self, input_dim, output_dim, lengthscales, p=3
    ):
        if type(input_dim) != int or input_dim < 1:
            raise ValueError("input_dim must be a positive integer.")
        if type(output_dim) != int or output_dim < 1:
            raise ValueError("output_dim must be a positive integer.")

        if lengthscales.shape != (output_dim, input_dim):
            raise ValueError("Lengthscales must have shape (O, I) where O is the number of outputs and I is the number of inputs.")
        if p != 3:
            raise ValueError("This kernel currently only supports p=3.")
            
        self._p = p
        self._lengthscale = lengthscales
        self._scale_factors = np.sqrt(2 * self.p + 1) / self._lengthscale

        super().__init__(input_shape=(input_dim,), output_shape=(output_dim, output_dim))

    @property
    def nu(self) -> float:
        return self._p + 0.5

    @property
    def p(self) -> int:
        return self._p

    @property
    def lengthscale(self) -> np.ndarray:
        return self._lengthscale
    
    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        result = np.zeros(broadcast_batch_shape + self._output_shape)
        n_outputs = self._output_shape[0]

        diags = self.diagonal_covs(x0, x1)
        result[..., np.arange(n_outputs), np.arange(n_outputs)] = diags

        return result

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        result = jnp.zeros(broadcast_batch_shape + self._output_shape)
        n_outputs = self._output_shape[0]

        diags = self.diagonal_covs(x0, x1)
        result = result.at[..., jnp.arange(n_outputs), jnp.arange(n_outputs)].set(diags)

        return result

    def _evaluate_factors(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            scaled_distances = np.zeros_like(x0)
        else:
            scaled_distances = self._scale_factors * np.abs(x0 - x1)[..., None, :]

        if self._p == 3:
            return (
                1.0
                + scaled_distances
                * (
                    1.0
                    + scaled_distances * (2.0 / 5.0 + scaled_distances * (1.0 / 15.0))
                )
            ) * np.exp(-scaled_distances)

        raise ValueError()

    def diagonal_covs(self, x0: np.ndarray, x1: Optional[np.ndarray]):
        return np.prod(self._evaluate_factors(x0, x1), axis=-1)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_factors_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            scaled_distances = jnp.zeros_like(x0)
        else:
            scaled_distances = self._scale_factors * jnp.abs(x0 - x1)[..., None, :]

        if self._p == 3:
            return (
                1.0
                + scaled_distances
                * (
                    1.0
                    + scaled_distances * (2.0 / 5.0 + scaled_distances * (1.0 / 15.0))
                )
            ) * jnp.exp(-scaled_distances)

        raise ValueError()

    def diagonal_covs_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]):
        return jnp.prod(self._evaluate_factors_jax(x0, x1), axis=-1)