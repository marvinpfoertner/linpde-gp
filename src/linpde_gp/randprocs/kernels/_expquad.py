import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike, ShapeLike

from ._jax import JaxKernel
from ._stationary import JaxStationaryMixin


class ExpQuad(JaxKernel, JaxStationaryMixin):
    def __init__(
        self,
        input_shape: ShapeLike,
        lengthscales: ArrayLike = 1.0,
    ):
        super().__init__(input_shape, output_shape=())

        lengthscales = np.asarray(lengthscales, dtype=np.double)

        if lengthscales.shape not in ((), self.input_shape):
            raise ValueError()

        self._lengthscales = lengthscales

    @property
    def lengthscales(self) -> np.ndarray:
        return self._lengthscales

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.ones_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return np.exp(
            -0.5 * self._squared_euclidean_distances(x0, x1, self._lengthscales)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.ones_like(
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return jnp.exp(
            -0.5 * self._squared_euclidean_distances_jax(x0, x1, self._lengthscales)
        )
