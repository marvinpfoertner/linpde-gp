import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike, FloatLike, ShapeLike

from ._jax import JaxKernel, JaxStationaryMixin


class ExpQuad(JaxKernel, JaxStationaryMixin):
    def __init__(
        self,
        input_shape: ShapeLike,
        lengthscales: ArrayLike = 1.0,
        output_scale: FloatLike = 1.0,
    ):
        super().__init__(input_shape, output_shape=())

        lengthscales = np.asarray(lengthscales, dtype=np.double)

        if not (lengthscales.shape == () or lengthscales.shape == self.input_shape):
            raise ValueError()

        self._lengthscales = lengthscales

        output_scale = float(output_scale)

        if output_scale < 0:
            raise ValueError()

        self._output_scale = np.asarray(output_scale, dtype=np.double)
        self._output_scale_sq = self._output_scale ** 2

    @property
    def lengthscales(self) -> np.ndarray:
        return self._lengthscales

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                self._output_scale_sq,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return self._output_scale_sq * np.exp(
            -0.5 * self._squared_euclidean_distances(x0, x1, self._lengthscales)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                self._output_scale ** 2,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return self._output_scale_sq * jnp.exp(
            -0.5 * self._squared_euclidean_distances_jax(x0, x1, self._lengthscales)
        )
