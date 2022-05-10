import functools
from multiprocessing.sharedctypes import Value
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from probnum.typing import FloatLike, ShapeLike

from ._jax import JaxKernel
from ._stationary import JaxStationaryMixin


class Matern(JaxKernel, JaxStationaryMixin):
    def __init__(
        self,
        input_shape: ShapeLike,
        p: int = 3,
        lengthscale: FloatLike = 1.0,
    ):
        super().__init__(input_shape, output_shape=())

        self._p = int(p)
        self._lengthscale = float(lengthscale)

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
        distances = np.sqrt(
            self._squared_euclidean_distances(x0, x1, self._lengthscale)
        )

        if self.p == 3:
            scaled_distances = np.sqrt(2 * self.p + 1) / self.lengthscale * distances

            return (
                1.0
                + scaled_distances
                * (
                    1.0
                    + scaled_distances * (2.0 / 5.0 + scaled_distances * (1.0 / 15.0))
                )
            ) * np.exp(-scaled_distances)

        raise ValueError()

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        distances = jnp.sqrt(
            self._squared_euclidean_distances_jax(x0, x1, self._lengthscale)
        )

        if self.p == 3:
            scaled_distances = np.sqrt(2 * self.p + 1) / self.lengthscale * distances

            return (
                1.0
                + scaled_distances
                * (
                    1.0
                    + scaled_distances * (2.0 / 5.0 + scaled_distances * (1.0 / 15.0))
                )
            ) * jnp.exp(-scaled_distances)

        raise ValueError()
