from typing import Optional

from jax import numpy as jnp
import numpy as np
from probnum.randprocs.kernels import Kernel

from ._jax import JaxKernel


class StationaryMixin:
    def _squared_euclidean_distances(
        self: Kernel,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
        lengthscales: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Implementation of the squared Euclidean distance, which supports scalar
        inputs and an optional second argument."""
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim] + self.output_shape,
            )

        diffs = x0 - x1

        if lengthscales is not None:
            diffs /= lengthscales

        return self._batched_euclidean_norm_sq(diffs)


class JaxStationaryMixin(StationaryMixin):
    def _squared_euclidean_distances_jax(
        self: JaxKernel,
        x0: jnp.ndarray,
        x1: Optional[jnp.ndarray],
        lengthscales: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Jax implementation of the squared Euclidean distance, which supports scalar
        inputs and an optional second argument."""
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim] + self.output_shape,
            )

        diffs = x0 - x1

        if lengthscales is not None:
            diffs /= lengthscales

        return self._batched_euclidean_norm_sq_jax(diffs)
