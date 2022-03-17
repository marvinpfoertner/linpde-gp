from typing import Optional

import numpy as np


class StationaryMixin:
    def _squared_euclidean_distances(
        self,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
        lengthscales: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Implementation of the squared Euclidean distance, which supports scalar
        inputs and an optional second argument."""
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        dists_sq = x0 - x1

        if lengthscales is not None:
            dists_sq /= lengthscales

        dists_sq = dists_sq ** 2

        if self.input_ndim > 0:
            assert self.input_ndim == 1

            dists_sq = np.sum(dists_sq, axis=-1)

        return dists_sq
