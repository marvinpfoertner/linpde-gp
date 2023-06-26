from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Optional

from jax import numpy as jnp
import numpy as np
from probnum.randprocs.covfuncs import CovarianceFunction
from probnum.typing import ArrayLike

CovarianceFunction._batched_sum = (  # pylint: disable=protected-access
    lambda self, a, **sum_kwargs: np.sum(
        a, axis=tuple(range(-self.input_ndim, 0)), **sum_kwargs
    )
)

CovarianceFunction._batched_euclidean_norm_sq = (  # pylint: disable=protected-access
    lambda self, a, **sum_kwargs: self._batched_sum(  # pylint: disable=protected-access
        a**2, **sum_kwargs
    )
)

CovarianceFunction._batched_euclidean_norm = (  # pylint: disable=protected-access
    lambda self, a, **sum_kwargs: np.sqrt(
        self._batched_euclidean_norm_sq(  # pylint: disable=protected-access
            a, **sum_kwargs
        )
    )
)


class JaxCovarianceFunctionMixin(abc.ABC):
    """Careful: Must come before `CovarianceFunction` in inheritance"""

    def jax(
        self: CovarianceFunction, x0: ArrayLike, x1: Optional[ArrayLike]
    ) -> jnp.ndarray:
        x0 = jnp.asarray(x0)

        if x1 is not None:
            x1 = jnp.asarray(x1)

        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        k_x0_x1 = self._evaluate_jax(x0, x1)

        assert (
            k_x0_x1.shape
            == broadcast_batch_shape + self.output_shape_0 + self.output_shape_1
        )

        return k_x0_x1

    @abc.abstractmethod
    def _evaluate_jax(
        self: CovarianceFunction, x0: jnp.ndarray, x1: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        pass

    def _batched_sum_jax(
        self: CovarianceFunction, a: jnp.ndarray, **sum_kwargs
    ) -> jnp.ndarray:
        return jnp.sum(a, axis=tuple(range(-self.input_ndim, 0)), **sum_kwargs)

    def _batched_euclidean_norm_sq_jax(
        self, a: jnp.ndarray, **sum_kwargs
    ) -> jnp.ndarray:
        return self._batched_sum_jax(a**2, **sum_kwargs)

    def _batched_euclidean_norm_jax(
        self: CovarianceFunction, a: jnp.ndarray, **sum_kwargs
    ) -> jnp.ndarray:
        return jnp.sqrt(self._batched_sum_jax(a**2, **sum_kwargs))

    def __add__(
        self: CovarianceFunction, other: CovarianceFunction
    ) -> JaxCovarianceFunction:
        from ._jax_arithmetic import (  # pylint: disable=import-outside-toplevel
            JaxSumCovarianceFunction,
        )

        return JaxSumCovarianceFunction(self, other)

    def __rmul__(self: CovarianceFunction, other: ArrayLike) -> JaxCovarianceFunction:
        if np.ndim(other) == 0:
            from ._jax_arithmetic import (  # pylint: disable=import-outside-toplevel
                JaxScaledCovarianceFunction,
            )

            return JaxScaledCovarianceFunction(self, scalar=other)

        return super().__rmul__(self, other)


class JaxCovarianceFunction(JaxCovarianceFunctionMixin, CovarianceFunction):
    ...


class JaxIsotropicMixin:  # pylint: disable=too-few-public-methods
    def _squared_euclidean_distances_jax(
        self: JaxCovarianceFunctionMixin,
        x0: jnp.ndarray,
        x1: Optional[jnp.ndarray],
        *,
        scale_factors: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Implementation of the squared (modified) Euclidean distance, which supports
        scalar inputs, an optional second argument, and different scale factors along
        all input dimensions."""

        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        if scale_factors is not None:
            diffs *= scale_factors

        return jnp.sum(diffs**2, axis=tuple(range(-self.input_ndim, 0)))

    def _euclidean_distances_jax(
        self: JaxCovarianceFunctionMixin,
        x0: np.ndarray,
        x1: Optional[np.ndarray],
        *,
        scale_factors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Implementation of the (modified) Euclidean distance, which supports scalar
        inputs, an optional second argument, and different scale factors along all input
        dimensions."""

        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return jnp.sqrt(
            self._squared_euclidean_distances_jax(x0, x1, scale_factors=scale_factors)
        )


class JaxLambdaCovarianceFunction(JaxCovarianceFunction):
    def __init__(
        self,
        k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        vectorize: bool = True,
        **covfunc_kwargs,
    ):
        super().__init__(**covfunc_kwargs)

        if vectorize:
            k = jnp.vectorize(
                k, signature="(),()->()" if self.input_shape == () else "(d),(d)->()"
            )

        self._k = k

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        if x1 is None:
            x1 = x0

        return np.array(self._k(x0, x1))

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            x1 = x0

        return self._k(x0, x1)
