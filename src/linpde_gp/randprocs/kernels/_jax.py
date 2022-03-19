import abc
from collections.abc import Callable
import functools
from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from ... import linfuncops
from ._stationary import StationaryMixin


class JaxKernel(pn.randprocs.kernels.Kernel):
    def jax(self, x0: ArrayLike, x1: Optional[ArrayLike]) -> jnp.ndarray:
        x0 = jnp.asarray(x0)

        if x1 is not None:
            x1 = jnp.asarray(x1)

        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        k_x0_x1 = self._evaluate_jax(x0, x1)

        assert k_x0_x1.shape == broadcast_batch_shape + self.output_shape

        return k_x0_x1

    @abc.abstractmethod
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        pass

    def __add__(
        self, other: pn.randprocs.kernels.Kernel
    ) -> pn.randprocs.kernels.Kernel:
        from ._jax_arithmetic import JaxSumKernel

        return JaxSumKernel(self, other)


@linfuncops.JaxLinearOperator.__call__.register
def _(self, k: JaxKernel, /, *, argnum=0):
    try:
        return super(linfuncops.JaxLinearOperator, self).__call__(k, argnum=argnum)
    except NotImplementedError:
        return JaxLambdaKernel(
            self._jax_fallback(k.jax, argnum=argnum),
            input_shape=self.output_domain_shape,
            vectorize=True,
        )


class JaxLambdaKernel(JaxKernel):
    def __init__(
        self,
        k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        input_shape: ShapeLike,
        output_shape: ShapeLike = (),
        vectorize: bool = True,
    ):
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        if vectorize:
            k = jnp.vectorize(
                k, signature="(),()->()" if input_shape == () else "(d),(d)->()"
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


class JaxStationaryMixin(StationaryMixin):
    def _squared_euclidean_distances_jax(
        self,
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

        return jnp.sum(diffs ** 2, axis=tuple(range(-self.input_ndim, 0)))
