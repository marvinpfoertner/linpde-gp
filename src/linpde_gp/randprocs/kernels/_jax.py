from __future__ import annotations

import abc
from collections.abc import Callable
import functools
import operator
from typing import Optional

from jax import numpy as jnp
import numpy as np
from probnum.randprocs.kernels import Kernel
from probnum.typing import ArrayLike, ShapeLike

from ... import linfuncops

Kernel.input_size = property(
    lambda self: functools.reduce(operator.mul, self.input_shape, 1)
)

Kernel._batched_sum = (  # pylint: disable=protected-access
    lambda self, a, **sum_kwargs: np.sum(
        a, axis=tuple(range(-self.input_ndim, 0)), **sum_kwargs
    )
)

Kernel._batched_euclidean_norm_sq = (  # pylint: disable=protected-access
    lambda self, a, **sum_kwargs: self._batched_sum(  # pylint: disable=protected-access
        a ** 2, **sum_kwargs
    )
)


class JaxKernelMixin:
    """Careful: Must come before Kernel in inheritance"""

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

    def _batched_sum_jax(self, a: jnp.ndarray, **sum_kwargs) -> jnp.ndarray:
        return jnp.sum(a, axis=tuple(range(-self.input_ndim, 0)), **sum_kwargs)

    def _batched_euclidean_norm_sq_jax(
        self, a: jnp.ndarray, **sum_kwargs
    ) -> jnp.ndarray:
        return self._batched_sum_jax(a ** 2, **sum_kwargs)

    def __add__(self, other: Kernel) -> JaxKernel:
        from ._jax_arithmetic import (  # pylint: disable=import-outside-toplevel
            JaxSumKernel,
        )

        return JaxSumKernel(self, other)

    def __rmul__(self, other: ArrayLike) -> JaxKernel:
        if np.ndim(other) == 0:
            from ._jax_arithmetic import (  # pylint: disable=import-outside-toplevel
                JaxScaledKernel,
            )

            return JaxScaledKernel(kernel=self, scalar=other)

        return super().__rmul__(self, other)


class JaxKernel(JaxKernelMixin, Kernel):
    ...


@linfuncops.JaxLinearOperator.__call__.register  # pylint: disable=no-member
def _(self, k: JaxKernelMixin, /, *, argnum=0):
    try:
        return super(linfuncops.JaxLinearOperator, self).__call__(k, argnum=argnum)
    except NotImplementedError:
        return JaxLambdaKernel(
            self._jax_fallback(  # pylint: disable=protected-access
                k.jax, argnum=argnum
            ),
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
