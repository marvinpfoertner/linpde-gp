from __future__ import annotations

import abc
from collections.abc import Callable
import functools

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike


class JaxFunction(pn.Function):
    def jax(self, x: ArrayLike) -> jnp.ndarray:
        x = jnp.asarray(x)

        try:
            # Note that this differs from
            # `np.broadcast_shapes(x.shape, self._input_shape)`
            # if self._input_shape contains `1`s
            jnp.broadcast_to(
                x,
                shape=x.shape[: x.ndim - self._input_ndim] + self._input_shape,
            )
        except ValueError as ve:
            raise ValueError(
                f"The shape of the input {x.shape} can not be broadcast to the "
                f"specified `input_shape` {self._input_shape} of the `JaxFunction`."
            ) from ve

        res = self._evaluate_jax(x)

        assert res.shape == (x.shape[: x.ndim - self._input_ndim] + self._output_shape)

        return res

    @abc.abstractmethod
    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return super()._evaluate(x)

    @functools.singledispatchmethod
    def __add__(self, other: JaxFunction) -> JaxFunction:
        from ._jax_arithmetic import JaxSumFunction

        return JaxSumFunction(self, other)


class JaxLambdaFunction(JaxFunction):
    def __init__(
        self,
        fn: Callable[[jnp.ndarray], jnp.ndarray],
        input_shape: ShapeLike,
        output_shape: ShapeLike = (),
        vectorize: bool = True,
    ) -> None:
        super().__init__(input_shape, output_shape)

        if len(input_shape) > 1:
            raise ValueError("The input shape must be at most one-dimensional")

        if vectorize:
            fn = jnp.vectorize(
                fn,
                signature="()->()" if input_shape == () else "(d)->()",
            )

        self._fn = fn

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.array(self._fn(x))

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._fn(x)
