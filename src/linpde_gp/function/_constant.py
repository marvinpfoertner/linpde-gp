from __future__ import annotations

import functools

from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike, ShapeLike

from ._jax import JaxFunction


class Constant(JaxFunction):
    def __init__(self, input_shape: ShapeLike, value: ArrayLike) -> None:
        self._value = np.asarray(value)

        super().__init__(input_shape, output_shape=self._value.shape)

    @property
    def value(self) -> np.ndarray:
        return self._value

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            self._value.copy(),
            shape=x.shape[: x.ndim - self.input_ndim] + self.output_shape,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(
            self._value.copy(),
            shape=x.shape[: x.ndim - self.input_ndim] + self.output_shape,
        )

    @functools.singledispatchmethod
    def __add__(self, other: JaxFunction) -> JaxFunction:
        return super().__add__(other)


@Constant.__add__.register
def _(self, other: Constant) -> Constant:
    assert self.input_shape == other.input_shape

    return Constant(self.input_shape, value=self.value + other.value)
