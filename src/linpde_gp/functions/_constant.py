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

    def __rmul__(self, other) -> JaxFunction:
        if np.ndim(other) == 0:
            return Constant(self.input_shape, value=np.asarray(other) * self.value)

        return super().__rmul__(other)


@Constant.__add__.register  # pylint: disable=no-member
def _(self, other: Constant) -> Constant:
    assert self.input_shape == other.input_shape

    return Constant(self.input_shape, value=self.value + other.value)


class Zero(Constant):
    def __init__(self, input_shape: ShapeLike, output_shape: ShapeLike = ()) -> None:
        super().__init__(input_shape, value=np.broadcast_to(0.0, output_shape))

    def __rmul__(self, other) -> JaxFunction:
        if np.ndim(other) == 0:
            return self

        return super().__rmul__(other)


@JaxFunction.__add__.register  # pylint: disable=no-member
@Constant.__add__.register  # pylint: disable=no-member
def _(self, other: Zero):
    assert other.input_shape == self.input_shape
    assert other.output_shape == self.output_shape

    return self
