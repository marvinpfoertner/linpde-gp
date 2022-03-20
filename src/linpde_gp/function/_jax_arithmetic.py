from __future__ import annotations

from collections.abc import Iterable
import functools
import operator

from jax import numpy as jnp
import numpy as np
from probnum.typing import ScalarLike, ScalarType

from ._jax import JaxFunction


class JaxScaledFunction(JaxFunction):
    def __init__(self, function: JaxFunction, scalar: ScalarLike):
        if not isinstance(function, JaxFunction):
            raise TypeError()

        self._function = function
        self._scalar = np.asarray(scalar, dtype=np.double)

        super().__init__(
            input_shape=self._function.input_shape,
            output_shape=self._function.output_shape,
        )

    @property
    def function(self) -> JaxFunction:
        return self._function

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._scalar * self._function(x)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._scalar * self._function.jax(x)

    def __rmul__(self, other) -> JaxFunction:
        if np.ndim(other) == 0:
            return JaxScaledFunction(
                function=self._function,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)


class JaxSumFunction(JaxFunction):
    def __init__(self, *summands: JaxFunction) -> None:
        self._summands = JaxSumFunction._expand_summands(summands)

        input_shape = summands[0].input_shape
        output_shape = summands[0].output_shape

        assert all(summand.input_shape == input_shape for summand in summands)
        assert all(summand.output_shape == output_shape for summand in summands)

        super().__init__(input_shape=input_shape, output_shape=output_shape)

    @property
    def summands(self) -> tuple[JaxSumFunction, ...]:
        return self._summands

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return functools.reduce(
            operator.add, (summand(x) for summand in self._summands)
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return functools.reduce(
            operator.add, (summand.jax(x) for summand in self._summands)
        )

    @staticmethod
    def _expand_summands(summands: Iterable[JaxFunction]) -> tuple[JaxFunction]:
        expanded_summands = []

        for summand in summands:
            if isinstance(summand, JaxSumFunction):
                expanded_summands.extend(summand.summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)
