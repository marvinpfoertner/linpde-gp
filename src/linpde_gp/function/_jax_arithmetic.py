from collections.abc import Iterable
import functools
import operator

from jax import numpy as jnp
import numpy as np

from ._jax import JaxFunction


class JaxSumFunction(JaxFunction):
    def __init__(self, *summands: JaxFunction) -> None:
        self._summands = JaxSumFunction._expand_summands(summands)

        input_shape = summands[0].input_shape
        output_shape = summands[0].output_shape

        assert all(summand.input_shape == input_shape for summand in summands)
        assert all(summand.output_shape == output_shape for summand in summands)

        super().__init__(input_shape=input_shape, output_shape=output_shape)

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
                expanded_summands.extend(summand._summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)
