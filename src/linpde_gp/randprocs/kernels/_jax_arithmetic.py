from collections.abc import Iterable
import functools
import operator
from typing import Optional

from jax import numpy as jnp
import numpy as np

from ._jax import JaxKernel


class JaxSumKernel(JaxKernel):
    def __init__(self, *summands: JaxKernel):
        self._summands = JaxSumKernel._expand_summands(summands)

        assert len(self._summands) > 0

        input_shape = self._summands[0].input_shape
        output_shape = self._summands[0].output_shape

        # TODO: Replace by broadcasting
        assert all(summand.input_shape == input_shape for summand in self._summands)
        assert all(summand.output_shape == output_shape for summand in self._summands)

        super().__init__(input_shape, output_shape=output_shape)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return functools.reduce(
            operator.add,
            (summand(x0, x1) for summand in self._summands),
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return functools.reduce(
            operator.add,
            (summand.jax(x0, x1) for summand in self._summands),
        )

    @staticmethod
    def _expand_summands(summands: Iterable[JaxKernel]) -> tuple[JaxKernel]:
        expanded_summands = []

        for summand in summands:
            if isinstance(summand, JaxSumKernel):
                expanded_summands.extend(summand._summands)
            else:
                expanded_summands.append(summand)

        return tuple(expanded_summands)


@JaxKernel.__add__.register
def _(self, other: JaxKernel) -> JaxKernel:
    return JaxSumKernel(self, other)
