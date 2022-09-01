from collections.abc import Sequence

from jax import numpy as jnp
import numpy as np
import probnum as pn

from . import _jax


class StackedFunction(_jax.JaxFunction):
    def __init__(self, *fns: pn.functions.Function, axis: int = -1) -> None:
        self._fns = tuple(fns)
        self._axis = axis

        input_shape = self._fns[0].input_shape

        assert all(f.input_shape == input_shape for f in self._fns)

        output_shape = np.stack(
            [np.empty(f.output_shape, dtype=[]) for f in self._fns],
            axis=self._axis,
        ).shape

        super().__init__(input_shape, output_shape)

    @property
    def fns(self) -> Sequence[pn.functions.Function]:
        return self._fns

    @property
    def axis(self) -> int:
        return self._axis

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.stack(
            [f(x) for f in self._fns],
            axis=self._axis,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [f.jax(x) for f in self._fns],
            axis=self._axis,
        )


def stack(fns: Sequence[pn.functions.Function], axis: int = -1) -> StackedFunction:
    return StackedFunction(*fns, axis=axis)
