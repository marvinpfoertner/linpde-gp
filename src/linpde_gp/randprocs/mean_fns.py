import functools

import jax
from jax import numpy as jnp
import numpy as np

from .. import linfuncops


class Zero(linfuncops.JaxFunction):
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(
            x, shape=x.shape[: x.ndim - self._input_ndim] + self.output_shape
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(
            x, shape=x.shape[: x.ndim - self._input_ndim] + self.output_shape
        )


@linfuncops.LinearFunctionOperator.__call__.register
def _(self, f: Zero, **kwargs):
    return Zero(
        input_shape=f.input_shape,  # TODO: This should be an attribute of `self`
        output_shape=f.output_shape,  # TODO: This should be an attribute of `self`
    )
