import functools

import jax
from jax import numpy as jnp
import numpy as np

from .. import function, linfuncops


class Zero(function.JaxFunction):
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(
            x, shape=x.shape[: x.ndim - self._input_ndim] + self.output_shape
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(
            x, shape=x.shape[: x.ndim - self._input_ndim] + self.output_shape
        )


@function.JaxLambdaFunction.__add__.register
def _(self, other: Zero):
    assert other.input_shape == self.input_shape
    assert other.output_shape == self.output_shape

    return self


@linfuncops.LinearFunctionOperator.__call__.register
def _(self, f: Zero, /):
    return Zero(
        input_shape=self.output_domain_shape,
        output_shape=self.output_codomain_shape,
    )
