import functools

import jax
import numpy as np
from jax import numpy as jnp

from .. import linfuncops


class JaxMean:
    def __init__(self, m, vectorize: bool = True):
        if vectorize:
            m = jnp.vectorize(m, signature="(d)->()")

        self._m = m

    def __call__(self, x: np.ndarray):
        return np.array(self._m(x))

    @property
    def jax(self):
        return self._m


@linfuncops.LinearFunctionOperator.__call__.register
def _(self, f: JaxMean, *, argnum=0, **kwargs):
    assert argnum == 0

    return JaxMean(
        self._L(f.jax, argnum=argnum, **kwargs),
        vectorize=True,
    )


class Zero(JaxMean):
    def __init__(self):
        super().__init__(self._jax, vectorize=False)

    def __call__(self, x: np.ndarray):
        return np.zeros_like(x[..., 0])

    @functools.partial(jax.jit, static_argnums=0)
    def _jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(x[..., 0])
