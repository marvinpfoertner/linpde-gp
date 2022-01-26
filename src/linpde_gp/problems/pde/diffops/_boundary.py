import jax
from jax import numpy as jnp

from ... import linfuncops
from ...typing import JaxFunction


class DirectionalDerivative(linfuncops.JaxLinearOperator):
    def __init__(self, direction):
        self._direction = jnp.asarray(direction)

        super().__init__(L=self._impl)

    def _impl(self, f: JaxFunction, argnum: int = 0) -> JaxFunction:
        @jax.jit
        def _f(*args, **kwargs) -> jnp.ndarray:
            t, x = args[argnum : argnum + 2]
            tx = jnp.concatenate((t[None], x), axis=0)

            return f(
                *args[:argnum],
                tx,
                *args[argnum + 2 :],
                **kwargs,
            )

        f_spatial_grad = jax.grad(_f, argnums=argnum + 1)

        @jax.jit
        def _f_dir_deriv(*args, **kwargs) -> jnp.ndarray:
            tx = args[argnum]
            t, x = tx[0], tx[1:]
            args = args[:argnum] + (t, x) + args[argnum + 1 :]

            return jnp.sum(self._direction * f_spatial_grad(*args, **kwargs))

        return _f_dir_deriv
