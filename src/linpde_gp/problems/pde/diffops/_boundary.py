import jax
from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike, ShapeLike

from .... import linfuncops


class DirectionalDerivative(linfuncops.JaxLinearOperator):
    def __init__(self, direction: ArrayLike):
        self._direction = np.asarray(direction)

        super().__init__(
            L=self._impl,
            input_shapes=(self._direction.shape, ()),
            output_shapes=(self._direction.shape, ()),
        )

    def _impl(
        self, f: linfuncops.JaxFunction, argnum: int = 0
    ) -> linfuncops.JaxFunction:
        f_grad = jax.grad(f, argnums=argnum)

        @jax.jit
        def _f_dir_deriv(*args, **kwargs) -> jnp.ndarray:
            return jnp.sum(self._direction * f_grad(*args, **kwargs))

        return _f_dir_deriv
