import jax
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from .... import linfuncops


class DirectionalDerivative(linfuncops.JaxLinearOperator):
    def __init__(self, direction: ArrayLike):
        self._direction = np.asarray(direction)

        super().__init__(
            L=self._jax_fallback,
            input_shapes=(self._direction.shape, ()),
            output_shapes=(self._direction.shape, ()),
        )

    def _jax_fallback(
        self, f: linfuncops.JaxFunction, argnum: int = 0
    ) -> linfuncops.JaxFunction:
        @jax.jit
        def _f_dir_deriv(*args):
            f_arg = lambda arg: f(*args[:argnum], arg, *args[argnum + 1 :])

            _, dir_deriv = jax.jvp(f_arg, (args[argnum],), (self._direction,))

            return dir_deriv

        return _f_dir_deriv


class PartialDerivative(DirectionalDerivative):
    def __init__(
        self,
        domain_shape: ShapeLike,
        domain_index,
    ) -> None:
        direction = np.zeros(domain_shape)
        direction[domain_index] = 1.0

        super().__init__(direction)


class TimeDerivative(PartialDerivative):
    def __init__(self, domain_shape: ShapeLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        assert len(domain_shape) <= 1

        super().__init__(
            domain_shape,
            domain_index=() if domain_shape == () else (0,),
        )
