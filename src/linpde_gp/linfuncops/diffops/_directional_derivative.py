from collections.abc import Callable
import functools

import jax
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from ._lindiffop import LinearDifferentialOperator


class DirectionalDerivative(LinearDifferentialOperator):
    def __init__(self, direction: ArrayLike):
        self._direction = np.asarray(direction)

        super().__init__(input_shapes=(self._direction.shape, ()))

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    def _jax_fallback(self, f: Callable, /, *, argnum: int = 0, **kwargs) -> Callable:
        @jax.jit
        def _f_dir_deriv(*args):
            def _f_arg(arg):
                return f(*args[:argnum], arg, *args[argnum + 1 :])

            _, dir_deriv = jax.jvp(_f_arg, (args[argnum],), (self._direction,))

            return dir_deriv

        return _f_dir_deriv

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()


class PartialDerivative(DirectionalDerivative):
    def __init__(
        self,
        domain_shape: ShapeLike,
        domain_index,
    ) -> None:
        direction = np.zeros(domain_shape)
        direction[domain_index] = 1.0

        super().__init__(direction)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()


class TimeDerivative(PartialDerivative):
    def __init__(self, domain_shape: ShapeLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        assert len(domain_shape) <= 1

        super().__init__(
            domain_shape,
            domain_index=() if domain_shape == () else (0,),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()
