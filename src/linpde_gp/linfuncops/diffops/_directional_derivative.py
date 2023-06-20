from collections.abc import Callable
import functools

import jax
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints

from ._lindiffop import LinearDifferentialOperator, PartialDerivativeCoefficients


class DirectionalDerivative(LinearDifferentialOperator):
    def __init__(self, direction: ArrayLike):
        direction = np.asarray(direction)
        if direction.ndim > 1:
            raise ValueError("Direction must be element of R^n.")

        def get_one_hot(index: int) -> np.ndarray:
            one_hot = np.zeros(direction.size, dtype=int)
            one_hot[index] = 1
            return tuple(one_hot)

        coefficients = PartialDerivativeCoefficients(
            {
                (): {
                    get_one_hot(domain_index): coefficient
                    for domain_index, coefficient in enumerate(direction.reshape(-1))
                    if coefficient != 0.0
                }
            }
        )
        super().__init__(
            coefficients=coefficients,
            input_shapes=(direction.shape, ()),
        )

        self._direction = direction

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
