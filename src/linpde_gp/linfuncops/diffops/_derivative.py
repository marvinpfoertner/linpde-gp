import functools
from typing import Callable

import jax
import probnum as pn

import linpde_gp  # pylint: disable=unused-import # for type hints

from ._coefficients import MultiIndex, PartialDerivativeCoefficients
from ._lindiffop import LinearDifferentialOperator


class Derivative(LinearDifferentialOperator):
    def __init__(
        self,
        order: int,
    ) -> None:
        if order < 0:
            raise ValueError(f"Order must be >= 0, but got {order}.")

        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                {(): {MultiIndex(order): 1.0}}, (), ()
            ),
            input_shapes=((), ()),
        )

        self._order = order

    @property
    def order(self) -> int:
        return self._order

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        if self.order == 0:
            return f
        return super().__call__(f, **kwargs)

    def _jax_fallback(self, f: Callable, /, *, argnum: int = 0, **kwargs) -> Callable:
        @jax.jit
        def _f_deriv(*args):
            def _f_arg(arg):
                return f(*args[:argnum], arg, *args[argnum + 1 :])

            _, deriv = jax.jvp(_f_arg, (args[argnum],), (1.0,))

            return deriv

        return _f_deriv

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()
