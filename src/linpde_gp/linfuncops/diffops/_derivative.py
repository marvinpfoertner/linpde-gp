import functools

import probnum as pn

import linpde_gp  # pylint: disable=unused-import # for type hints

from ._lindiffop import LinearDifferentialOperator


class Derivative(LinearDifferentialOperator):
    def __init__(
        self,
        order: int,
    ) -> None:
        if order < 0:
            raise ValueError(f"Order must be >= 0, but got {order}.")

        super().__init__(input_shapes=((), ()))

        self._order = order

    @property
    def order(self) -> int:
        return self._order

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        if self.order == 0:
            return f
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()
