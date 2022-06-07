from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Type

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike, ShapeType

from linpde_gp import functions

if TYPE_CHECKING:
    import linpde_gp


class LinearFunctionOperator:
    def __init__(
        self,
        input_shapes: tuple[ShapeLike, ShapeLike],
        output_shapes: tuple[ShapeLike, ShapeLike],
    ) -> None:
        input_domain_shape, input_codomain_shape = input_shapes

        self._input_domain_shape = pn.utils.as_shape(input_domain_shape)
        self._input_codomain_shape = pn.utils.as_shape(input_codomain_shape)

        output_domain_shape, output_codomain_shape = output_shapes

        self._output_domain_shape = pn.utils.as_shape(output_domain_shape)
        self._output_codomain_shape = pn.utils.as_shape(output_codomain_shape)

    @property
    def input_shapes(self) -> ShapeType:
        return (self._input_domain_shape, self._input_codomain_shape)

    @property
    def input_domain_shape(self) -> ShapeType:
        return self._input_domain_shape

    @property
    def input_domain_ndim(self) -> ShapeType:
        return len(self._input_domain_shape)

    @property
    def input_codomain_shape(self) -> ShapeType:
        return self._input_codomain_shape

    @property
    def input_codomain_ndim(self) -> ShapeType:
        return len(self._input_codomain_shape)

    @property
    def output_shapes(self) -> ShapeType:
        return (self._output_domain_shape, self._output_codomain_shape)

    @property
    def output_domain_shape(self) -> ShapeType:
        return self._output_domain_shape

    @property
    def output_domain_ndim(self) -> ShapeType:
        return len(self._output_domain_shape)

    @property
    def output_codomain_shape(self) -> ShapeType:
        return self._output_codomain_shape

    @property
    def output_codomain_ndim(self) -> ShapeType:
        return len(self._output_codomain_shape)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        raise NotImplementedError()

    @__call__.register
    def _(self, f: functions.JaxSumFunction, /) -> functions.JaxSumFunction:
        return functions.JaxSumFunction(*(self(summand) for summand in f.summands))

    @__call__.register
    def _(self, f: functions.JaxScaledFunction, /) -> functions.JaxScaledFunction:
        return functions.JaxScaledFunction(self(f), scalar=f.scalar)

    @__call__.register
    def _(
        self, f: functions.Zero, /  # pylint: disable=unused-argument
    ) -> functions.JaxSumFunction:
        return functions.Zero(
            input_shape=self.output_domain_shape,
            output_shape=self.output_codomain_shape,
        )

    def to_linfunctl(self, X: np.ndarray) -> "linpde_gp.linfunctls.LinearFunctional":
        from linpde_gp.linfunctls import (  # pylint: disable=import-outside-toplevel
            DiracFunctional,
        )

        return (
            DiracFunctional(
                input_domain_shape=self.output_domain_shape,
                input_codomain_shape=self.output_codomain_shape,
                X=X,
            )
            @ self
        )

    def __neg__(self) -> LinearFunctionOperator:
        return -1.0 * self

    def __add__(self, other: LinearFunctionOperator) -> LinearFunctionOperator:
        from ._arithmetic import (  # pylint: disable=import-outside-toplevel
            SumLinearFunctionOperator,
        )

        return SumLinearFunctionOperator(self, other)

    def __sub__(self, other) -> LinearFunctionOperator | Type[NotImplemented]:
        return self + -other

    def __rmul__(self, other) -> LinearFunctionOperator:
        if np.ndim(other) == 0:
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                ScaledLinearFunctionOperator,
            )

            return ScaledLinearFunctionOperator(self, scalar=other)

        return NotImplemented

    def __matmul__(self, other) -> LinearFunctionOperator:
        from ._arithmetic import (  # pylint: disable=import-outside-toplevel
            CompositeLinearFunctionOperator,
        )

        return CompositeLinearFunctionOperator(self, other)
