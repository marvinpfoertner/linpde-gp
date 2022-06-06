from __future__ import annotations

import functools
from typing import Type

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike, ShapeType

from linpde_gp import functions, linfuncops


class LinearFunctional:
    def __init__(
        self,
        input_shapes: tuple[ShapeLike, ShapeLike],
        output_shape: ShapeLike,
    ) -> None:
        input_domain_shape, input_codomain_shape = input_shapes

        self._input_domain_shape = pn.utils.as_shape(input_domain_shape)
        self._input_codomain_shape = pn.utils.as_shape(input_codomain_shape)

        self._output_shape = pn.utils.as_shape(output_shape)

    @property
    def input_shapes(self) -> ShapeType:
        return (self._input_domain_shape, self._input_codomain_shape)

    @property
    def input_domain_shape(self) -> ShapeType:
        return self._input_domain_shape

    @property
    def input_domain_ndim(self) -> int:
        return len(self.input_domain_shape)

    @property
    def input_codomain_shape(self) -> ShapeType:
        return self._input_codomain_shape

    @property
    def input_codomain_ndim(self) -> int:
        return len(self.input_codomain_shape)

    @property
    def output_shape(self) -> ShapeType:
        return self._output_shape

    @property
    def output_ndim(self) -> int:
        return len(self.output_shape)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        raise NotImplementedError()

    @__call__.register
    def _(self, f: functions.JaxSumFunction, /) -> np.ndarray:
        return sum(self(summand) for summand in f.summands)

    @__call__.register
    def _(self, f: functions.JaxScaledFunction, /) -> np.ndarray:
        return f.scalar * self(f)

    @__call__.register
    def _(self, f: functions.Zero, /) -> np.ndarray:  # pylint: disable=unused-argument
        return np.zeros(shape=())

    def __neg__(self) -> LinearFunctional:
        return -1.0 * self

    def __add__(self, other) -> LinearFunctional | Type[NotImplemented]:
        if isinstance(other, LinearFunctional):
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                SumLinearFunctional,
            )

            return SumLinearFunctional(self, other)

        return NotImplemented

    def __sub__(self, other) -> LinearFunctional | Type[NotImplemented]:
        return self + (-other)

    def __rmul__(self, other) -> LinearFunctional | Type[NotImplemented]:
        if np.ndim(other) == 0:
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                ScaledLinearFunctional,
            )

            return ScaledLinearFunctional(linfunctl=self, scalar=other)

        return NotImplemented

    def __matmul__(self, other) -> LinearFunctional | Type[NotImplemented]:
        if isinstance(other, linfuncops.LinearFunctionOperator):
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                CompositeLinearFunctional,
            )

            return CompositeLinearFunctional(self, other)

        return NotImplemented
