import abc
from collections.abc import Callable
import functools
from typing import TYPE_CHECKING, Union

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints

from .._arithmetic import CompositeLinearFunctionOperator, SumLinearFunctionOperator
from .._linfuncop import LinearFunctionOperator
from .._select_output import SelectOutput
from ._coefficients import PartialDerivativeCoefficients

if TYPE_CHECKING:
    from ._partial_derivative import JaxPartialDerivative, PartialDerivative


class LinearDifferentialOperator(LinearFunctionOperator):
    """Linear differential operator that maps to functions with codomain R."""

    def __init__(
        self,
        coefficients: PartialDerivativeCoefficients,
        input_shapes: tuple[ShapeLike, ShapeLike],
    ) -> None:
        if coefficients.input_domain_shape != input_shapes[0]:
            raise ValueError()
        if coefficients.input_codomain_shape != input_shapes[1]:
            raise ValueError()

        super().__init__(
            input_shapes=input_shapes,
            output_shapes=(input_shapes[0], ()),
        )

        self._coefficients = coefficients

    @property
    def coefficients(self) -> PartialDerivativeCoefficients:
        return self._coefficients

    @property
    def has_mixed(self) -> bool:
        return self._coefficients.has_mixed

    # PartialDerivative or PartialDerivative @ SelectOutput
    PartialDerivativeSummand = Union[PartialDerivative, CompositeLinearFunctionOperator]

    def to_sum(
        self,
    ) -> (
        "SumLinearFunctionOperator[SumLinearFunctionOperator[PartialDerivativeSummand]]"
    ):
        from ._partial_derivative import PartialDerivative

        outer_summands = []
        for output_index in self.coefficients:
            inner_summands = []
            for multi_index in self.coefficients[output_index]:
                partial_diffop = PartialDerivative(multi_index, use_jax_fallback=False)
                if output_index != ():  # pylint: disable=comparison-with-callable
                    partial_diffop = partial_diffop @ SelectOutput(
                        self.input_shapes, output_index
                    )
                inner_summands.append(
                    self.coefficients[output_index][multi_index] * partial_diffop
                )
            outer_summands.append(SumLinearFunctionOperator(*inner_summands))
        return SumLinearFunctionOperator(*outer_summands)

    def to_sum_jax(
        self,
    ) -> "SumLinearFunctionOperator[SumLinearFunctionOperator[JaxPartialDerivative]]":
        from ._partial_derivative import JaxPartialDerivative

        outer_summands = []
        for output_index in self.coefficients:
            inner_summands = []
            for multi_index in self.coefficients[output_index]:
                partial_diffop = JaxPartialDerivative(
                    multi_index, output_index if output_index != () else None
                )
                inner_summands.append(
                    self.coefficients[output_index][multi_index] * partial_diffop
                )
            outer_summands.append(SumLinearFunctionOperator(*inner_summands))
        return SumLinearFunctionOperator(*outer_summands)

    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        try:
            return super().__call__(f, **kwargs)
        except NotImplementedError:
            pass
        try:
            return self.to_sum()(f, **kwargs)
        except NotImplementedError:
            # We intentionally disable the jax fallbacks of `PartialDerivative`
            # in to_sum so that we have the option of using a more efficient
            # jax fallback.
            pass
        return self._jax_fallback(f, **kwargs)

    def _jax_fallback(  # pylint: disable=arguments-differ
        self, f: Callable, /, *, argnum: int = 0, **kwargs
    ) -> Callable:
        return self.to_sum_jax()(f, argnum=argnum, **kwargs)

    def __rmul__(self, other) -> LinearFunctionOperator:
        if np.ndim(other) == 0:
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                ScaledLinearDifferentialOperator,
            )

            return ScaledLinearDifferentialOperator(self, scalar=other)

        return NotImplemented

    @functools.singledispatchmethod
    def weak_form(
        self, basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()
