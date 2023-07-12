from collections.abc import Callable
import functools
from typing import TYPE_CHECKING

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints
from linpde_gp.functions import JaxFunction, JaxLambdaFunction

from .._arithmetic import SumLinearFunctionOperator
from .._linfuncop import LinearFunctionOperator
from .._select_output import SelectOutput
from ._coefficients import PartialDerivativeCoefficients

if TYPE_CHECKING:
    from typing import Union

    from .._arithmetic import CompositeLinearFunctionOperator
    from ._partial_derivative import PartialDerivative, _PartialDerivativeNoJax


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
    PartialDerivativeSummand = "Union[PartialDerivative, CompositeLinearFunctionOperator]"  # pylint: disable=line-too-long

    def to_sum(
        self,
    ) -> (
        "SumLinearFunctionOperator[SumLinearFunctionOperator[PartialDerivativeSummand]]"
    ):
        from ._partial_derivative import (  # pylint: disable=import-outside-toplevel
            PartialDerivative,
        )

        outer_summands = []
        for output_index in self.coefficients:
            inner_summands = []
            for multi_index in self.coefficients[output_index]:
                partial_diffop = PartialDerivative(multi_index)
                if output_index != ():  # pylint: disable=comparison-with-callable
                    partial_diffop = partial_diffop @ SelectOutput(
                        self.input_shapes, output_index
                    )
                inner_summands.append(
                    self.coefficients[output_index][multi_index] * partial_diffop
                )
            outer_summands.append(SumLinearFunctionOperator(*inner_summands))
        return SumLinearFunctionOperator(*outer_summands)

    def _to_sum_no_jax(
        self,
    ) -> (
        "SumLinearFunctionOperator[SumLinearFunctionOperator[_PartialDerivativeNoJax]]"
    ):
        from ._partial_derivative import (  # pylint: disable=import-outside-toplevel
            _PartialDerivativeNoJax,
        )

        outer_summands = []
        for output_index in self.coefficients:
            inner_summands = []
            for multi_index in self.coefficients[output_index]:
                partial_diffop = _PartialDerivativeNoJax(multi_index)
                if output_index != ():  # pylint: disable=comparison-with-callable
                    partial_diffop = partial_diffop @ SelectOutput(
                        self.input_shapes, output_index
                    )
                inner_summands.append(
                    self.coefficients[output_index][multi_index] * partial_diffop
                )
            outer_summands.append(SumLinearFunctionOperator(*inner_summands))
        return SumLinearFunctionOperator(*outer_summands)

    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        try:
            return self._call_no_jax(f, **kwargs)
        except NotImplementedError:
            pass

        if isinstance(f, JaxFunction):
            if f.input_shape != self.input_domain_shape:
                raise ValueError()

            if f.output_shape != self.input_codomain_shape:
                raise ValueError()

            return JaxLambdaFunction(
                self._jax_fallback(f.jax, **kwargs),
                input_shape=self.output_domain_shape,
                output_shape=self.output_codomain_shape,
                vectorize=True,
            )

        return JaxLambdaFunction(
            self._jax_fallback(f, **kwargs),
            input_shape=self.output_domain_shape,
            output_shape=self.output_codomain_shape,
            vectorize=True,
        )

    def _call_no_jax(self, f, **kwargs):
        try:
            return super().__call__(f, **kwargs)
        except NotImplementedError:
            pass

        from ._partial_derivative import (  # pylint: disable=import-outside-toplevel
            PartialDerivative,
        )

        if isinstance(self, PartialDerivative):
            raise NotImplementedError()
        return self._to_sum_no_jax()(f, **kwargs)

    def _jax_fallback(  # pylint: disable=arguments-differ
        self, f: Callable, /, *, argnum: int = 0, **kwargs
    ) -> Callable:
        return self.to_sum()(f, argnum=argnum, **kwargs)

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
