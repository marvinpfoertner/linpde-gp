import abc
from collections.abc import Callable
import functools

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints
from linpde_gp.functions import JaxFunction, JaxLambdaFunction

from .._linfuncop import LinearFunctionOperator
from ._coefficients import PartialDerivativeCoefficients


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

    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        try:
            return super().__call__(f, **kwargs)
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

    @abc.abstractmethod
    def _jax_fallback(self, f: Callable, /, **kwargs) -> Callable:
        pass

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
