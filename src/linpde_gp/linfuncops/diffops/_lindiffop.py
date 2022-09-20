import abc
from collections.abc import Callable
import functools

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints
from linpde_gp.functions import JaxFunction, JaxLambdaFunction

from .._linfuncop import LinearFunctionOperator


class LinearDifferentialOperator(LinearFunctionOperator):
    def __init__(
        self,
        input_shapes: tuple[ShapeLike, ShapeLike],
        output_codomain_shape: ShapeLike = (),
    ) -> None:
        input_shapes = tuple(input_shapes)

        super().__init__(
            input_shapes=input_shapes,
            output_shapes=(input_shapes[0], output_codomain_shape),
        )

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


class LambdaLinearDifferentialOperator(LinearDifferentialOperator):
    def __init__(
        self,
        jax_diffop_fn,
        /,
        input_shapes: tuple[ShapeLike, ShapeLike],
        output_codomain_shape: ShapeLike = (),
    ) -> None:
        super().__init__(input_shapes, output_codomain_shape)

        self._jax_diffop_fn = jax_diffop_fn

    def _jax_fallback(self, f: Callable, /, **kwargs) -> Callable:
        return self._jax_diffop_fn(f, **kwargs)
