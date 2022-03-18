import abc
from collections.abc import Callable
import functools
from typing import Any

from probnum.typing import ShapeLike

from . import _linfuncop
from ..function import JaxFunction, JaxLambdaFunction


class JaxLinearOperator(_linfuncop.LinearFunctionOperator):
    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        try:
            return super().__call__(f, **kwargs)
        except NotImplementedError:
            if isinstance(f, JaxFunction):
                if f.input_shape != self.input_domain_shape:
                    raise ValueError()

                if f.output_ndim != self.input_codomain_shape:
                    raise ValueError()

                return JaxLambdaFunction(
                    self._jax_fallback(f.jax, **kwargs),
                    input_shape=self.output_domain_shape,
                    output_shape=self.output_codomain_shape,
                    vectorize=True,
                )

            return self._jax_fallback(f, **kwargs)

    @abc.abstractmethod
    def _jax_fallback(self, f: Callable, /, **kwargs) -> Callable:
        pass


class JaxLambdaLinearOperator(JaxLinearOperator):
    def __init__(
        self,
        jax_linop_fn: Callable[[Callable, Any], Callable],
        /,
        input_shapes: tuple[ShapeLike, ShapeLike],
        output_shapes: tuple[ShapeLike, ShapeLike],
    ) -> None:
        super().__init__(input_shapes=input_shapes, output_shapes=output_shapes)

        self._jax_linop_fn = jax_linop_fn

    def _jax_fallback(self, f: Callable, **kwargs) -> Callable:
        return self._jax_linop_fn(f, **kwargs)
