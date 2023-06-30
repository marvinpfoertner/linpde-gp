from collections.abc import Callable
import functools
from typing import Union

import jax
import jax.numpy as jnp
import probnum as pn
from probnum.typing import ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints
from linpde_gp.functions import JaxFunction, JaxLambdaFunction

from .._arithmetic import SumLinearFunctionOperator
from ._coefficients import MultiIndex, PartialDerivativeCoefficients
from ._lindiffop import LinearDifferentialOperator


class PartialDerivative(LinearDifferentialOperator):
    def __init__(
        self,
        multi_index: MultiIndex,
        *,
        use_jax_fallback: bool = True,
    ) -> None:
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                {(): {multi_index: 1.0}}, multi_index.shape, ()
            ),
            input_shapes=(multi_index.shape, ()),
        )

        self._multi_index = multi_index
        self._use_jax_fallback = use_jax_fallback

    @property
    def multi_index(self) -> MultiIndex:
        return self._multi_index

    @property
    def is_mixed(self) -> bool:
        return self._multi_index.is_mixed

    @property
    def order(self) -> int:
        return self._multi_index.order

    def to_sum(self) -> SumLinearFunctionOperator:
        return SumLinearFunctionOperator(SumLinearFunctionOperator(self))

    def __getitem__(self, idx: tuple[int, ...]) -> "PartialDerivative":
        return PartialDerivative(MultiIndex(self.multi_index[idx]))

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        if self.order == 0:
            return f
        try:
            return super(LinearDifferentialOperator, self).__call__(f, **kwargs)
        except NotImplementedError:
            pass

        if self._use_jax_fallback:
            return self._jax_fallback(f, **kwargs)
        raise NotImplementedError()

    def _jax_fallback(self, f: Callable, /, *, argnum: int = 0, **kwargs) -> Callable:
        return JaxPartialDerivative(self.multi_index)(f, argnum=argnum, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(multi_index={self.multi_index})"


class TimeDerivative(PartialDerivative):
    def __init__(self, domain_shape: ShapeLike) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        if len(domain_shape) == 0:
            multi_index = 1
        elif len(domain_shape) == 1:
            multi_index = (1,) + (0,) * (domain_shape[0] - 1)
        else:
            raise ValueError()

        super().__init__(
            MultiIndex(multi_index),
        )

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()


class JaxPartialDerivative(LinearDifferentialOperator):
    def __init__(
        self,
        multi_index: MultiIndex,
        output_idx: Union[int, tuple[int, ...]] = None,
    ) -> None:
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                {(): {multi_index: 1.0}}, multi_index.shape, ()
            ),
            input_shapes=(multi_index.shape, ()),
        )

        self._multi_index = multi_index
        self._output_idx = output_idx

    @property
    def multi_index(self) -> MultiIndex:
        return self._multi_index

    @property
    def output_idx(self) -> Union[int, tuple[int, ...]]:
        return self._output_idx

    @property
    def is_mixed(self) -> bool:
        return self._multi_index.is_mixed

    def to_sum(self) -> SumLinearFunctionOperator:
        return SumLinearFunctionOperator(SumLinearFunctionOperator(self))

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return JaxLambdaFunction(
            self._derive(f, **kwargs),
            input_shape=self.output_domain_shape,
            output_shape=self.output_codomain_shape,
            vectorize=True,
        )

    @__call__.register
    def _(self, f: JaxFunction, /, **kwargs):
        if f.input_shape != self.input_domain_shape:
            raise ValueError()

        if f.output_shape != self.input_codomain_shape:
            raise ValueError()

        return JaxLambdaFunction(
            self._derive(f.jax, **kwargs),
            input_shape=self.output_domain_shape,
            output_shape=self.output_codomain_shape,
            vectorize=True,
        )

    def _derive(self, f, /, *, argnum=0):
        @jax.jit
        def f_deriv(*args):
            def _f_arg(arg):
                return f(*args[:argnum], arg, *args[argnum + 1 :])

            df = _f_arg
            for single_idx in self.multi_index.split_to_single_order():
                df = lambda x, df=df, single_idx=single_idx: (  # pylint: disable=unnecessary-lambda-assignment,line-too-long
                    jax.jvp(
                        df, (x,), (jnp.array(single_idx.array, dtype=jnp.float64),)
                    )[1]
                )
            if self.output_idx:
                return df(args[argnum])[self.output_idx]
            return df(args[argnum])

        return f_deriv

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(multi_index={self.multi_index})"
