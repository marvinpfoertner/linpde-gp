from collections.abc import Callable
import functools

import jax
import jax.numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

import linpde_gp  # pylint: disable=unused-import # for type hints

from .._arithmetic import CompositeLinearFunctionOperator, SumLinearFunctionOperator
from ._coefficients import MultiIndex, PartialDerivativeCoefficients
from ._lindiffop import LinearDifferentialOperator


class PartialDerivative(LinearDifferentialOperator):
    def __init__(
        self,
        multi_index: MultiIndex,
    ) -> None:
        super().__init__(
            coefficients=PartialDerivativeCoefficients(
                {(): {multi_index: 1.0}}, multi_index.shape, ()
            ),
            input_shapes=(multi_index.shape, ()),
        )

        self._multi_index = multi_index

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

    def get_factor_at_dim(self, idx: tuple[int, ...]) -> "PartialDerivative":
        """Get the factor of the partial derivative at the given input dimension
        index.

        This can be used, for example, to compute a mixed partial derivative
        as the composition of several unmixed partial derivatives.
        """
        return PartialDerivative(MultiIndex(self.multi_index[idx]))

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

    def factorize_first_order(
        self,
    ) -> CompositeLinearFunctionOperator["PartialDerivative"]:
        factors = []
        for idx, order in np.ndenumerate(self.multi_index.array):
            for _ in range(order):
                factors.append(
                    PartialDerivative(
                        MultiIndex.from_index(idx, self.multi_index.shape, 1)
                    )
                )
        return CompositeLinearFunctionOperator(*factors)

    def factorize_dimwise(self) -> CompositeLinearFunctionOperator["PartialDerivative"]:
        factors = []
        for idx, order in np.ndenumerate(self.multi_index.array):
            if order > 0:
                factors.append(
                    PartialDerivative(
                        MultiIndex.from_index(idx, self.multi_index.shape, order)
                    )
                )
        return CompositeLinearFunctionOperator(*factors)

    def _jax_fallback(self, f: Callable, /, *, argnum: int = 0, **kwargs) -> Callable:
        @jax.jit
        def f_deriv(*args):
            def _f_arg(arg):
                return f(*args[:argnum], arg, *args[argnum + 1 :])

            df = _f_arg
            for single_idx in [
                pd.multi_index.array for pd in self.factorize_first_order().linfuncops
            ]:
                df = lambda x, df=df, single_idx=single_idx: (  # pylint: disable=unnecessary-lambda-assignment,line-too-long
                    jax.jvp(df, (x,), (jnp.array(single_idx, dtype=jnp.float64),))[1]
                )
            return df(args[argnum])

        return f_deriv

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(multi_index={self.multi_index})"


class _PartialDerivativeNoJax(PartialDerivative):
    @functools.singledispatchmethod
    def __call__(self, f: Callable, /, **kwargs) -> Callable:
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def weak_form(
        self, test_basis: pn.functions.Function, /
    ) -> "linpde_gp.linfunctls.LinearFunctional":
        return super().weak_form(test_basis)

    def _jax_fallback(self, f: Callable, /, *, argnum: int = 0, **kwargs) -> Callable:
        raise NotImplementedError()


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
