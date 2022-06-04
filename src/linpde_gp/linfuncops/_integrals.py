import functools

import numpy as np
import probnum as pn
from probnum.typing import FloatLike

from linpde_gp.functions import Affine, Constant

from . import _linfuncop


class UndefinedLebesgueIntegral(_linfuncop.LinearFunctionOperator):
    def __init__(self, lower_bound: FloatLike = 0.0) -> None:
        super().__init__(input_shapes=((), ()), output_shapes=((), ()))

        self._lower_bound = lower_bound

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.Function, /) -> pn.Function:
        import scipy.integrate

        def _int_f(b):
            unsqueeze = False

            if b.size == 1 and b.ndim == 1:
                unsqueeze = True
                b = b[0]

            assert b.shape == ()

            int_f = np.asarray(scipy.integrate.quad(f, a=self.lower_bound, b=b)[0])

            if unsqueeze:
                return int_f[None]

            return int_f

        return pn.LambdaFunction(
            _int_f,
            input_shape=(),
            output_shape=(),
        )


@UndefinedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Constant) -> Affine:
    return Affine(
        A=f.value,
        b=-f.value * self.lower_bound,
    )
