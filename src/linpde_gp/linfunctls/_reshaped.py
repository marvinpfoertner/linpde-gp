import functools

import numpy as np
import probnum as pn

from ._linfunctl import LinearFunctional

class ReshapedLinearFunctional(LinearFunctional):
    def __init__(self, linfctl: LinearFunctional, order='C'):
        self._linfctl = linfctl
        self._order = order
        super().__init__(
            input_shapes=linfctl.input_shapes,
            output_shape=(linfctl.output_size,)
        )

    @property
    def linfctl(self) -> LinearFunctional:
        return self._linfctl

    @property
    def order(self) -> str:
        return self._order

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /) -> np.ndarray:
        return self._linfctl(f).flatten(order=self._order)