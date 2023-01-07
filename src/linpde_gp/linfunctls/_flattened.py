import functools

import numpy as np
import probnum as pn

from ._linfunctl import LinearFunctional

class FlattenedLinearFunctional(LinearFunctional):
    def __init__(self, L: LinearFunctional):
        output_shape = L.output_shape
        if len(output_shape) == 0:
            output_shape = (1,)
        else:
            num_vals = np.prod(output_shape)
            output_shape = (num_vals,)
        self._L = L
        super().__init__(
            input_shapes=L.input_shapes,
            output_shape=output_shape
        )

    @property
    def inner_functional(self) -> LinearFunctional:
        return self._L

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /) -> np.ndarray:
        return self._L(f).flatten()