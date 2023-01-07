import functools
import numpy as np

from ._linfunctl import LinearFunctional
from ._flattened import FlattenedLinearFunctional

class StackedLinearFunctional(LinearFunctional):
    def __init__(self, L1: LinearFunctional, L2: LinearFunctional):
        assert L1.input_shapes == L2.input_shapes
        self._L1 = FlattenedLinearFunctional(L1)
        self._L2 = FlattenedLinearFunctional(L2)

        output_shape = (self._L1.output_shape[0] + self._L2.output_shape[0],)

        super().__init__(
            input_shapes=self._L1.input_shapes,
            output_shape=output_shape
        )

    @property
    def L1(self) -> FlattenedLinearFunctional:
        return self._L1
    
    @property
    def L2(self) -> FlattenedLinearFunctional:
        return self._L2

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return np.concatenate((self._L1(f), self._L2(f)))
