import functools
import numpy as np

from ._linfunctl import LinearFunctional
from ._reshaped import ReshapedLinearFunctional

class StackedLinearFunctional(LinearFunctional):
    def __init__(self, linfctl_1: LinearFunctional, linfctl_2: LinearFunctional):
        assert linfctl_1.input_shapes == linfctl_2.input_shapes
        self._linfctl_1 = ReshapedLinearFunctional(linfctl_1)
        self._linfctl_2 = ReshapedLinearFunctional(linfctl_2)

        output_shape = (self._linfctl_1.output_shape[0] + self._linfctl_2.output_shape[0],)

        super().__init__(
            input_shapes=self._linfctl_1.input_shapes,
            output_shape=output_shape
        )

    @property
    def linfctl_1(self) -> ReshapedLinearFunctional:
        return self._linfctl_1
    
    @property
    def linfctl_2(self) -> ReshapedLinearFunctional:
        return self._linfctl_2

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return np.concatenate((self._linfctl_1(f), self._linfctl_2(f)))
