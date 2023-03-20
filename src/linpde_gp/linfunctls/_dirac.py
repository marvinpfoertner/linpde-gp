import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike, ShapeType

from . import _linfunctl


class DiracFunctional(_linfunctl.LinearFunctional):
    def __init__(
        self,
        input_domain_shape: ShapeLike,
        input_codomain_shape: ShapeLike,
        X: ArrayLike,
    ) -> None:
        self._X = np.asarray(X)

        self._X_batch_shape = self._X.shape[: self._X.ndim - len(input_domain_shape)]
        assert self._X.shape == self._X_batch_shape + input_domain_shape

        super().__init__(
            input_shapes=(input_domain_shape, input_codomain_shape),
            output_shape=self._X_batch_shape + input_codomain_shape,
        )

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def X_batch_shape(self) -> ShapeType:
        return self._X_batch_shape

    @property
    def X_batch_ndim(self) -> ShapeType:
        return len(self._X_batch_shape)

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /) -> np.ndarray:
        return f(self._X)
