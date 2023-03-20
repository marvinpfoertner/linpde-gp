import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike, ShapeType

from . import _linfunctl


class _EvaluationFunctional(_linfunctl.LinearFunctional):
    """A linear functional specifically for evaluating a function at some
    training data. Reshapes the output such that the output shape comes
    before the batch shape, which is how multi-output kernel matrices are
    flattened to 2D in ProbNum.

    As a user, you do not need to touch this class - simply condition without
    specifying a linear functional, and an instance of this class will be
    constructed automatically for you.
    """

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
            output_shape=input_codomain_shape + self._X_batch_shape,
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
        res = f(self._X)
        assert res.shape == self._X_batch_shape + f.output_shape
        if f.output_ndim > 0:
            # Move output dimensions to the front
            return np.moveaxis(
                res,
                res.ndim - f.output_ndim + np.arange(f.output_ndim),
                np.arange(f.output_ndim),
            )
        return res
