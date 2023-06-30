from __future__ import annotations

import functools

import numpy as np
import probnum as pn
from probnum.typing import FloatLike, ShapeLike

from .._arithmetic import SumLinearFunctionOperator
from ._partial_derivative import TimeDerivative
from ._laplacian import WeightedLaplacian


class HeatOperator(SumLinearFunctionOperator):
    def __init__(self, domain_shape: ShapeLike, alpha: FloatLike = 1.0) -> None:
        domain_shape = pn.utils.as_shape(domain_shape)

        if len(domain_shape) != 1:
            raise ValueError(
                "The `HeatOperator` only applies to functions with `input_ndim == 1`."
            )

        self._alpha = float(alpha)

        laplacian_weights = np.zeros(domain_shape, dtype=np.double)
        laplacian_weights[1:] = -self._alpha

        super().__init__(
            TimeDerivative(domain_shape),
            WeightedLaplacian(laplacian_weights),
        )

    @property
    def alpha(self) -> np.ndarray:
        return self._alpha

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)
