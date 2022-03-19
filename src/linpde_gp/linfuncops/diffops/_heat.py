from __future__ import annotations

import functools

from probnum.typing import ShapeLike

from .._arithmetic import SumLinearFunctionOperator
from ._directional_derivative import TimeDerivative
from ._laplacian import SpatialLaplacian


class HeatOperator(SumLinearFunctionOperator):
    def __init__(self, domain_shape: ShapeLike, alpha: float = 1.0) -> None:
        self._alpha = float(alpha)

        super().__init__(
            TimeDerivative(domain_shape),
            SpatialLaplacian(domain_shape, alpha=-self._alpha),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)
