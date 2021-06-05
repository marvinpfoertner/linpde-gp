from typing import Tuple

import numpy as np
import probnum as pn
from probnum.type import FloatArgType


class DirichletBoundaryCondition:
    def __init__(
        self,
        domain: Tuple[FloatArgType, FloatArgType],
        boundary_values: Tuple[FloatArgType, FloatArgType],
    ):
        self._domain = tuple(pn.utils.as_numpy_scalar(bound) for bound in domain)
        self._boundary_values = tuple(
            pn.utils.as_numpy_scalar(value) for value in boundary_values
        )

    @property
    def domain(self) -> Tuple[np.floating, np.floating]:
        return self._domain

    def __call__(self, x: FloatArgType) -> np.floating:
        if x == self._domain[0]:
            return self._boundary_values[0]
        elif x == self._domain[1]:
            return self._boundary_values[1]
        else:
            raise ValueError(f"`{x}` is not an element of the boundary of the domain")
