import numbers
from typing import Callable, Tuple, Union

import numpy as np
import probnum as pn
from probnum.type import FloatArgType

from .. import domains


class DirichletBoundaryCondition:
    def __init__(
        self,
        domain: domains.DomainLike,
        boundary_values: Union[
            FloatArgType,
            Tuple[FloatArgType, FloatArgType],
            Callable[[FloatArgType], FloatArgType],
        ],
    ):
        self._domain = domains.asdomain(domain)

        if isinstance(boundary_values, tuple) and len(boundary_values) == 2:
            if not isinstance(self._domain, domains.Interval):
                raise TypeError(
                    "Boundary values can only be given as a tuple if the domain is an "
                    "interval."
                )

            self._boundary_values = tuple(
                pn.utils.as_numpy_scalar(value, dtype=self._domain.dtype)
                for value in boundary_values
            )

            def g(x):
                if x == self._domain[0]:
                    return self._boundary_values[0]
                elif x == self._domain[1]:
                    return self._boundary_values[1]
                else:
                    raise ValueError(
                        f"`{x}` is not an element of the boundary of the domain"
                    )

            self._boundary_fn = g
        elif isinstance(boundary_values, numbers.Real):
            self._boundary_values = pn.utils.as_numpy_scalar(
                boundary_values, dtype=self._domain.dtype
            )
            self._boundary_fn = lambda x: self._boundary_values
        else:
            self._boundary_values = None
            self._boundary_fn = lambda x: pn.utils.as_numpy_scalar(
                boundary_values(x), dtype=self._domain.dtype
            )

    @property
    def domain(self) -> domains.Domain:
        return self._domain

    @property
    def constant(self) -> bool:
        return isinstance(self._boundary_values, np.inexact)

    @property
    def boundary_value(self) -> np.floating:
        if not self.constant:
            raise RuntimeError(
                "The property `boundary_value` is only defined for constant boundary "
                "value."
            )

        return self._boundary_values

    def __call__(self, x: FloatArgType) -> np.floating:
        return self._boundary_fn(x)
