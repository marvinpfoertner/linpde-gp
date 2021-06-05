import abc
from typing import Callable, Tuple, Union

import numpy as np
import probnum as pn
from probnum.type import FloatArgType
from probnum_galerkin import bases

from . import _boundary_conditions


class LinearPDE(abc.ABC):
    def __init__(
        self,
        domain: Tuple[FloatArgType, FloatArgType],
        rhs: FloatArgType,
        boundary_condition: _boundary_conditions.DirichletBoundaryCondition,
    ):
        self._domain = tuple(pn.utils.as_numpy_scalar(bound) for bound in domain)
        self._rhs = pn.utils.as_numpy_scalar(rhs)
        self._boundary_condition = boundary_condition

    def solution(self) -> Callable[[FloatArgType], np.floating]:
        raise NotImplementedError("A closed-form solution of the PDE is not available")

    def discretize(self, basis: bases.Basis) -> pn.problems.LinearSystem:
        raise NotImplementedError(
            f"Discretization with basis of type {basis.__class__.__name__} is not "
            f"implemented."
        )
