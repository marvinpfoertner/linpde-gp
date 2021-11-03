import abc
from typing import Callable, Tuple, Union

import numpy as np
import probnum as pn
from probnum.typing import FloatArgType
from linpde_gp import bases

from .. import domains
from . import _boundary_conditions


class LinearPDE(abc.ABC):
    def __init__(
        self,
        domain: domains.DomainLike,
        rhs: FloatArgType,
        boundary_condition: _boundary_conditions.DirichletBoundaryCondition,
    ):
        self._domain = domains.asdomain(domain)
        self._rhs = pn.utils.as_numpy_scalar(rhs)
        self._boundary_condition = boundary_condition

    @property
    def domain(self) -> domains.Domain:
        return self._domain

    def solution(self) -> Callable[[FloatArgType], np.floating]:
        raise NotImplementedError("A closed-form solution of the PDE is not available")

    def discretize(self, basis: bases.Basis) -> pn.problems.LinearSystem:
        raise NotImplementedError(
            f"Discretization with basis of type {basis.__class__.__name__} is not "
            f"implemented."
        )
