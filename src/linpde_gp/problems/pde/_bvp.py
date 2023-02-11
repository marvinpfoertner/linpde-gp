from collections.abc import Sequence
import dataclasses

import numpy as np
import probnum as pn

from linpde_gp import domains

from ._linear_pde import LinearPDE


@dataclasses.dataclass(frozen=True)
class DirichletBoundaryCondition:
    boundary: domains.Domain
    values: pn.functions.Function | np.ndarray


@dataclasses.dataclass(frozen=True)
class BoundaryValueProblem:
    pde: LinearPDE
    boundary_conditions: Sequence[DirichletBoundaryCondition]
    solution: pn.functions.Function | None = None

    @property
    def domain(self):
        return self.pde.domain

    def __post_init__(self):
        for boundary_condition in self.boundary_conditions:
            if boundary_condition.boundary.shape != self.domain.shape:
                raise ValueError(
                    "The shape of the boundary must be equal to the shape of the domain"
                )

        if self.solution.input_shape != self.domain.shape:
            raise ValueError(
                "The input shape of the solution function should be equal to the shape "
                "of the domain."
            )

        if self.solution.output_shape != self.pde.diffop.input_codomain_shape:
            raise ValueError(
                "The output shape of the solution function should be equal to the "
                "output shape of the differential operator's input function."
            )
