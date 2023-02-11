from collections.abc import Sequence
import dataclasses
import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike

from linpde_gp import domains, functions
from linpde_gp.typing import DomainLike

from ._linear_pde import LinearPDE


class DirichletBoundaryCondition:
    def __init__(
        self,
        boundary: DomainLike,
        values: pn.functions.Function | ArrayLike,
    ) -> None:
        self._boundary = domains.asdomain(boundary)

        if not isinstance(values, pn.functions.Function):
            values = functions.Constant(self._boundary.shape, values)

        if values.input_shape != self._boundary.shape:
            raise ValueError(
                "The shape of the value function should be equal to the shape of "
                "the domain."
            )

        self._values = values

    @property
    def boundary(self) -> domains.Domain:
        return self._boundary

    @property
    def values(self) -> pn.functions.Function:
        return self._values


def get_1d_dirichlet_boundary_observations(
    dirichlet_bcs: Sequence[DirichletBoundaryCondition],
) -> tuple[np.ndarray, np.ndarray]:
    if len(dirichlet_bcs) != 2 or not all(
        isinstance(bc.boundary, domains.Point) for bc in dirichlet_bcs
    ):
        raise ValueError()

    X_bc = np.asarray([float(bc.boundary) for bc in dirichlet_bcs])
    Y_bc = np.asarray([bc.values(x) for bc, x in zip(dirichlet_bcs, X_bc)])

    return X_bc, Y_bc


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

        if self.solution is not None:
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


class InitialBoundaryValueProblem(BoundaryValueProblem):
    def __init__(
        self,
        pde: LinearPDE,
        initial_condition: DirichletBoundaryCondition,
        boundary_conditions: Sequence[DirichletBoundaryCondition],
        solution: pn.functions.Function | None = None,
    ):
        if not isinstance(pde.domain, domains.CartesianProduct) or len(pde.domain) != 2:
            raise ValueError()

        if initial_condition.boundary != pde.domain[1]:
            raise ValueError()

        self._initial_condition = initial_condition

        super().__init__(
            pde=pde,
            boundary_conditions=boundary_conditions,
            solution=solution,
        )

    @property
    def t0(self) -> float:
        return self.domain[0][0]

    @property
    def T(self) -> float:
        return self.domain[0][1]

    @property
    def spatial_domain(self) -> domains.Domain:
        return self.domain[1]

    @functools.cached_property
    def initial_domain(self) -> domains.CartesianProduct:
        return domains.CartesianProduct(
            domains.Point(self.t0),
            self.spatial_domain,
        )

    @property
    def initial_condition(self) -> DirichletBoundaryCondition:
        return self._initial_condition
