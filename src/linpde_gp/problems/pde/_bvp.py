from collections.abc import Sequence
import dataclasses

import probnum as pn

from linpde_gp import domains

from ._linear_pde import LinearPDE


@dataclasses.dataclass(frozen=True)
class DirichletBoundaryCondition:
    boundary: domains.Domain
    values: pn.randprocs.RandomProcess | pn.randvars.RandomVariable


@dataclasses.dataclass(frozen=True)
class BoundaryValueProblem:
    pde: LinearPDE
    boundary_conditions: Sequence[DirichletBoundaryCondition]
    solution: pn.functions.Function | None = None
