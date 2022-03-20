from collections.abc import Sequence
import dataclasses
from typing import Optional, Union

import probnum as pn

from linpde_gp import domains, linfuncops


@dataclasses.dataclass(frozen=True)
class DirichletBoundaryCondition:
    boundary: domains.Domain
    values: Union[pn.randprocs.RandomProcess, pn.randvars.RandomVariable]


@dataclasses.dataclass(frozen=True)
class BoundaryValueProblem:
    domain: domains.Domain
    diffop: linfuncops.LinearDifferentialOperator
    rhs: Union[pn.Function, pn.randprocs.RandomProcess]
    boundary_conditions: Sequence[DirichletBoundaryCondition]
    solution: Optional[pn.Function] = None
