import dataclasses
from typing import Callable, List, Optional, Union

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike

from ...typing import JaxLinearOperator
from . import domains


@dataclasses.dataclass(frozen=True)
class DirichletBoundaryCondition:
    boundary: domains.Domain
    values: Union[pn.randprocs.RandomProcess, pn.randvars.RandomVariable]


@dataclasses.dataclass(frozen=True)
class BoundaryValueProblem:
    diffop: JaxLinearOperator
    rhs: pn.randprocs.RandomProcess
    boundary_conditions: List[DirichletBoundaryCondition]
    solution: Optional[Callable[[ArrayLike], Union[np.floating]]] = None
