import dataclasses
from collections.abc import Callable, Sequence
from typing import Optional, Union

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
    domain: domains.Domain
    diffop: JaxLinearOperator
    rhs: pn.randprocs.RandomProcess
    boundary_conditions: Sequence[DirichletBoundaryCondition]
    solution: Optional[Callable[[ArrayLike], Union[np.floating, np.ndarray]]] = None
