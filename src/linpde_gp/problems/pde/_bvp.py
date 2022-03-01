from collections.abc import Callable, Sequence
import dataclasses
from typing import Optional, Union

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike

from . import domains
from ... import linfuncops


@dataclasses.dataclass(frozen=True)
class DirichletBoundaryCondition:
    boundary: domains.Domain
    values: Union[pn.randprocs.RandomProcess, pn.randvars.RandomVariable]


@dataclasses.dataclass(frozen=True)
class BoundaryValueProblem:
    domain: domains.Domain
    diffop: linfuncops.LinearFunctionOperator
    rhs: pn.randprocs.RandomProcess
    boundary_conditions: Sequence[DirichletBoundaryCondition]
    solution: Optional[Callable[[ArrayLike], Union[np.floating, np.ndarray]]] = None
