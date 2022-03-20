from multiprocessing.sharedctypes import Value
from typing import Sequence

import numpy as np

from linpde_gp.typing import DomainLike

from ._box import Box
from ._domain import Domain
from ._interval import Interval


def asdomain(arg: DomainLike) -> Domain:
    if isinstance(arg, Domain):
        return arg

    if isinstance(arg, Sequence) and len(arg) == 2:
        if all(np.ndim(bound) == 0 for bound in arg):
            return Interval(float(arg[0]), float(arg[1]))

        if np.ndim(arg[0]) == 1 and np.shape(arg[0]) == np.shape(arg[1]):
            return Box(np.stack(arg, axis=1))

    raise ValueError(f"Could not convert {arg} to a domain")
