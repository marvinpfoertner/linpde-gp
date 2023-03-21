from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import probnum as pn
from probnum.typing import ArrayLike, FloatLike

DomainLike = Union[
    "linpde_gp.domains.Domain",
    tuple[FloatLike, FloatLike],  # -> Interval
    Sequence[FloatLike],  # -> Interval
    tuple[ArrayLike, ArrayLike],  # -> Box
    Sequence[ArrayLike],  # -> Box
]

RandomProcessLike = Union[
    pn.functions.Function,
    pn.randprocs.RandomProcess,
]

RandomVariableLike = Union[
    ArrayLike,
    pn.randvars.RandomVariable,
]
