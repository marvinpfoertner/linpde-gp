from __future__ import annotations

from collections.abc import Sequence
from typing import Union

from probnum.typing import ArrayLike, FloatLike

import linpde_gp

DomainLike = Union[
    "linpde_gp.domains.Domain",
    tuple[FloatLike, FloatLike],  # -> Interval
    Sequence[FloatLike],  # -> Interval
    tuple[ArrayLike, ArrayLike],  # -> Box
    Sequence[ArrayLike],  # -> Box
]
