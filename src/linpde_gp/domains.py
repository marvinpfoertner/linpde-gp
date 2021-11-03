from typing import Optional, Tuple, Union

import numpy as np
import probnum as pn
from numpy import typing as npt
from probnum.typing import FloatArgType, IntArgType

DomainLike = Union["Domain", Tuple[FloatArgType, FloatArgType]]


def asdomain(arg: DomainLike) -> "Domain":
    if isinstance(arg, Domain):
        return arg
    elif isinstance(arg, tuple) and len(arg) == 2:
        return Interval(float(arg[0]), float(arg[1]))
    else:
        raise TypeError(f"`{arg}` could not be converted into a `Domain` object")


class Domain:
    def __init__(self, dim: IntArgType, dtype: npt.DTypeLike) -> None:
        self._dim = int(dim)
        self._dtype = np.dtype(dtype)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype


class Interval(Domain):
    def __init__(
        self,
        lower_bound: FloatArgType,
        upper_bound: FloatArgType,
        dtype: npt.DTypeLike = np.double,
    ) -> None:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                "The dtype of an interval must be a sub dtype of `np.floating`"
            )

        if lower_bound > upper_bound:
            raise ValueError("The lower bound must not be larger than the upper bound")

        self._lower_bound = pn.utils.as_numpy_scalar(lower_bound, dtype=dtype)
        self._upper_bound = pn.utils.as_numpy_scalar(upper_bound, dtype=dtype)

        assert self._lower_bound.dtype == self._upper_bound.dtype

        super().__init__(dim=1, dtype=self._lower_bound.dtype)

    def __getitem__(self, idx: int) -> np.floating:
        if idx in (0, -2):
            return self._lower_bound
        elif idx in (1, -1):
            return self._upper_bound
        else:
            return KeyError(f"Index {idx} is out of range")

    def __iter__(self):
        yield self._lower_bound
        yield self._upper_bound
