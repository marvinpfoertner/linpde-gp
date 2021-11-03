import abc
from typing import Iterable, List, Set, Tuple, Union

import numpy as np
import probnum as pn
from numpy import typing as npt
from probnum.typing import ArrayLike, FloatArgType, IntArgType

DomainLike = Union[
    "Domain", Tuple[FloatArgType, FloatArgType], List[FloatArgType], Set[ArrayLike]
]


def asdomain(arg: DomainLike) -> "Domain":
    if isinstance(arg, Domain):
        return arg
    elif isinstance(arg, (tuple, list)) and len(arg) == 2:
        return Interval(float(arg[0]), float(arg[1]))
    elif isinstance(arg, set):
        return PointSet(arg)
    else:
        raise TypeError(f"`{arg}` could not be converted into a `Domain` object")


class Domain(abc.ABC):
    def __init__(self, dim: IntArgType, dtype: npt.DTypeLike) -> None:
        self._dim = int(dim)
        self._dtype = np.dtype(dtype)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    @abc.abstractmethod
    def boundary(self) -> "Domain":
        pass


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

    @property
    def boundary(self) -> "Domain":
        return PointSet(list(self))


class PointSet(Domain):
    def __init__(self, points: Iterable[ArrayLike]) -> None:
        self._points = list(points)

        assert all(point.ndim in (0, 1) for point in self._points)
        assert all(point.size == self._points[0].size for point in self._points)
        assert all(point.dtype == self._points[0].dtype for point in self._points)

        super().__init__(dim=self._points[0].size, dtype=self._points[0].dtype)

    def __len__(self) -> int:
        return len(self._points)

    def __getitem__(self, idx: int) -> Union[np.ndarray, np.floating]:
        self._points[idx]

    def __iter__(self):
        for point in self._points:
            yield point

    @property
    def boundary(self) -> "Domain":
        raise NotImplementedError()
