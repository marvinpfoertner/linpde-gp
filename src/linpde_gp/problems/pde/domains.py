import abc
import functools
import operator
from collections.abc import Sequence
from typing import Union

import numpy as np
import probnum as pn
from numpy import typing as npt
from probnum.typing import ArrayLike, FloatArgType, ShapeArgType, ShapeType

DomainLike = Union[
    "Domain", tuple[FloatArgType, FloatArgType], list[FloatArgType], ArrayLike
]


def asdomain(arg: DomainLike) -> "Domain":
    if isinstance(arg, Domain):
        return arg
    elif isinstance(arg, (tuple, list)) and len(arg) == 2:
        return Interval(float(arg[0]), float(arg[1]))
    else:
        return Point(arg)


class Domain(abc.ABC):
    def __init__(self, shape: ShapeArgType, dtype: npt.DTypeLike) -> None:
        self._shape = pn.utils.as_shape(shape)
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def ndims(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    @abc.abstractmethod
    def boundary(self) -> Sequence["Domain"]:
        pass


class Interval(Domain, Sequence):
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

        super().__init__(shape=(), dtype=self._lower_bound.dtype)

    def __len__(self) -> int:
        return 2

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

    @functools.cached_property
    def boundary(self) -> Sequence["Point"]:
        return (Point(self._lower_bound), Point(self._upper_bound))

    def __repr__(self) -> str:
        return (
            f"<Interval {[self._lower_bound, self._upper_bound]} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def __array__(self) -> np.ndarray:
        return np.hstack((self._lower_bound, self._upper_bound))


class Point(Domain):
    def __init__(self, point: ArrayLike) -> None:
        self._point = np.asarray(point)

        super().__init__(shape=self._point.shape, dtype=self._point.dtype)

    @functools.cached_property
    def boundary(self) -> Sequence["Domain"]:
        return ()

    def __repr__(self) -> str:
        return (
            f"<Point {str(self._point)} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        return np.array(self._point, dtype=dtype, copy=True)

    def __float__(self):
        if self.ndims > 1:
            raise NotImplementedError()

        return float(self._point)
