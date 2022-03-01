from __future__ import annotations

import abc
from collections.abc import Iterator, Sequence
import functools
import operator
from typing import Union

import numpy as np
from numpy import typing as npt
import probnum as pn
from probnum.typing import ArrayLike, FloatLike, ShapeLike, ShapeType

DomainLike = Union["Domain", tuple[FloatLike, FloatLike], list[FloatLike], ArrayLike]


def asdomain(arg: DomainLike) -> Domain:
    if isinstance(arg, Domain):
        return arg
    elif isinstance(arg, (tuple, list)) and len(arg) == 2:
        return Interval(float(arg[0]), float(arg[1]))
    else:
        return Point(arg)


class Domain(abc.ABC):
    def __init__(self, shape: ShapeLike, dtype: npt.DTypeLike) -> None:
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
    def boundary(self) -> Sequence[Domain]:
        pass

    @abc.abstractmethod
    def __contains__(self, item: ArrayLike) -> bool:
        pass


class Box(Domain):
    def __init__(self, bounds: ArrayLike) -> None:
        self._bounds = np.array(bounds, copy=True)
        self._bounds.flags.writeable = False

        if not (self._bounds.ndim == 2 and self._bounds.shape[-1] == 2):
            raise ValueError(
                f"`bounds` must have shape (D, 2), but an object of shape "
                f"{self._bounds.shape} was given."
            )

        if not np.issubdtype(self._bounds.dtype, np.floating):
            raise TypeError(
                f"The dtype of `bounds` must be a sub dtype of `np.floating`, but "
                f"{self._bounds.dtype} was given."
            )

        if not np.all(self._bounds[:, 0] <= self._bounds[:, 1]):
            raise ValueError(
                f"The left boundaries of the bounds must not be smaller than the "
                f"right boundaries."
            )

        self._collapsed = self._bounds[:, 0] == self._bounds[:, 1]
        self._interior_idcs = np.nonzero(~self._collapsed)[0]

        super().__init__(
            shape=self._bounds.shape[:-1],
            dtype=self._bounds.dtype,
        )

    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def boundary(self) -> Sequence[Domain]:
        res = []

        for interior_idx in self._interior_idcs:
            for bound_idx in (0, 1):
                boundary_bounds = self._bounds.copy()
                boundary_bounds[interior_idx, :] = 2 * (
                    self._bounds[interior_idx, bound_idx],
                )

                res.append(Box(boundary_bounds))

        return tuple(res)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx) -> np.ndarray:
        if isinstance(idx, int):
            return Interval(*self._bounds[idx, :], dtype=self.dtype)

        return Box(self._bounds[idx, :])

    def __iter__(self) -> Iterator[Interval]:
        for idx in range(self.shape[0]):
            yield self[idx]

    def __array__(self) -> np.ndarray:
        return self._bounds

    def __repr__(self) -> str:
        interval_strs = [str(list(self._bounds[idx, :])) for idx in range(len(self))]

        return (
            f"<Box {' x '.join(interval_strs)} with "
            f"shape={self.shape} and "
            f"dtype={self.dtype}>"
        )

    def __contains__(self, item: ArrayLike) -> bool:
        arr = np.asarray(item, dtype=self.dtype)

        if arr.shape != self.shape:
            return False

        return np.all((self._bounds[:, 0] <= arr) & (arr <= self._bounds[:, 1]))


class Interval(Domain, Sequence):
    def __init__(
        self,
        lower_bound: FloatLike,
        upper_bound: FloatLike,
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
    def boundary(self) -> Sequence[Point]:
        return (Point(self._lower_bound), Point(self._upper_bound))

    def __repr__(self) -> str:
        return (
            f"<Interval {[self._lower_bound, self._upper_bound]} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def __array__(self) -> np.ndarray:
        return np.hstack((self._lower_bound, self._upper_bound))

    def __contains__(self, item: ArrayLike) -> bool:
        arr = np.asarray(item, dtype=self.dtype)

        if arr.shape != self.shape:
            return False

        return self._lower_bound <= arr <= self._upper_bound


class Point(Domain):
    def __init__(self, point: ArrayLike) -> None:
        self._point = np.asarray(point)

        super().__init__(shape=self._point.shape, dtype=self._point.dtype)

    @functools.cached_property
    def boundary(self) -> Sequence[Domain]:
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

    def __contains__(self, item: ArrayLike) -> bool:
        arr = np.asarray(item, dtype=self.dtype)

        if arr.shape != self.shape:
            return False

        return np.all(self._point == arr)
