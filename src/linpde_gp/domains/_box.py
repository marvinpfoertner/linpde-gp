from __future__ import annotations

from collections.abc import Iterator, Sequence
import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, DTypeLike, FloatLike, ScalarType

from ._domain import Domain
from ._point import PointSet


class Interval(Domain, Sequence[np.ndarray]):
    def __init__(
        self,
        lower_bound: FloatLike,
        upper_bound: FloatLike,
        dtype: DTypeLike = np.double,
    ) -> None:
        self._lower_bound = pn.utils.as_numpy_scalar(lower_bound, dtype=dtype)
        self._upper_bound = pn.utils.as_numpy_scalar(upper_bound, dtype=dtype)

        if self._lower_bound > self._upper_bound:
            raise ValueError("The lower bound must not be larger than the upper bound")

        assert self._lower_bound.dtype == self._upper_bound.dtype

        super().__init__(shape=(), dtype=self._lower_bound.dtype)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx in (0, -2):
            return self._lower_bound

        if idx in (1, -1):
            return self._upper_bound

        return KeyError(f"Index {idx} is out of range")

    def __iter__(self):
        yield self._lower_bound
        yield self._upper_bound

    @functools.cached_property
    def boundary(self) -> PointSet:
        return PointSet((self._lower_bound, self._upper_bound))

    @property
    def volume(self) -> ScalarType:
        return self._upper_bound - self._lower_bound

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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Interval) and tuple(self) == tuple(other)


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
                "The lower bounds must not be smaller than the upper bounds."
            )

        self._collapsed = self._bounds[:, 0] == self._bounds[:, 1]
        self._interior_idcs = np.nonzero(~self._collapsed)[0]

        super().__init__(
            shape=self._bounds.shape[:-1],
            dtype=self._bounds.dtype,
        )

    @property
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

    @functools.cached_property
    def volume(self) -> ScalarType:
        return np.prod(self._bounds[..., 1] - self._bounds[..., 0])

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

    def __eq__(self, other) -> bool:
        return isinstance(other, Box) and np.all(self.bounds == other.bounds)
