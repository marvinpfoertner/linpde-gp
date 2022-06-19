from __future__ import annotations

from collections.abc import Sequence
import functools

import numpy as np
from probnum.typing import ArrayLike, ScalarType, ShapeLike

from ._domain import Domain


class Point(Domain):
    def __init__(self, point: ArrayLike) -> None:
        self._point = np.asarray(point)

        super().__init__(shape=self._point.shape, dtype=self._point.dtype)

    @functools.cached_property
    def boundary(self) -> Sequence[Domain]:
        return ()

    @property
    def volume(self) -> ScalarType:
        return np.zeros_like(self._point, shape=())

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

    def __eq__(self, other) -> bool:
        return isinstance(other, Point) and np.all(self._point == other._point)

    def uniform_grid(self, shape: ShapeLike, inset: float = 0.0) -> np.ndarray:
        raise NotImplementedError()


class PointSet(Domain, Sequence[Point]):
    def __init__(self, points: ArrayLike) -> None:
        self._points = np.asarray(points)

        if self._points.ndim < 1:
            raise ValueError()

        super().__init__(
            shape=self._points.shape[1:],
            dtype=self._points.dtype,
        )

    @functools.cached_property
    def boundary(self) -> Sequence[Domain]:
        return ()

    @property
    def volume(self) -> ScalarType:
        return np.zeros_like(self._points, shape=())

    def __repr__(self) -> str:
        return (
            "<PointSet {"
            f"{', '.join(str(point) for point in self._points)}"
            "} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def __getitem__(self, idx: int) -> Point:
        return Point(self._points[idx])

    def __len__(self) -> int:
        return len(self._points)

    def __array__(self, dtype: np.dtype = None) -> np.ndarray:
        return np.array(self._points, dtype=dtype, copy=True)

    def __contains__(self, item: ArrayLike) -> bool:
        arr = np.asarray(item, dtype=self.dtype)

        if arr.shape != self.shape:
            return False

        return np.any(
            np.all(
                self._points == arr,
                axis=tuple(range(1, len(self._points.shape))),
            )
        )

    def __eq__(self, other) -> bool:
        return isinstance(other, PointSet) and np.all(self._points == other._points)

    def uniform_grid(self, shape: ShapeLike, inset: float = 0.0) -> np.ndarray:
        raise NotImplementedError()
