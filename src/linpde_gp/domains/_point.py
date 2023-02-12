from __future__ import annotations

from collections.abc import Sequence
import functools

import numpy as np
from probnum.typing import ArrayLike, ScalarType

from ._domain import Domain


class Point(Domain):
    def __init__(self, point: ArrayLike) -> None:
        self._point = np.asarray(point)

        super().__init__(shape=self._point.shape, dtype=self._point.dtype)

    @functools.cached_property
    def boundary(self) -> Sequence[Domain]:
        return (self,)

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
