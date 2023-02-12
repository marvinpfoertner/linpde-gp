from __future__ import annotations

from collections.abc import Sequence
import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, DTypeLike, FloatLike, ScalarType, ShapeLike

from ._domain import Domain
from ._point import Point


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
    def boundary(self) -> Point:
        return (Point(self._lower_bound), Point(self._upper_bound))

    @property
    def volume(self) -> ScalarType:
        return self._upper_bound - self._lower_bound

    def __repr__(self) -> str:
        return (
            f"<Interval {[self._lower_bound, self._upper_bound]} with "
            f"shape={self.shape} and "
            f"dtype={str(self.dtype)}>"
        )

    def __contains__(self, item: ArrayLike) -> bool:
        arr = np.asarray(item, dtype=self.dtype)

        if arr.shape != self.shape:
            return False

        return self._lower_bound <= arr <= self._upper_bound

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Interval) and tuple(self) == tuple(other)

    def uniform_grid(self, shape: ShapeLike, inset: ArrayLike = 0.0) -> np.ndarray:
        shape = pn.utils.as_shape(shape)
        inset = np.asarray(inset)

        assert len(shape) == 1 and inset.ndim == 0

        return np.linspace(
            self._lower_bound + inset,
            self._upper_bound - inset,
            shape[0],
        )
