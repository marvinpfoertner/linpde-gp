from __future__ import annotations

from collections.abc import Sequence
import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, DTypeLike, FloatLike

from ._domain import Domain
from ._point import Point


class Interval(Domain, Sequence):
    def __init__(
        self,
        lower_bound: FloatLike,
        upper_bound: FloatLike,
        dtype: DTypeLike = np.double,
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
