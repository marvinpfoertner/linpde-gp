from collections.abc import Iterator, Sequence

import numpy as np
from probnum.typing import ArrayLike

from ._domain import Domain
from ._interval import Interval


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
