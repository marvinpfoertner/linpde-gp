from __future__ import annotations

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from ._cartesian_product import CartesianProduct
from ._interval import Interval
from ._point import Point


class Box(CartesianProduct):
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
            *(
                Interval(lower_bound, upper_bound, dtype=self._bounds.dtype)
                if lower_bound != upper_bound
                else Point(lower_bound)
                for (lower_bound, upper_bound) in self._bounds
            )
        )

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    def __getitem__(self, idx) -> Interval | Box:
        if isinstance(idx, int):
            return super().__getitem__(idx)

        return Box(self._bounds[idx, :])

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

    def uniform_grid(self, shape: ShapeLike, inset: ArrayLike = 0.0) -> np.ndarray:
        shape = pn.utils.as_shape(shape, ndim=len(self._interior_idcs))
        insets = np.broadcast_to(inset, len(self._interior_idcs))

        noncollapsed_grids = np.meshgrid(
            *(
                self[int(idx)].uniform_grid(num_points, inset=inset)
                for idx, num_points, inset in zip(self._interior_idcs, shape, insets)
            )
        )

        grids = [
            np.broadcast_to(lower_bound, noncollapsed_grids[0].shape)
            for lower_bound in self.bounds[..., 0]
        ]

        for idx, noncollapsed_grid in zip(self._interior_idcs, noncollapsed_grids):
            grids[idx] = noncollapsed_grid

        return np.stack(grids, axis=-1)
