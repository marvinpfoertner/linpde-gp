from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from ._cartesian_product import CartesianProduct
from ._interval import Interval
from ._point import Point

if TYPE_CHECKING:
    from linpde_gp.randprocs.covfuncs import TensorProductGrid


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

    def uniform_grid(
        self, shape: ShapeLike, inset: ArrayLike = 0.0
    ) -> "TensorProductGrid":
        from linpde_gp.randprocs.covfuncs import (  # pylint: disable=import-outside-toplevel
            TensorProductGrid,
        )

        total_dim = len(self.bounds)

        interior_shape = pn.utils.as_shape(shape, ndim=len(self._interior_idcs))
        interior_inset = np.broadcast_to(inset, len(self._interior_idcs))
        total_shape = np.ones((total_dim,), dtype=int)
        total_insets = np.zeros((total_dim,), dtype=float)

        total_shape[self._interior_idcs] = interior_shape
        total_insets[self._interior_idcs] = interior_inset

        def get_grid(idx, num_points, inset):
            sub_domain = self[int(idx)]
            if isinstance(sub_domain, Interval):
                return sub_domain.uniform_grid(num_points, inset=inset)
            assert isinstance(sub_domain, Point)
            assert num_points == 1
            return np.array(sub_domain).reshape((1,))

        return TensorProductGrid(
            *(
                get_grid(idx, num_points, inset)
                for idx, (num_points, inset) in enumerate(
                    zip(total_shape, total_insets)
                )
            ),
            indexing="ij",
        )
