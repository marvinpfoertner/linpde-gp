import numpy as np
import probnum as pn

from linpde_gp.typing import ArrayLike


class UnivariateLinearInterpolationBasis(pn.functions.Function):
    def __init__(self, grid: ArrayLike, zero_boundary: bool = False) -> None:
        # Input Normalization
        grid = np.asarray(grid)
        zero_boundary = bool(zero_boundary)

        if grid.ndim != 1 or grid.size < 3:
            raise ValueError("TODO")

        if not zero_boundary:
            # Add sentinel grid points
            self._grid = np.concatenate(
                (
                    [grid[0] - (grid[1] - grid[0])],
                    grid,
                    [grid[-1] + (grid[-1] - grid[-2])],
                )
            )
        else:
            self._grid = grid

        self._zero_boundary = bool(zero_boundary)

        self._left_normalization_factors = 1.0 / (self.x_i - self.x_im1)
        self._right_normalization_factors = 1.0 / (self.x_ip1 - self.x_i)

        super().__init__(
            input_shape=(),
            output_shape=(self._grid.size - 2,),
        )

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def x_im1(self) -> np.ndarray:
        return self._grid[:-2]

    @property
    def x_i(self) -> np.ndarray:
        return self._grid[1:-1]

    @property
    def x_ip1(self) -> np.ndarray:
        return self._grid[2:]

    @property
    def zero_boundary(self) -> bool:
        return self._zero_boundary

    def _evaluate(self, x: ArrayLike) -> np.ndarray:
        res = np.maximum(
            0,
            np.where(
                x[..., None] < self.x_i,
                (x[..., None] - self.x_im1) * self._left_normalization_factors,
                (self.x_ip1 - x[..., None]) * self._right_normalization_factors,
            ),
        )

        if not self._zero_boundary:
            res[x < self._grid[1], 0] = 0.0
            res[x > self._grid[-2], -1] = 0.0

        return res

    def eval_elem(self, idx: int, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x)

        res = np.maximum(
            0,
            np.where(
                x < self.x_i[idx],
                (x - self.x_im1[idx]) * self._left_normalization_factors[idx],
                (self.x_ip1[idx] - x) * self._right_normalization_factors[idx],
            ),
        )

        if not self._zero_boundary:
            res = np.asarray(res)

            res[x < self._grid[1]] = 0.0
            res[x > self._grid[-2]] = 0.0

        return res

    def support_bounds(self, idx: int):
        assert -len(self) <= idx < len(self)

        if not self._zero_boundary:
            if idx in (0, -len(self)):
                return self.x_i[0], self.x_ip1[0]

            if idx in (len(self) - 1, -1):
                return self.x_im1[-1], self.x_i[-1]

        return self.x_im1[idx], self.x_ip1[idx]

    def __len__(self):
        return self._output_shape[0]

    def l2_projection(
        self, normalized: bool = True
    ) -> "linpde_gp.linfunctls.projections.l2.L2Projection_UnivariateLinearInterpolationBasis":
        from linpde_gp.linfunctls.projections.l2 import (
            L2Projection_UnivariateLinearInterpolationBasis,
        )

        return L2Projection_UnivariateLinearInterpolationBasis(
            self, normalized=normalized
        )
