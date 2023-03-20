import functools

import numpy as np
import probnum as pn
from probnum.typing import LinearOperatorLike


class ConcatenatedLinearOperator(pn.linops.LinearOperator):
    def __init__(self, linops: tuple[LinearOperatorLike], axis: int):
        linops = tuple(pn.linops.aslinop(linop) for linop in linops)
        if len(linops) < 1:
            raise ValueError("At least one linear operator must be given.")
        if axis not in [0, 1, -1, -2]:
            raise ValueError(f"axis is {axis}, expected one of 0, 1, -1, -2.")
        if axis < 0:
            axis += 2
        dtype = functools.reduce(np.promote_types, (linop.dtype for linop in linops))
        if axis == 0:
            shape_0 = sum(linop.shape[0] for linop in linops)
            assert all(linop.shape[1] == linops[0].shape[1] for linop in linops)
            shape_1 = linops[0].shape[1]
        elif axis == 1:
            assert all(linop.shape[0] == linops[0].shape[0] for linop in linops)
            shape_0 = linops[0].shape[0]
            shape_1 = sum(linop.shape[1] for linop in linops)
            self._split_indices = np.array(
                tuple(linop.shape[1] for linop in linops)
            ).cumsum()[:-1]
        super().__init__((shape_0, shape_1), dtype)
        self._linops = linops
        self._axis = axis

    @property
    def linops(self) -> tuple[pn.linops.LinearOperator]:
        return self._linops

    @property
    def axis(self) -> int:
        return self._axis

    def _split_input(self, x: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            raise ValueError(
                "Input does not need to be split when concatenating along axis 0."
            )
        return np.split(x, self._split_indices, axis=-2)

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        if self.axis == 0:
            return np.concatenate(tuple(linop @ x for linop in self.linops), axis=-2)
        # axis == 1
        return np.sum(
            tuple(
                linop @ cur_x for linop, cur_x in zip(self.linops, self._split_input(x))
            ),
            axis=0,
        )

    def _transpose(self) -> pn.linops.LinearOperator:
        return ConcatenatedLinearOperator(
            tuple(linop.T for linop in self.linops), 1 - self.axis
        )

    def _todense(self) -> np.ndarray:
        return np.concatenate(
            tuple(linop.todense() for linop in self.linops), axis=self.axis
        )
