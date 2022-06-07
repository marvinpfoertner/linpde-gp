from __future__ import annotations

import functools
import operator
from typing import Type

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike, ShapeType

from linpde_gp import functions


# TODO: This should not inherit from `Function`
class ProcessVectorCrossCovariance(functions.JaxFunction):
    def __init__(
        self,
        randproc_input_shape: ShapeLike,
        randproc_output_shape: ShapeLike,
        randvar_shape: ShapeLike,
        reverse: bool = True,
    ):
        self._randproc_input_shape = pn.utils.as_shape(randproc_input_shape)
        self._randproc_output_shape = pn.utils.as_shape(randproc_output_shape)
        self._randvar_shape = pn.utils.as_shape(randvar_shape)

        self._reverse = bool(reverse)

        super().__init__(
            input_shape=randproc_input_shape,
            output_shape=(
                self._randvar_shape + self._randproc_output_shape
                if self._reverse
                else self._randproc_output_shape + self._randvar_shape
            ),
        )

    @property
    def randproc_input_shape(self) -> ShapeType:
        return self._randproc_input_shape

    @property
    def randproc_input_ndim(self) -> int:
        return len(self._randproc_input_shape)

    @property
    def randproc_output_shape(self) -> ShapeType:
        return self._randproc_output_shape

    @property
    def randproc_output_ndim(self) -> int:
        return len(self._randproc_output_shape)

    @property
    def randvar_shape(self) -> ShapeType:
        return self._randvar_shape

    @property
    def randvar_ndim(self) -> int:
        return len(self._randvar_shape)

    @property
    def randvar_size(self) -> int:
        return functools.reduce(operator.mul, self._randvar_shape, 1)

    @property
    def reverse(self) -> bool:
        return self._reverse

    def __neg__(self):
        return -1.0 * self

    def __add__(self, other) -> ProcessVectorCrossCovariance | Type[NotImplemented]:
        if isinstance(other, ProcessVectorCrossCovariance):
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                SumProcessVectorCrossCovariance,
            )

            return SumProcessVectorCrossCovariance(self, other)

        return NotImplemented

    def __sub__(self, other) -> ProcessVectorCrossCovariance | Type[NotImplemented]:
        return self + (-other)

    def __rmul__(self, other) -> ProcessVectorCrossCovariance | Type[NotImplemented]:
        if np.ndim(other) == 0:
            from ._arithmetic import (  # pylint: disable=import-outside-toplevel
                ScaledProcessVectorCrossCovariance,
            )

            return ScaledProcessVectorCrossCovariance(self, scalar=other)

        return NotImplemented
