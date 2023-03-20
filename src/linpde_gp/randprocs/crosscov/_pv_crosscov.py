from __future__ import annotations

import abc
import functools
import operator
from typing import List, Type

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike, ShapeType


class ProcessVectorCrossCovariance(abc.ABC):
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

    def __call__(self, x: ArrayLike) -> np.ndarray:
        x = np.asarray(x)

        # Shape checking
        if x.shape[x.ndim - self.randproc_input_ndim :] != self.randproc_input_shape:
            err_msg = (
                "The shape of the input array must match the `randproc_input_shape` "
                f"`{self.randproc_input_shape}` of the function along its last "
                f"dimensions, but an array with shape `{x.shape}` was given."
            )

            raise ValueError(err_msg)

        batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

        fx = self._evaluate(x)

        if self.reverse:
            assert fx.shape == (
                self.randvar_shape + batch_shape + self.randproc_output_shape
            )
        else:
            assert fx.shape == (
                batch_shape + self.randproc_output_shape + self.randvar_shape
            )

        return fx

    @abc.abstractmethod
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    def jax(self, x: ArrayLike) -> jnp.ndarray:
        x = jnp.asarray(x)

        if x.shape[x.ndim - self.randproc_input_ndim :] != self.randproc_input_shape:
            err_msg = (
                "The shape of the input array must match the `randproc_input_shape` "
                f"`{self.randproc_input_shape}` of the function along its last "
                f"dimensions, but an array with shape `{x.shape}` was given."
            )

            raise ValueError(err_msg)

        batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]

        fx = self._evaluate_jax(x)

        if self.reverse:
            assert fx.shape == (
                self.randvar_shape + batch_shape + self.randproc_output_shape
            )
        else:
            assert fx.shape == (
                batch_shape + self.randproc_output_shape + self.randvar_shape
            )

        return fx

    @abc.abstractmethod
    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        raise NotImplementedError()

    def evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        x = np.asarray(x)

        # Shape checking
        if x.shape[x.ndim - self.randproc_input_ndim :] != self.randproc_input_shape:
            err_msg = (
                "The shape of the input array must match the `randproc_input_shape` "
                f"`{self.randproc_input_shape}` of the function along its last "
                f"dimensions, but an array with shape `{x.shape}` was given."
            )

            raise ValueError(err_msg)

        try:
            return self._evaluate_linop(x)
        except NotImplementedError:
            batch_shape = x.shape[: x.ndim - self.randproc_input_ndim]
            batch_size = int(np.prod(batch_shape))
            randproc_output_size = int(np.prod(self.randproc_output_shape))
            res = self(x)
            if self.reverse:
                res = _move_shape_blocks(
                    res,
                    [self.randvar_shape, batch_shape, self.randproc_output_shape],
                    [0, 2, 1],
                )
                target_shape = (self.randvar_size, randproc_output_size * batch_size)
            else:
                res = _move_shape_blocks(
                    res,
                    [batch_shape, self.randproc_output_shape, self.randvar_shape],
                    [1, 0, 2],
                )
                target_shape = (randproc_output_size * batch_size, self.randvar_size)
            res = np.reshape(res, target_shape, order="C")
            return pn.linops.Matrix(res)

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


def _move_shape_blocks(
    x: np.ndarray, input_shapes: List[ShapeType], new_positions: List[int]
):
    shape_indices = []
    cur_offset = 0
    for input_shape in input_shapes:
        shape_indices.append(tuple(range(cur_offset, cur_offset + len(input_shape))))
        cur_offset += len(input_shape)
    output_shape_indices = [shape_indices[i] for i in new_positions]
    return np.transpose(x, functools.reduce(operator.add, output_shape_indices))
