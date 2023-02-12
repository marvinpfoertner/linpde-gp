from __future__ import annotations

import abc
from collections.abc import Sequence
import functools
import operator

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, DTypeLike, ScalarType, ShapeLike, ShapeType


class Domain(abc.ABC):
    def __init__(self, shape: ShapeLike, dtype: DTypeLike) -> None:
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(
                "The dtype of an interval must be a sub dtype of `np.floating`"
            )

        self._shape = pn.utils.as_shape(shape)
        self._dtype = np.dtype(dtype)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def ndims(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    @abc.abstractmethod
    def boundary(self) -> Sequence[Domain]:
        pass

    @property
    @abc.abstractmethod
    def volume(self) -> ScalarType:
        pass

    @abc.abstractmethod
    def __contains__(self, item: ArrayLike) -> bool:
        pass

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        pass
