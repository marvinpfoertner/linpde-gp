import abc
from typing import Callable, Tuple

import numpy as np
import probnum as pn
from probnum.type import FloatArgType


class Basis(abc.ABC):
    def __init__(self, size):
        self._size = size

    def __len__(self) -> int:
        return self._size

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Callable[[FloatArgType], np.floating]:
        pass

    @abc.abstractmethod
    def coords2fn(self, coords: np.ndarray) -> Callable[[FloatArgType], np.floating]:
        pass


class FiniteElementBasis(Basis):
    def __init__(
        self,
        domain: Tuple[FloatArgType, FloatArgType],
        num_elements: int,
    ):
        self._domain = tuple(pn.utils.as_numpy_scalar(bound) for bound in domain)
        self._num_elements = num_elements
        self._grid = np.linspace(*self._domain, len(self))

        super().__init__(size=self._num_elements + 2)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def __len__(self) -> int:
        return self._num_elements + 2

    def __getitem__(self, idx: int) -> Callable[[FloatArgType], np.floating]:
        assert -len(self) <= idx < len(self)

        if idx < 0:
            idx += len(self)

        xs = self._grid[max(idx - 1, 0) : min(idx + 2, len(self))]

        if idx == 0:
            ys = (1.0, 0.0)
        elif idx == len(self) - 1:
            ys = (0.0, 1.0)
        else:
            ys = (0.0, 1.0, 0.0)

        ys = np.array(ys)

        return lambda x: np.interp(x, xs, ys)

    def coords2fn(self, coords: np.ndarray) -> Callable[[FloatArgType], np.floating]:
        return lambda x: np.interp(x, self._grid, coords)


class FourierBasis(Basis):
    pass  # TODO
