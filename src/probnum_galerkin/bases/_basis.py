import abc
from typing import Callable, Union

import numpy as np
import probnum as pn
from probnum.type import FloatArgType


class Basis(abc.ABC):
    def __init__(self, size):
        self._size = size

    def __len__(self) -> int:
        return self._size

    @abc.abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Callable[[Union[FloatArgType, np.ndarray]], np.floating]:
        pass

    @abc.abstractmethod
    def coords2fn(
        self,
        coords: Union[
            np.ndarray,
            pn.randvars.RandomVariable,  # TODO: Add probnum type RandomVariableLike
        ],
    ) -> Union[
        Callable[[Union[FloatArgType, np.ndarray]], np.floating],
        pn.randprocs.RandomProcess,
    ]:
        pass

    def observation_operator(self, xs: np.ndarray):
        return self[:](xs)
