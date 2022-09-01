import abc
from typing import Callable, Union

import numpy as np
import probnum as pn


class Basis(abc.ABC):
    def __init__(self, size):
        self._size = size

    def __len__(self) -> int:
        return self._size

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> pn.functions.Function:
        pass

    @abc.abstractmethod
    def coords2fn(
        self,
        coords: Union[
            np.ndarray,
            pn.randvars.RandomVariable,  # TODO: Add probnum type RandomVariableLike
        ],
    ) -> Union[pn.functions.Function, pn.randprocs.RandomProcess]:
        pass

    def observation_operator(self, xs: np.ndarray):
        return self[:](xs)
