import abc
from typing import Callable, Tuple, Union

import numpy as np
import probnum as pn
import scipy.interpolate
from probnum.type import FloatArgType

from . import randprocs


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


class FiniteElementBasis(Basis):
    def __init__(
        self,
        domain: Tuple[FloatArgType, FloatArgType],
        num_elements: int,
    ):
        self._domain = tuple(pn.utils.as_numpy_scalar(bound) for bound in domain)
        self._num_elements = num_elements
        self._grid = np.linspace(*self._domain, self._num_elements + 2)

        super().__init__(size=self._grid.size)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def __getitem__(
        self, idx: int
    ) -> Callable[[Union[FloatArgType, np.ndarray]], np.floating]:
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

    def coords2fn(
        self,
        coords: Union[np.ndarray, pn.randvars.RandomVariable],
    ) -> Union[
        Callable[[Union[FloatArgType, np.ndarray]], np.floating],
        pn.randprocs.RandomProcess,
    ]:
        if isinstance(coords, np.ndarray):
            return scipy.interpolate.interp1d(
                x=self._grid,
                y=coords,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=True,
            )

        # Interpret as random variable
        coords = pn.randvars.asrandvar(coords)

        if isinstance(coords, pn.randvars.Constant):
            return randprocs.Function(
                self.coords2fn(coords.support),
                input_dim=1,
                output_dim=1,
                dtype=coords.dtype,
            )
        elif isinstance(coords, pn.randvars.Normal):
            return randprocs.LinearTransformGaussianProcess(
                input_dim=1,
                base_rv=coords,
                # TODO: Implement this as a `LinearOperator`
                linop_fn=scipy.interpolate.interp1d(
                    x=self._grid,
                    y=np.eye(self._grid.shape[0]),
                    kind="linear",
                    axis=0,
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True,
                ),
                mean=self.coords2fn(coords.mean),
            )

        raise TypeError("Unsupported type of random variable for argument `coords`")

class FourierBasis(Basis):
    pass  # TODO
