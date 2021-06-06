import abc
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import probnum as pn
from probnum.type import (
    DTypeArgType,
    FloatArgType,
    IntArgType,
    RandomStateArgType,
    ShapeArgType,
)


class Function(pn.randprocs.RandomProcess):
    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        input_dim: IntArgType,
        output_dim: IntArgType,
        dtype: DTypeArgType,
    ):
        self._fn = fn

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype,
        )

    def __call__(self, args: np.ndarray) -> pn.randvars.Constant:
        return pn.randvars.Constant(support=self._fn(args))

    def mean(self, args: np.ndarray) -> np.ndarray:
        return self._fn(args)

    def cov(self, args0: np.ndarray, args1: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def push_forward(
        self,
        args: np.ndarray,
        base_measure: Type[pn.randvars.RandomVariable],
        sample: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError()

    def _sample_at_input(
        self, args: np.ndarray, size: ShapeArgType, random_state: RandomStateArgType
    ) -> np.ndarray:
        return pn.randvars.Constant(
            support=self._fn(args), random_state=random_state
        ).sample(size=size)


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
        self._grid = np.linspace(*self._domain, len(self))

        super().__init__(size=self._num_elements + 2)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def __len__(self) -> int:
        return self._num_elements + 2

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
            return lambda x: np.interp(x, self._grid, coords)

        # Interpret as random variable
        coords = pn.randvars.asrandvar(coords)

        if isinstance(coords, pn.randvars.Constant):
            return Function(
                self.coords2fn(coords.support),
                input_dim=1,
                output_dim=1,
                dtype=coords.dtype,
            )

        raise TypeError("Unsupported type of random variable for argument `coords`")

class FourierBasis(Basis):
    pass  # TODO
