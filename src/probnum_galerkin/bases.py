import abc
from typing import Callable, Union

import numpy as np
import probnum as pn
import scipy.interpolate
from probnum.type import FloatArgType

from . import domains, randprocs


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

class ZeroBoundaryFiniteElementBasis(Basis):
    def __init__(
        self,
        domain: domains.DomainLike,
        num_elements: int,
    ):
        super().__init__(size=num_elements)

        self._domain = domains.asdomain(domain)
        self._grid = np.linspace(*self._domain, len(self) + 2)

        # TODO: Implement this as a `LinearOperator`
        self._observation_operator_fn = scipy.interpolate.interp1d(
            x=self._grid,
            y=np.eye(len(self) + 2, len(self), k=-1),
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def __getitem__(
        self, idx: int
    ) -> Callable[[Union[FloatArgType, np.ndarray]], np.floating]:
        assert -len(self) <= idx < len(self)

        if idx < 0:
            idx += len(self)

        return scipy.interpolate.interp1d(
            x=self._grid[idx : idx + 3],
            y=np.array((0.0, 1.0, 0.0)),
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )

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
                y=np.hstack((0.0, coords, 0.0)),
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
                linop_fn=self._observation_operator_fn,
                mean=self.coords2fn(coords.mean),
            )

        raise TypeError("Unsupported type of random variable for argument `coords`")

class FiniteElementBasis(Basis):
    def __init__(
        self,
        domain: domains.DomainLike,
        num_elements: int,
    ):
        super().__init__(size=num_elements + 2)

        self._domain = domains.asdomain(domain)
        self._grid = np.linspace(*self._domain, len(self))

        # TODO: Implement this as a `LinearOperator`
        self._observation_operator_fn = scipy.interpolate.interp1d(
            x=self._grid,
            y=np.eye(len(self)),
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )

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
                linop_fn=self._observation_operator_fn,
                mean=self.coords2fn(coords.mean),
            )

        raise TypeError("Unsupported type of random variable for argument `coords`")


class FourierBasis(Basis):
    def __init__(
        self,
        domain: domains.DomainLike,
        num_frequencies: int,
        const: bool = False,
        sin: bool = True,
        cos: bool = False,
    ):
        self._domain = domains.asdomain(domain)
        self._num_frequencies = num_frequencies

        assert not const
        assert sin
        assert not cos

        super().__init__(size=self._num_frequencies)

    def __getitem__(
        self, idx: Union[int, slice, np.ndarray]
    ) -> Callable[[Union[FloatArgType, np.ndarray]], np.floating]:
        l, r = self._domain

        if isinstance(idx, slice):
            idx = np.arange(
                idx.start if idx.start is not None else 0,
                idx.stop if idx.stop is not None else len(self),
                idx.step,
            )

        return lambda x: np.sin((idx + 1) * np.pi * (x - l) / (r - l))

    def coords2fn(
        self,
        coords: Union[np.ndarray, pn.randvars.RandomVariable],
    ) -> Union[
        Callable[[Union[FloatArgType, np.ndarray]], np.floating],
        pn.randprocs.RandomProcess,
    ]:
        if isinstance(coords, np.ndarray):
            return lambda x: self[:](x[:, None]) @ coords

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
                linop_fn=lambda x: self[:](x[:, None]),
            )

        raise TypeError("Unsupported type of random variable for argument `coords`")
