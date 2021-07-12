from typing import Callable, Union

import numpy as np
import probnum as pn
import scipy.interpolate
from probnum.typing import FloatArgType

from .. import domains, randprocs
from . import _basis


class ZeroBoundaryFiniteElementBasis(_basis.Basis):
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

    def observation_operator(self, xs: np.ndarray) -> np.ndarray:
        return self._observation_operator_fn(xs)


class FiniteElementBasis(_basis.Basis):
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

    def observation_operator(self, xs: np.ndarray) -> np.ndarray:
        return self._observation_operator_fn(xs)
