from typing import Callable, Union

import numpy as np
import probnum as pn
from probnum.typing import FloatArgType

from .. import domains, randprocs
from . import _basis


class FourierBasis(_basis.Basis):
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
