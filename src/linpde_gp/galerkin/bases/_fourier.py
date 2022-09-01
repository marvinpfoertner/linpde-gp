from typing import Callable, Union

import numpy as np
import probnum as pn

from linpde_gp import domains, randprocs
from linpde_gp.typing import DomainLike

from . import _basis


class FourierBasis(_basis.Basis):
    def __init__(
        self,
        domain: DomainLike,
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

    def __getitem__(self, idx: Union[int, slice, np.ndarray]) -> pn.functions.Function:
        l, r = self._domain

        if isinstance(idx, slice):
            idx = np.arange(
                idx.start if idx.start is not None else 0,
                idx.stop if idx.stop is not None else len(self),
                idx.step,
            )
        else:
            idx = np.asarray(idx)

        return pn.functions.LambdaFunction(
            lambda x: np.sin((idx + 1) * np.pi * (x[..., None] - l) / (r - l)),
            input_shape=(),
            output_shape=idx.shape,
        )

    def coords2fn(
        self,
        coords: Union[np.ndarray, pn.randvars.RandomVariable],
    ) -> Union[pn.functions.Function, pn.randprocs.RandomProcess]:
        if isinstance(coords, np.ndarray):
            return pn.functions.LambdaFunction(
                lambda x: self[:](x) @ coords,
                input_shape=(),
                output_shape=(),
            )

        # Interpret as random variable
        coords = pn.randvars.asrandvar(coords)

        if isinstance(coords, pn.randvars.Constant):
            return randprocs.DeterministicProcess(self.coords2fn(coords.support))

        if isinstance(coords, pn.randvars.Normal):
            return randprocs.ParametricGaussianProcess(
                weights=coords,
                feature_fn=self[:],
            )

        raise TypeError("Unsupported type of random variable for argument `coords`")
