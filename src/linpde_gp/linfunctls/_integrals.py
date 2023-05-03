import functools

import numpy as np
import probnum as pn
from probnum.typing import ShapeLike

from linpde_gp import domains, functions
from linpde_gp.typing import DomainLike

from . import _linfunctl


class LebesgueIntegral(_linfunctl.LinearFunctional):
    def __init__(
        self,
        input_domain: DomainLike,
        input_codomain_shape: ShapeLike = (),
    ) -> None:
        self._domain = domains.asdomain(input_domain)

        if not isinstance(self._domain, (domains.Interval, domains.Box)):
            raise TypeError("TODO")

        super().__init__(
            input_shapes=(self._domain.shape, input_codomain_shape),
            output_shape=input_codomain_shape,
        )

    @property
    def domain(self) -> domains.Domain:
        return self._domain

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /) -> np.ndarray:
        try:
            return super().__call__(f)
        except NotImplementedError as err:
            import scipy.integrate  # pylint: disable=import-outside-toplevel

            if self._output_shape != ():
                raise NotImplementedError from err

            match self._domain:
                case domains.Interval():
                    return scipy.integrate.quad(
                        f, a=self._domain[0], b=self._domain[1]
                    )[0]
                case domains.Box():
                    return scipy.integrate.nquad(
                        f,
                        ranges=[tuple(interval) for interval in self._domain],
                    )[0]

            raise NotImplementedError from err

    @__call__.register
    def _(self, f: functions.Constant, /) -> np.ndarray:
        return f.value * self._domain.volume
