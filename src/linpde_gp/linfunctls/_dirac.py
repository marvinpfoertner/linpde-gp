import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from . import _linfunctl


class DiracFunctional(_linfunctl.LinearFunctional):
    def __init__(
        self,
        domain_shape: ShapeLike,
        codomain_shape: ShapeLike,
        xs: ArrayLike,
    ) -> None:
        xs = np.asarray(xs)

        domain_ndim = len(domain_shape)
        batch_shape = xs.shape[: xs.ndim - domain_ndim]

        super().__init__(
            input_shapes=(domain_shape, codomain_shape),
            output_shape=batch_shape + codomain_shape,
        )

        assert xs.shape[xs.ndim - domain_ndim :] == self.input_domain_shape

        self._xs = xs

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.Function, /) -> np.ndarray:
        return f(self._xs)
