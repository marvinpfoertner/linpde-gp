import functools

import numpy as np
import probnum as pn
import scipy.integrate
import scipy.linalg
import scipy.sparse

from linpde_gp import functions

from ... import _linfunctl


class L2Projection_UnivariateLinearInterpolationBasis(_linfunctl.LinearFunctional):
    def __init__(
        self,
        basis: functions.bases.UnivariateLinearInterpolationBasis,
        *,
        normalized: bool = True,
    ) -> None:
        self._basis = basis
        self._normalized = bool(normalized)

        super().__init__(
            input_shapes=((), ()),
            output_shape=basis.output_shape,
        )

    @property
    def basis(self) -> functions.bases.UnivariateLinearInterpolationBasis:
        return self._basis

    @functools.cached_property
    def normalizer(self) -> pn.linops.LinearOperator:
        if not self._normalized:
            return pn.linops.Identity(len(self._basis))

        x_im1 = self._basis.x_im1
        x_i = self._basis.x_i
        x_ip1 = self._basis.x_ip1

        diag = (x_ip1 - x_im1) / 3.0
        offdiag = (x_ip1[:-1] - x_i[:-1]) / 6.0

        if not self._basis.zero_boundary:
            diag[0] = (x_ip1[0] - x_i[0]) / 3.0
            diag[-1] = (x_i[-1] - x_im1[-1]) / 3.0

        return pn.linops.aslinop(
            scipy.sparse.diags(
                (offdiag, diag, offdiag),
                (-1, 0, 1),
            ).toarray()
        ).inv()

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register(pn.functions.Function)
    def _(self, f: pn.functions.Function, /) -> np.ndarray:
        res = np.array(
            [
                scipy.integrate.quad(
                    lambda x: self._basis.eval_elem(idx, x) * f(x),
                    *self._basis.support_bounds(idx),
                )[0]
                for idx in range(len(self._basis))
            ]
        )

        return self.normalizer(res, axis=-1)

    @__call__.register(functions.Constant)
    def _(self, f: functions.Constant, /) -> np.ndarray:
        res = (f.value / 2.0) * (self.basis.x_ip1 - self.basis.x_im1)

        if not self.basis.zero_boundary:
            res[0] = (f.value / 2.0) * (self.basis.x_ip1[0] - self.basis.x_i[0])
            res[-1] = (f.value / 2.0) * (self.basis.x_i[-1] - self.basis.x_im1[-1])

        return self.normalizer(res, axis=-1)
