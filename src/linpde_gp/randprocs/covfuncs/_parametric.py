import numpy as np
import probnum as pn
from probnum.typing import LinearOperatorLike

from linpde_gp.randvars import Covariance


class ParametricCovarianceFunction(pn.randprocs.covfuncs.CovarianceFunction):
    def __init__(
        self,
        basis: pn.functions.Function,
        cov: Covariance,
    ):
        self._basis = basis
        self._cov = cov

        if self._cov.shape1 != self._basis.output_shape:
            raise ValueError()

        super().__init__(input_shape=self._basis.input_shape)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        phi_x0 = self._basis(x0)
        phi_x1 = phi_x0 if x1 is None else self._basis(x1)

        cov_phi_x1 = self._cov.linop(phi_x1, axis=-1)

        return (phi_x0[..., None, :] @ cov_phi_x1[..., :, None])[..., 0, 0]
