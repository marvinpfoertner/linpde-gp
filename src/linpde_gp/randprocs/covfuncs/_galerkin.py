import numpy as np
import probnum as pn

from linpde_gp import linfunctls

from ._parametric import ParametricCovarianceFunction


class GalerkinCovarianceFunction(pn.randprocs.covfuncs.CovarianceFunction):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        projection: linfunctls.LinearFunctional,
    ):
        self._covfunc = covfunc
        self._projection = projection

        self._kPa = self._projection(self._covfunc, argnum=1)
        self._PkPa = self._projection(self._kPa)

        self._kPaP = GalerkinCovarianceFunction._EmbeddedProcessVectorCrossCovariance(
            self._kPa, basis=self._projection.basis
        )
        self._PaPkPaP = ParametricCovarianceFunction(
            self._projection.basis, cov=self._PkPa
        )

        super().__init__(
            input_shape=self._covfunc.input_shape,
            output_shape_0=self._covfunc.output_shape_0,
            output_shape_1=self._covfunc.output_shape_1,
        )

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        PaPkPaP_x0_x1 = self._PaPkPaP(x0, x1)

        return (
            PaPkPaP_x0_x1
            + self._covfunc(x0, x1)
            - self._kPaP(x0, x1)
            - self._kPaP(x0 if x1 is None else x1, None if x1 is None else x0)
            + PaPkPaP_x0_x1
        )

    class _EmbeddedProcessVectorCrossCovariance(
        pn.randprocs.covfuncs.CovarianceFunction
    ):
        def __init__(self, pv_crosscov, basis: pn.functions.Function):
            self._pv_crosscov = pv_crosscov
            self._basis = basis

            super().__init__(
                input_shape=self._pv_crosscov.randproc_input_shape,
            )

        def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
            if x1 is None:
                x1 = x0

            if self._pv_crosscov.reverse:
                basis_x0 = self._basis(x0)
                pv_crosscov_x1 = self._pv_crosscov(x1).T

                return (basis_x0[..., None, :] @ pv_crosscov_x1[..., :, None])[
                    ..., 0, 0
                ]

            pv_crosscov_x0 = self._pv_crosscov(x0)
            basis_x1 = self._basis(x1)

            return (pv_crosscov_x0[..., None, :] @ basis_x1[..., :, None])[..., 0, 0]
