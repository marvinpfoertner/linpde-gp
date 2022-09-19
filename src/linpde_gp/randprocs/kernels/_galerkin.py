import numpy as np
import probnum as pn

from linpde_gp import linfunctls

from ._parametric_kernel import ParametricKernel


class GalerkinKernel(pn.randprocs.kernels.Kernel):
    def __init__(
        self,
        kernel: pn.randprocs.kernels.Kernel,
        projection: linfunctls.LinearFunctional,
    ):
        self._kernel = kernel
        self._projection = projection

        self._kPa = self._projection(self._kernel, argnum=1)
        self._PkPa = self._projection(self._kPa)

        self._kPaP = GalerkinKernel._EmbeddedProcessVectorCrossCovariance(
            self._kPa, basis=self._projection.basis
        )
        self._PaPkPaP = ParametricKernel(self._projection.basis, cov=self._PkPa)

        super().__init__(
            input_shape=self._kernel.input_shape,
            output_shape=self._kernel.output_shape,
        )

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        PaPkPaP_x0_x1 = self._PaPkPaP(x0, x1)

        return (
            PaPkPaP_x0_x1
            + self._kernel(x0, x1)
            - self._kPaP(x0, x1)
            - self._kPaP(x0 if x1 is None else x1, None if x1 is None else x0)
            + PaPkPaP_x0_x1
        )

    class _EmbeddedProcessVectorCrossCovariance(pn.randprocs.kernels.Kernel):
        def __init__(self, pv_crosscov, basis: pn.functions.Function):
            self._pv_crosscov = pv_crosscov
            self._basis = basis

            super().__init__(
                input_shape=self._pv_crosscov.randproc_input_shape,
                output_shape=(),
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
