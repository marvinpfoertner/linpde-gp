from jax import numpy as jnp
import numpy as np
import probnum as pn
import scipy.integrate
import scipy.linalg
import scipy.sparse

from linpde_gp.linfunctls.projections.l2 import (
    L2Projection_UnivariateLinearInterpolationBasis,
)

from .. import _pv_crosscov


class Kernel_L2Projection_UnivariateLinearInterpolationBasis(
    _pv_crosscov.ProcessVectorCrossCovariance
):
    def __init__(
        self,
        kernel: pn.randprocs.kernels.Kernel,
        proj: L2Projection_UnivariateLinearInterpolationBasis,
        reverse: bool = True,
    ):
        self._kernel = kernel
        self._projection = proj

        super().__init__(
            randproc_input_shape=(),
            randproc_output_shape=(),
            randvar_shape=self._projection.output_shape,
            reverse=reverse,
        )

    @property
    def kernel(self) -> pn.randprocs.kernels.Kernel:
        return self._kernel

    @property
    def projection(self) -> L2Projection_UnivariateLinearInterpolationBasis:
        return self._projection

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        basis = self._projection.basis

        res = np.vectorize(
            lambda x: np.array(
                [
                    scipy.integrate.quad(
                        lambda t: basis.eval_elem(idx, t) * self._kernel(x, t),
                        *basis.support_bounds(idx),
                    )[0]
                    for idx in range(len(basis))
                ]
            ),
            signature="()->(m)",
        )(x)

        res = self._projection.normalizer(res, axis=-1)

        if self._reverse:
            return np.moveaxis(res, -1, 0)

        return res

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register(  # pylint: disable=no-member
    Kernel_L2Projection_UnivariateLinearInterpolationBasis
)
def _(
    self, pv_crosscov: Kernel_L2Projection_UnivariateLinearInterpolationBasis, /
) -> np.ndarray:
    if pv_crosscov.reverse:
        proj0 = pv_crosscov.projection
        proj1: L2Projection_UnivariateLinearInterpolationBasis = self
    else:
        proj0: L2Projection_UnivariateLinearInterpolationBasis = self
        proj1 = pv_crosscov.projection

    basis0 = proj0.basis
    basis1 = proj1.basis

    def kernel_variance(idx0: int, idx1: int):
        res, _ = scipy.integrate.dblquad(
            lambda x1, x0: (
                basis0.eval_elem(idx0, x0)
                * pv_crosscov.kernel(x0, x1)
                * basis1.eval_elem(idx1, x1)
            ),
            *basis0.support_bounds(idx0),
            *basis1.support_bounds(idx1),
        )

        return res

    res = np.array(
        [
            [kernel_variance(idx0, idx1) for idx1 in range(len(basis1))]
            for idx0 in range(len(basis0))
        ]
    )

    res = proj1.normalizer(res, axis=-1)
    res = proj0.normalizer(res, axis=0)

    return res


class Matern32_L2Projection_UnivariateLinearInterpolationBasis(
    Kernel_L2Projection_UnivariateLinearInterpolationBasis
):
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        x = x[..., None]

        def _aux_int(a, b, t0, alpha):
            def _antideriv(t):
                return -(
                    (t - x + 2.0 / alpha) * (t - t0 + 1.0 / alpha) + 1 / alpha**2
                ) * np.exp(-alpha * (t - x))

            return _antideriv(b) - _antideriv(a)

        x_im1 = self.projection.basis.x_im1
        x_i = self.projection.basis.x_i
        x_ip1 = self.projection.basis.x_ip1

        alpha = np.sqrt(3) / self._kernel.lengthscale

        int_im1_i = (
            _aux_int(np.maximum(x_im1, x), np.maximum(x_i, x), x_im1, alpha)
            + _aux_int(np.minimum(x_im1, x), np.minimum(x_i, x), x_im1, -alpha)
        ) / (x_i - x_im1)

        int_i_ip1 = -(
            _aux_int(np.maximum(x_i, x), np.maximum(x_ip1, x), x_ip1, alpha)
            + _aux_int(np.minimum(x_i, x), np.minimum(x_ip1, x), x_ip1, -alpha)
        ) / (x_ip1 - x_i)

        res = int_im1_i + int_i_ip1

        if not self.projection.basis.zero_boundary:
            res[..., 0] = int_i_ip1[..., 0]
            res[..., -1] = int_im1_i[..., -1]

        res = self.projection.normalizer(res, axis=-1)

        if self._reverse:
            return np.moveaxis(res, -1, 0)

        return res
