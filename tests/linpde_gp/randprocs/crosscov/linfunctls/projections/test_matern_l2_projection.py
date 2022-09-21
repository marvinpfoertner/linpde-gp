import numpy as np
import probnum as pn

from pytest_cases import fixture

import linpde_gp
from linpde_gp.linfunctls.projections.l2 import (
    L2Projection_UnivariateLinearInterpolationBasis,
)


@fixture
def kernel() -> pn.randprocs.kernels.Kernel:
    return pn.randprocs.kernels.Matern(
        input_shape=(),
        lengthscale=1.0,
        nu=1.5,
    )


@fixture
def kernel_lambda(
    kernel: pn.randprocs.kernels.Kernel,
) -> linpde_gp.randprocs.kernels.JaxLambdaKernel:
    return linpde_gp.randprocs.kernels.JaxLambdaKernel(
        kernel,
        input_shape=kernel.input_shape,
        output_shape=kernel.output_shape,
        vectorize=False,
    )


@fixture
def projection() -> L2Projection_UnivariateLinearInterpolationBasis:
    return linpde_gp.functions.bases.UnivariateLinearInterpolationBasis(
        np.linspace(-1.0, 1.0, 7),
        zero_boundary=False,
    ).l2_projection()


@fixture
def kPa(
    kernel: pn.randprocs.kernels.Kernel,
    projection: L2Projection_UnivariateLinearInterpolationBasis,
) -> linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance:
    return projection(kernel, argnum=1)


def test_kPa_eval(
    kPa: linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance,
    kernel_lambda: linpde_gp.randprocs.kernels.JaxLambdaKernel,
    projection: L2Projection_UnivariateLinearInterpolationBasis,
):
    bounds = projection.basis.grid[[0, -1]]
    xs = np.linspace(*bounds, 50)

    np.testing.assert_allclose(
        kPa(xs),
        projection(kernel_lambda, argnum=1)(xs),
    )
