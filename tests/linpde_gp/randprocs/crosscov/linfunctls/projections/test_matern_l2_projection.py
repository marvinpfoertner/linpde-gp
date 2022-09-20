import numpy as np
import probnum as pn

from pytest_cases import fixture

import linpde_gp


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
def linfunctl() -> linpde_gp.linfunctls.LinearFunctional:
    return linpde_gp.functions.bases.UnivariateLinearInterpolationBasis(
        np.linspace(-1.0, 1.0, 7),
        zero_boundary=False,
    ).l2_projection()


@fixture
def kLa(
    kernel: pn.randprocs.kernels.Kernel,
    linfunctl: linpde_gp.linfunctls.LinearFunctional,
) -> linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance:
    return linfunctl(kernel, argnum=1)


def test_kLa_eval(
    kLa,
    kernel_lambda,
    linfunctl: linpde_gp.functions.bases.UnivariateLinearInterpolationBasis,
):
    bounds = linfunctl._phis._grid[[0, -1]]

    xs = np.linspace(*bounds, 50)

    np.testing.assert_allclose(
        kLa(xs),
        linfunctl(kernel_lambda, argnum=1)(xs),
    )
