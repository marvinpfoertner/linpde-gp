import numpy as np
import probnum as pn

from pytest_cases import fixture

import linpde_gp
from linpde_gp.linfunctls.projections.l2 import (
    L2Projection_UnivariateLinearInterpolationBasis,
)


@fixture
def covfunc() -> pn.randprocs.covfuncs.CovarianceFunction:
    return pn.randprocs.covfuncs.Matern(
        input_shape=(),
        nu=1.5,
        lengthscales=1.0,
    )


@fixture
def covfunc_lambda(
    covfunc: pn.randprocs.covfuncs.CovarianceFunction,
) -> linpde_gp.randprocs.covfuncs.JaxLambdaCovarianceFunction:
    return linpde_gp.randprocs.covfuncs.JaxLambdaCovarianceFunction(
        covfunc,
        input_shape=covfunc.input_shape,
        output_shape_0=covfunc.output_shape_0,
        output_shape_1=covfunc.output_shape_1,
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
    covfunc: pn.randprocs.covfuncs.CovarianceFunction,
    projection: L2Projection_UnivariateLinearInterpolationBasis,
) -> linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance:
    return projection(covfunc, argnum=1)


def test_kPa_eval(
    kPa: linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance,
    covfunc_lambda: linpde_gp.randprocs.covfuncs.JaxLambdaCovarianceFunction,
    projection: L2Projection_UnivariateLinearInterpolationBasis,
):
    bounds = projection.basis.grid[[0, -1]]
    xs = np.linspace(*bounds, 50)

    np.testing.assert_allclose(
        kPa(xs),
        projection(covfunc_lambda, argnum=1)(xs),
    )
