import numpy as np
from probnum.typing import ShapeType

import pytest
from pytest_cases import parametrize

from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import covfuncs
from linpde_gp.randprocs.covfuncs.linfuncops import diffops as covfuncs_diffops

from ._test_case import CovarianceFunctionDiffOpTestCase

input_shapes = ((), (1,), (3,))
nus_directional_derivative = (1.5, 2.5, 3.5, 4.5)
nus_laplacian = (2.5, 3.5, 4.5)


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_identity_directional_derivative(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.Matern(input_shape, nu=nu),
        L0=None,
        L1=diffops.DirectionalDerivative(direction),
        expected_type=covfuncs_diffops.HalfIntegerMatern_Identity_DirectionalDerivative,
    )


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_directional_derivative_identity(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.Matern(input_shape, nu=nu),
        L0=diffops.DirectionalDerivative(direction),
        L1=None,
        expected_type=covfuncs_diffops.HalfIntegerMatern_Identity_DirectionalDerivative,
    )


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_directional_derivative_directional_derivative(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    k = covfuncs.Matern(input_shape, nu=nu)

    if k.input_size > 1 and nu <= 1.5:
        pytest.skip("Not enough differentiability")

    rng = np.random.default_rng(413598)

    direction0 = rng.standard_normal(size=input_shape)
    direction1 = rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=k,
        L0=diffops.DirectionalDerivative(direction0),
        L1=diffops.DirectionalDerivative(direction1),
        expected_type=(
            covfuncs_diffops.HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative  # pylint: disable=line-too-long
            if k.input_size > 1
            else covfuncs_diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative  # pylint: disable=line-too-long
        ),
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_identity_weighted_laplacian(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(5468907)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.Matern(input_shape, nu=nu),
        L0=None,
        L1=diffops.WeightedLaplacian(weights),
        expected_type=covfuncs_diffops.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,  # pylint: disable=line-too-long
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_weighted_laplacian_identity(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(87905642)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.Matern(input_shape, nu=nu),
        L0=diffops.WeightedLaplacian(weights),
        L1=None,
        expected_type=covfuncs_diffops.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,  # pylint: disable=line-too-long
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_weighted_laplacian_weighted_laplacian(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(257834)

    weights0 = 2.0 * rng.standard_normal(size=input_shape)
    weights1 = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.Matern(input_shape, nu=nu),
        L0=diffops.WeightedLaplacian(weights0),
        L1=diffops.WeightedLaplacian(weights1),
        expected_type=covfuncs_diffops.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian,  # pylint: disable=line-too-long
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_directional_derivative_weighted_laplacian(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)
    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.Matern(input_shape, nu=nu),
        L0=diffops.DirectionalDerivative(direction),
        L1=diffops.WeightedLaplacian(weights),
        expected_type=covfuncs_diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian,  # pylint: disable=line-too-long
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_weighted_laplacian_directional_derivative(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(654890)

    direction = 2.0 * rng.standard_normal(size=input_shape)
    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.Matern(input_shape, nu=nu),
        L0=diffops.WeightedLaplacian(weights),
        L1=diffops.DirectionalDerivative(direction),
        expected_type=covfuncs_diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian,  # pylint: disable=line-too-long
    )


@parametrize(
    input_shape=((),),
    nu=nus_directional_derivative,
)
def case_matern_deriv_deriv(
    input_shape: ShapeType, nu: float
) -> CovarianceFunctionDiffOpTestCase:
    k = covfuncs.Matern(input_shape, nu=nu, lengthscales=2.0)

    return CovarianceFunctionDiffOpTestCase(
        k=k,
        L0=diffops.Derivative(3),
        L1=diffops.Derivative(3),
        expected_type=covfuncs_diffops.UnivariateHalfIntegerMatern_Derivative_Derivative,
    )
