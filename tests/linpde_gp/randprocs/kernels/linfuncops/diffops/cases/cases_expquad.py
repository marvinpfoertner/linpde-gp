import numpy as np
from probnum.typing import ShapeType

from pytest_cases import parametrize

from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import covfuncs
from linpde_gp.randprocs.covfuncs.linfuncops import diffops as covfuncs_diffops

from ._test_case import CovarianceFunctionDiffOpTestCase

input_shapes = ((), (1,), (3,))


@parametrize(input_shape=input_shapes)
def case_expquad_identity_directional_derivative(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.ExpQuad(input_shape),
        L0=None,
        L1=diffops.DirectionalDerivative(direction),
        expected_type=covfuncs_diffops.ExpQuad_Identity_DirectionalDerivative,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_directional_derivative_identity(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.ExpQuad(input_shape),
        L0=diffops.DirectionalDerivative(direction),
        L1=None,
        expected_type=covfuncs_diffops.ExpQuad_Identity_DirectionalDerivative,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_directional_derivative_directional_derivative(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    k = covfuncs.ExpQuad(input_shape)

    rng = np.random.default_rng(52469753628)

    # pylint: disable=duplicate-code
    direction0 = rng.standard_normal(size=input_shape)
    direction1 = rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=k,
        L0=diffops.DirectionalDerivative(direction0),
        L1=diffops.DirectionalDerivative(direction1),
        expected_type=covfuncs_diffops.ExpQuad_DirectionalDerivative_DirectionalDerivative,  # pylint: disable=line-too-long
    )


@parametrize(input_shape=input_shapes)
def case_expquad_identity_weighted_laplacian(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(524390)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.ExpQuad(input_shape),
        L0=None,
        L1=diffops.WeightedLaplacian(weights),
        expected_type=covfuncs_diffops.ExpQuad_Identity_WeightedLaplacian,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_weighted_laplacian_identity(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(2309823372)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.ExpQuad(input_shape),
        L0=diffops.WeightedLaplacian(weights),
        L1=None,
        expected_type=covfuncs_diffops.ExpQuad_Identity_WeightedLaplacian,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_weighted_laplacian_weighted_laplacian(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(235890)

    weights0 = 2.0 * rng.standard_normal(size=input_shape)
    weights1 = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.ExpQuad(input_shape),
        L0=diffops.WeightedLaplacian(weights0),
        L1=diffops.WeightedLaplacian(weights1),
        expected_type=covfuncs_diffops.ExpQuad_WeightedLaplacian_WeightedLaplacian,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_directional_derivative_weighted_laplacian(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)
    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.ExpQuad(input_shape),
        L0=diffops.DirectionalDerivative(direction),
        L1=diffops.WeightedLaplacian(weights),
        expected_type=covfuncs_diffops.ExpQuad_DirectionalDerivative_WeightedLaplacian,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_weighted_laplacian_directional_derivative(
    input_shape: ShapeType,
) -> CovarianceFunctionDiffOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)
    weights = 2.0 * rng.standard_normal(size=input_shape)

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.ExpQuad(input_shape),
        L0=diffops.WeightedLaplacian(weights),
        L1=diffops.DirectionalDerivative(direction),
        expected_type=covfuncs_diffops.ExpQuad_DirectionalDerivative_WeightedLaplacian,
    )
