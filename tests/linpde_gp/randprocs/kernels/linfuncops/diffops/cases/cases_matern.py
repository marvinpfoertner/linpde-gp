import numpy as np
from probnum.typing import ShapeType

import pytest
from pytest_cases import parametrize

from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import kernels
from linpde_gp.randprocs.kernels.linfuncops import diffops as kernels_diffops

from ._test_case import KernelDiffOpTestCase

input_shapes = ((), (1,), (3,))
nus_directional_derivative = (1.5, 2.5, 3.5, 4.5)
nus_laplacian = (2.5, 3.5, 4.5)


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_identity_directional_derivative(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=kernels.Matern(input_shape, nu=nu),
        L0=None,
        L1=diffops.DirectionalDerivative(direction),
        expected_type=kernels_diffops.HalfIntegerMatern_Identity_DirectionalDerivative,
    )


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_directional_derivative_identity(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=kernels.Matern(input_shape, nu=nu),
        L0=diffops.DirectionalDerivative(direction),
        L1=None,
        expected_type=kernels_diffops.HalfIntegerMatern_Identity_DirectionalDerivative,
    )


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_directional_derivative_directional_derivative(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    k = kernels.Matern(input_shape, nu=nu)

    if k.input_size > 1 and nu <= 1.5:
        pytest.skip("Not enough differentiability")

    rng = np.random.default_rng(413598)

    direction0 = rng.standard_normal(size=input_shape)
    direction1 = rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=k,
        L0=diffops.DirectionalDerivative(direction0),
        L1=diffops.DirectionalDerivative(direction1),
        expected_type=(
            kernels_diffops.HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
            if k.input_size > 1
            else kernels_diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative
        ),
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_identity_weighted_laplacian(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    rng = np.random.default_rng(5468907)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=kernels.Matern(input_shape, nu=nu),
        L0=None,
        L1=diffops.WeightedLaplacian(weights),
        expected_type=kernels_diffops.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_weighted_laplacian_identity(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    rng = np.random.default_rng(87905642)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=kernels.Matern(input_shape, nu=nu),
        L0=diffops.WeightedLaplacian(weights),
        L1=None,
        expected_type=kernels_diffops.UnivariateHalfIntegerMatern_Identity_WeightedLaplacian,
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_weighted_laplacian_weighted_laplacian(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    rng = np.random.default_rng(257834)

    weights0 = 2.0 * rng.standard_normal(size=input_shape)
    weights1 = 2.0 * rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=kernels.Matern(input_shape, nu=nu),
        L0=diffops.WeightedLaplacian(weights0),
        L1=diffops.WeightedLaplacian(weights1),
        expected_type=kernels_diffops.UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian,
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_directional_derivative_weighted_laplacian(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)
    weights = 2.0 * rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=kernels.Matern(input_shape, nu=nu),
        L0=diffops.DirectionalDerivative(direction),
        L1=diffops.WeightedLaplacian(weights),
        expected_type=kernels_diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian,
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_weighted_laplacian_directional_derivative(
    input_shape: ShapeType, nu: float
) -> KernelDiffOpTestCase:
    rng = np.random.default_rng(654890)

    direction = 2.0 * rng.standard_normal(size=input_shape)
    weights = 2.0 * rng.standard_normal(size=input_shape)

    return KernelDiffOpTestCase(
        k=kernels.Matern(input_shape, nu=nu),
        L0=diffops.WeightedLaplacian(weights),
        L1=diffops.DirectionalDerivative(direction),
        expected_type=kernels_diffops.UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian,
    )
