import numpy as np
from probnum.typing import ShapeType

from pytest_cases import parametrize

import linpde_gp

from ._test_case import KernelLinFuncOpTestCase

input_shapes = ((), (1,), (3,))


@parametrize(input_shape=input_shapes)
def case_expquad_identity_directional_derivative(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ExpQuad(input_shape),
        L0=None,
        L1=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
    )


@parametrize(input_shape=input_shapes)
def case_expquad_directional_derivative_identity(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ExpQuad(input_shape),
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
        L1=None,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_directional_derivative_directional_derivative(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    k = linpde_gp.randprocs.kernels.ExpQuad(input_shape)

    rng = np.random.default_rng(413598)

    direction0 = rng.standard_normal(size=input_shape)
    direction1 = rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=k,
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction0),
        L1=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction1),
    )


@parametrize(input_shape=input_shapes)
def case_expquad_identity_weighted_laplacian(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(524390)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ExpQuad(input_shape),
        L0=None,
        L1=linpde_gp.linfuncops.diffops.WeightedLaplacian(weights),
    )


@parametrize(input_shape=input_shapes)
def case_expquad_weighted_laplacian_identity(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(2309823372)

    weights = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ExpQuad(input_shape),
        L0=linpde_gp.linfuncops.diffops.WeightedLaplacian(weights),
        L1=None,
    )


@parametrize(input_shape=input_shapes)
def case_expquad_weighted_laplacian_weighted_laplacian(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(235890)

    weights0 = 2.0 * rng.standard_normal(size=input_shape)
    weights1 = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ExpQuad(input_shape),
        L0=linpde_gp.linfuncops.diffops.WeightedLaplacian(weights0),
        L1=linpde_gp.linfuncops.diffops.WeightedLaplacian(weights1),
    )


@parametrize(input_shape=input_shapes)
def case_expquad_directional_derivative_weighted_laplacian(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)
    weights = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ExpQuad(input_shape),
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
        L1=linpde_gp.linfuncops.diffops.WeightedLaplacian(weights),
    )
