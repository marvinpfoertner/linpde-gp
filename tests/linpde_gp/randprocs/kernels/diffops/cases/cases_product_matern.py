import numpy as np
from probnum.typing import ShapeType

from pytest_cases import parametrize

import linpde_gp

from ._test_case import KernelLinFuncOpTestCase

input_shapes = ((1,), (2,))


@parametrize(
    input_shape=input_shapes,
)
def case_identity_directional_derivative(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction**2))

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ProductMatern(input_shape=input_shape),
        L0=None,
        L1=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
    )


@parametrize(
    input_shape=input_shapes,
)
def case_directional_derivative_identity(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=input_shape)
    direction /= np.sqrt(np.sum(direction**2))

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ProductMatern(input_shape=input_shape),
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
        L1=None,
    )


@parametrize(
    input_shape=input_shapes,
)
def case_directional_derivative_directional_derivative(
    input_shape: ShapeType,
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction0 = rng.standard_normal(size=input_shape)
    direction0 /= np.sqrt(np.sum(direction0**2))

    direction1 = rng.standard_normal(size=input_shape)
    direction1 /= np.sqrt(np.sum(direction1**2))

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ProductMatern(input_shape=input_shape),
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction0),
        L1=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction1),
    )


@parametrize(
    input_shape=input_shapes,
)
def case_identity_laplacian(input_shape: ShapeType) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ProductMatern(input_shape=input_shape),
        L0=None,
        L1=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
    )


@parametrize(
    input_shape=input_shapes,
)
def case_laplacian_identity(input_shape: ShapeType) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ProductMatern(input_shape=input_shape),
        L0=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
        L1=None,
    )


@parametrize(
    input_shape=input_shapes,
)
def case_laplacian_laplacian(input_shape: ShapeType) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.ProductMatern(input_shape=input_shape),
        L0=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
        L1=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
    )
