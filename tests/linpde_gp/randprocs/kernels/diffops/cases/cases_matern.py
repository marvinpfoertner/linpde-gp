import numpy as np
from probnum.typing import ShapeType

import pytest
from pytest_cases import parametrize

import linpde_gp

from ._test_case import KernelLinFuncOpTestCase

input_shapes = ((), (1,), (3,))
nus_directional_derivative = (1.5, 2.5, 3.5, 4.5)
nus_laplacian = (2.5, 3.5, 4.5)


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_identity_directional_derivative(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu),
        L0=None,
        L1=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
    )


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_directional_derivative_identity(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu),
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
        L1=None,
    )


@parametrize(
    input_shape=input_shapes,
    nu=nus_directional_derivative,
)
def case_matern_directional_derivative_directional_derivative(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    k = linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu)

    if k.input_size > 1 and nu <= 1.5:
        pytest.skip("Not enough differentiability")

    rng = np.random.default_rng(413598)

    direction0 = rng.standard_normal(size=input_shape)
    direction1 = rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=k,
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction0),
        L1=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction1),
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_identity_laplacian(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu),
        L0=None,
        L1=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_laplacian_identity(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu),
        L0=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
        L1=None,
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_laplacian_laplacian(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu),
        L0=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
        L1=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_directional_derivative_laplacian(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu),
        L0=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
        L1=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
    )


@parametrize(
    input_shape=((),),
    nu=nus_laplacian,
)
def case_matern_laplacian_directional_derivative(
    input_shape: ShapeType, nu: float
) -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(4158976)

    direction = 2.0 * rng.standard_normal(size=input_shape)

    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape, nu=nu),
        L0=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
        L1=linpde_gp.linfuncops.diffops.DirectionalDerivative(direction),
    )
