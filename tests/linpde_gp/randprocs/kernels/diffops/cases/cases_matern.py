from probnum.typing import ShapeType

from pytest_cases import parametrize

import linpde_gp

from ._test_case import KernelLinFuncOpTestCase

input_shapes = ((),)


@parametrize(
    input_shape=input_shapes,
)
def case_identity_laplacian(input_shape: ShapeType) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape=input_shape),
        L0=None,
        L1=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
    )


@parametrize(
    input_shape=input_shapes,
)
def case_laplacian_identity(input_shape: ShapeType) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape=input_shape),
        L0=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
        L1=None,
    )


@parametrize(
    input_shape=input_shapes,
)
def case_laplacian_laplacian(input_shape: ShapeType) -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=linpde_gp.randprocs.kernels.Matern(input_shape=input_shape),
        L0=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
        L1=linpde_gp.linfuncops.diffops.Laplacian(domain_shape=input_shape),
    )
