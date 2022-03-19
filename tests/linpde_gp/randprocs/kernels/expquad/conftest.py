import jax
from probnum.typing import ArrayLike, ShapeType

import pytest_cases

import linpde_gp

jax.config.update("jax_enable_x64", True)


@pytest_cases.fixture(scope="module")
def k(
    input_shape: ShapeType,
    lengthscales: ArrayLike,
    output_scale: float,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return linpde_gp.randprocs.kernels.ExpQuad(
        input_shape=input_shape,
        lengthscales=lengthscales,
        output_scale=output_scale,
    )


@pytest_cases.fixture(scope="module")
def k_jax(
    k: linpde_gp.randprocs.kernels.JaxKernel,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return linpde_gp.randprocs.kernels.JaxLambdaKernel(
        k=k.jax,
        input_shape=k.input_shape,
        vectorize=False,
    )
