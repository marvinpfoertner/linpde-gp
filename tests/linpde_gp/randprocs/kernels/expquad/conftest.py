import jax
from probnum.typing import ArrayLike, ShapeType

import pytest_cases

import linpde_gp

jax.config.update("jax_enable_x64", True)


@pytest_cases.fixture(scope="module")
def expquad(
    input_shape: ShapeType,
    lengthscales: ArrayLike,
    output_scale: float,
) -> linpde_gp.randprocs.kernels.ExpQuad:
    return linpde_gp.randprocs.kernels.ExpQuad(
        input_shape=input_shape,
        lengthscales=lengthscales,
        output_scale=output_scale,
    )


@pytest_cases.fixture(scope="module")
def expquad_jax(
    expquad: linpde_gp.randprocs.kernels.ExpQuad,
) -> linpde_gp.randprocs.kernels.JaxKernel:
    return linpde_gp.randprocs.kernels.JaxLambdaKernel(
        k=expquad.jax,
        input_shape=expquad.input_shape,
        vectorize=False,
    )
