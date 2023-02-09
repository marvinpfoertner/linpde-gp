import numpy as np

from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import kernels

from ._test_case import KernelLinFuncOpTestCase


def case_tensor_product_identity_directional_derivative() -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=None,
        L1=diffops.DirectionalDerivative(direction),
    )


def case_tensor_product_directional_derivative_identity() -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=diffops.DirectionalDerivative(direction),
        L1=None,
    )


def case_tensor_product_directional_derivative_directional_derivative() -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction0 = rng.standard_normal(size=(2,))
    direction0 /= np.sqrt(np.sum(direction0**2))

    direction1 = rng.standard_normal(size=(2,))
    direction1 /= np.sqrt(np.sum(direction1**2))

    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=diffops.DirectionalDerivative(direction0),
        L1=diffops.DirectionalDerivative(direction1),
    )


def case_tensor_product_identity_laplacian() -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=None,
        L1=diffops.Laplacian(domain_shape=(2,)),
    )


def case_tensor_product_laplacian_identity() -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=diffops.Laplacian(domain_shape=(2,)),
        L1=None,
    )


def case_tensor_product_laplacian_laplacian() -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=diffops.Laplacian(domain_shape=(2,)),
        L1=diffops.Laplacian(domain_shape=(2,)),
    )


def case_tensor_product_directional_derivative_laplacian() -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=diffops.DirectionalDerivative(direction),
        L1=diffops.Laplacian(domain_shape=(2,)),
    )


def case_tensor_product_laplacian_directional_derivative() -> KernelLinFuncOpTestCase:
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=diffops.Laplacian(domain_shape=(2,)),
        L1=diffops.DirectionalDerivative(direction),
    )


def case_tensor_product_identity_heat() -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=None,
        L1=diffops.HeatOperator(domain_shape=(2,), alpha=0.1),
    )


def case_tensor_product_heat_heat() -> KernelLinFuncOpTestCase:
    return KernelLinFuncOpTestCase(
        k=kernels.TensorProductKernel(
            kernels.Matern((), p=3),
            kernels.Matern((), p=3),
        ),
        L0=diffops.HeatOperator(domain_shape=(2,), alpha=0.2),
        L1=diffops.HeatOperator(domain_shape=(2,), alpha=0.1),
    )
