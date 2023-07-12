import numpy as np

from linpde_gp.linfuncops import diffops
from linpde_gp.randprocs import covfuncs
from linpde_gp.randprocs.covfuncs.linfuncops import diffops as covfuncs_diffops

from ._test_case import CovarianceFunctionDiffOpTestCase


def case_tensor_product_identity_directional_derivative() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=1.5),
            covfuncs.Matern((), nu=1.5),
        ),
        L0=None,
        L1=diffops.DirectionalDerivative(direction),
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_directional_derivative_identity() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=1.5),
            covfuncs.Matern((), nu=1.5),
        ),
        L0=diffops.DirectionalDerivative(direction),
        L1=None,
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_directional_derivative_directional_derivative() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(390852098)

    direction0 = rng.standard_normal(size=(2,))
    direction0 /= np.sqrt(np.sum(direction0**2))

    direction1 = rng.standard_normal(size=(2,))
    direction1 /= np.sqrt(np.sum(direction1**2))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=1.5),
            covfuncs.Matern((), nu=1.5),
        ),
        L0=diffops.DirectionalDerivative(direction0),
        L1=diffops.DirectionalDerivative(direction1),
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_identity_weighted_laplacian() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(67835487)

    weights = 2.0 * rng.standard_normal(size=(2,))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=2.5),
            covfuncs.Matern((), nu=2.5),
        ),
        L0=None,
        L1=diffops.WeightedLaplacian(weights),
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_weighted_laplacian_identity() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(89012645)

    weights = 2.0 * rng.standard_normal(size=(2,))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=2.5),
            covfuncs.Matern((), nu=2.5),
        ),
        L0=diffops.WeightedLaplacian(weights),
        L1=None,
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_weighted_laplacian_weighted_laplacian() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(89012645)

    weights0 = 2.0 * rng.standard_normal(size=(2,))
    weights1 = 2.0 * rng.standard_normal(size=(2,))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=2.5),
            covfuncs.Matern((), nu=2.5),
        ),
        L0=diffops.WeightedLaplacian(weights0),
        L1=diffops.WeightedLaplacian(weights1),
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_directional_derivative_weighted_laplacian() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    weights = 2.0 * rng.standard_normal(size=(2,))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=3.5),
            covfuncs.Matern((), nu=3.5),
        ),
        L0=diffops.DirectionalDerivative(direction),
        L1=diffops.WeightedLaplacian(weights),
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_weighted_laplacian_directional_derivative() -> (
    CovarianceFunctionDiffOpTestCase
):
    rng = np.random.default_rng(390852098)

    direction = rng.standard_normal(size=(2,))
    direction /= np.sqrt(np.sum(direction**2))

    weights = 2.0 * rng.standard_normal(size=(2,))

    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=2.5),
            covfuncs.Matern((), nu=2.5),
        ),
        L0=diffops.WeightedLaplacian(weights),
        L1=diffops.DirectionalDerivative(direction),
        expected_type=covfuncs_diffops.TensorProduct_LinDiffOp_LinDiffOp,
    )


def case_tensor_product_identity_heat() -> CovarianceFunctionDiffOpTestCase:
    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=1.5),
            covfuncs.Matern((), nu=2.5),
        ),
        L0=None,
        L1=diffops.HeatOperator(domain_shape=(2,), alpha=0.1),
    )


def case_tensor_product_heat_heat() -> CovarianceFunctionDiffOpTestCase:
    return CovarianceFunctionDiffOpTestCase(
        k=covfuncs.TensorProduct(
            covfuncs.Matern((), nu=1.5),
            covfuncs.Matern((), nu=2.5),
        ),
        L0=diffops.HeatOperator(domain_shape=(2,), alpha=0.2),
        L1=diffops.HeatOperator(domain_shape=(2,), alpha=0.1),
    )
