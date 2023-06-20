import numpy as np

import pytest
from pytest_cases import fixture

from linpde_gp.linfuncops.diffops import PartialDerivativeCoefficients


@fixture
@pytest.mark.parametrize("ndim", [0, 1, 2])
def input_codomain_shape(ndim: int) -> tuple:
    rng = np.random.default_rng(60324902 + ndim)
    return tuple(rng.integers(1, 4, size=(ndim,)))


@fixture
@pytest.mark.parametrize("size", [1, 2])
def input_domain_size(size: int) -> int:
    return size


@pytest.fixture
def random_coefficients(input_domain_size: int, input_codomain_shape: tuple):
    # Sample random indices from domain shape
    rng = np.random.default_rng(2348091)

    if input_codomain_shape == ():
        codomain_indices = np.array([()])
    else:
        codomain_indices = np.indices(input_codomain_shape)
        codomain_indices = np.moveaxis(codomain_indices, 0, -1)
        codomain_indices = codomain_indices.reshape((-1, len(input_codomain_shape)))

    multi_indices = np.meshgrid(*([np.arange(0, 3, 1)] * input_domain_size))
    multi_indices = np.moveaxis(multi_indices, 0, -1).reshape((-1, input_domain_size))

    num_codomain = int(len(codomain_indices) / 2) + 1
    codomain_indices = rng.choice(codomain_indices, size=(num_codomain,), replace=False)

    num_coefficients_per_codomain = int(len(multi_indices) / 2) + 1

    coefficient_dict = {}

    for codomain_idx in codomain_indices:
        codomain_idx = tuple(codomain_idx)
        domain_indices = rng.choice(
            multi_indices, size=(num_coefficients_per_codomain,), replace=False
        )
        coefficient_dict[codomain_idx] = {}
        for domain_idx in domain_indices:
            domain_idx = tuple(domain_idx)
            coefficient = 5 * rng.uniform(-1.0, 1.0)
            coefficient_dict[codomain_idx][domain_idx] = coefficient

    return PartialDerivativeCoefficients(coefficient_dict)


def test_partial_derivative_coefficients_validate_input_shapes(
    random_coefficients: PartialDerivativeCoefficients,
    input_domain_size,
    input_codomain_shape,
):
    assert random_coefficients.validate_input_domain_shape((input_domain_size,))
    if input_domain_size == 1:
        assert random_coefficients.validate_input_domain_shape(())
    assert random_coefficients.validate_input_codomain_shape(input_codomain_shape)
    assert not random_coefficients.validate_input_domain_shape((input_domain_size + 1,))
    assert not random_coefficients.validate_input_codomain_shape(
        input_codomain_shape + (1,)
    )


@fixture
def coefficients():
    coefficient_dict = {
        (0, 0): {(0, 1, 0, 0): 1.0, (0, 0, 1, 0): 2.0},
        (1, 0): {(0, 0, 0, 1): 3.0},
    }
    return PartialDerivativeCoefficients(coefficient_dict)


def test_getitem(coefficients: PartialDerivativeCoefficients):
    assert coefficients[(0, 0)] == {(0, 1, 0, 0): 1.0, (0, 0, 1, 0): 2.0}
    assert coefficients[(1, 0)] == {(0, 0, 0, 1): 3.0}


def test_partial_derivative_coefficients_len(
    coefficients: PartialDerivativeCoefficients,
):
    assert len(coefficients) == 2


def test_num_coefficients(coefficients: PartialDerivativeCoefficients):
    assert coefficients.num_entries == 3


def test_neg(coefficients: PartialDerivativeCoefficients):
    neg_coefficients = -coefficients
    assert neg_coefficients[(0, 0)] == {(0, 1, 0, 0): -1.0, (0, 0, 1, 0): -2.0}
    assert neg_coefficients[(1, 0)] == {(0, 0, 0, 1): -3.0}


def test_add(coefficients: PartialDerivativeCoefficients):
    added_coefficients = coefficients + coefficients
    assert added_coefficients[(0, 0)] == {(0, 1, 0, 0): 2.0, (0, 0, 1, 0): 4.0}
    assert added_coefficients[(1, 0)] == {(0, 0, 0, 1): 6.0}


def test_rmul(coefficients: PartialDerivativeCoefficients):
    multiplied_coefficients = 4.0 * coefficients
    assert multiplied_coefficients[(0, 0)] == {(0, 1, 0, 0): 4.0, (0, 0, 1, 0): 8.0}
    assert multiplied_coefficients[(1, 0)] == {(0, 0, 0, 1): 12.0}


def test_input_domain_shape_mismatch():
    coefficients_dict = {(): {(0, 1): 1.0, (1, 0, 0): 4.0}}
    with pytest.raises(ValueError):
        PartialDerivativeCoefficients(coefficients_dict)


def test_input_codomain_shape_mismatch():
    coefficients_dict = {(0,): {(0, 1): 1.0}, (1, 1): {(1, 0): 4.0}}
    with pytest.raises(ValueError):
        PartialDerivativeCoefficients(coefficients_dict)


def test_add_codomain_shape_mismatch():
    coefficients_dict1 = {(0, 0): {(1, 0): 1.0}}
    coefficients_dict2 = {(0,): {(1, 0): 1.0}}
    coefficients1 = PartialDerivativeCoefficients(coefficients_dict1)
    coefficients2 = PartialDerivativeCoefficients(coefficients_dict2)
    with pytest.raises(ValueError):
        coefficients1 + coefficients2  # pylint: disable=pointless-statement


def test_add_domain_shape_mismatch():
    coefficients_dict1 = {(0,): {(0, 1): 1.0}}
    coefficients_dict2 = {(0,): {(0, 0, 1): 1.0}}
    coefficients1 = PartialDerivativeCoefficients(coefficients_dict1)
    coefficients2 = PartialDerivativeCoefficients(coefficients_dict2)
    with pytest.raises(ValueError):
        coefficients1 + coefficients2  # pylint: disable=pointless-statement
