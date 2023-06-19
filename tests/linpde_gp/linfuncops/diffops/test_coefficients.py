import numpy as np

import pytest
from pytest_cases import fixture

from linpde_gp.linfuncops.diffops import PartialDerivativeCoefficients


@fixture
@pytest.mark.parametrize("ndim", [0, 1, 2])
def input_domain_shape(ndim: int) -> np.ndarray:
    rng = np.random.default_rng(60324902 + ndim)
    return tuple(rng.integers(1, 4, size=(ndim,)))


@fixture
@pytest.mark.parametrize("ndim", [0, 1, 2])
def input_codomain_shape(ndim: int) -> np.ndarray:
    rng = np.random.default_rng(893294872 + ndim)
    return tuple(rng.integers(1, 4, size=(ndim,)))


@pytest.fixture
def random_coefficients(input_domain_shape, input_codomain_shape):
    # Sample random indices from domain shape
    rng = np.random.default_rng(2348091)

    combined_shape = input_codomain_shape + input_domain_shape

    if combined_shape == ():
        all_indices = np.array([()])
    else:
        all_indices = np.indices(combined_shape)
        all_indices = np.moveaxis(all_indices, 0, -1)
        all_indices = all_indices.reshape((-1, len(combined_shape)))

    num_coefficients = int(np.prod(combined_shape) / 2) + 1
    coefficient_indices = rng.choice(
        all_indices, size=(num_coefficients,), replace=False
    )
    coefficients = 5.0 * rng.standard_normal((num_coefficients,))
    orders = rng.integers(0, 3, size=(num_coefficients,))

    multi_indices = {}

    codomain_ndim = len(input_codomain_shape)
    for i, coefficient_idx in enumerate(coefficient_indices):
        coefficient_idx = tuple(coefficient_idx)
        codomain_idx = coefficient_idx[:codomain_ndim]
        domain_idx = coefficient_idx[codomain_ndim:]
        if codomain_idx in multi_indices:
            multi_indices[codomain_idx][(domain_idx, orders[i])] = coefficients[i]
        else:
            multi_indices[codomain_idx] = {(domain_idx, orders[i]): coefficients[i]}

    return PartialDerivativeCoefficients(multi_indices)


def test_partial_derivative_coefficients_validate_input_shapes(
    random_coefficients: PartialDerivativeCoefficients,
    input_domain_shape,
    input_codomain_shape,
):
    assert random_coefficients.validate_input_domain_shape(input_domain_shape)
    assert random_coefficients.validate_input_codomain_shape(input_codomain_shape)
    assert not random_coefficients.validate_input_domain_shape(
        input_domain_shape + (1,)
    )
    assert not random_coefficients.validate_input_codomain_shape(
        input_codomain_shape + (1,)
    )


@fixture
def coefficients():
    coefficient_dict = {
        (0, 0): {((1,), 0): 1.0, ((2,), 0): 2.0},
        (1, 0): {((3,), 1): 3.0},
    }
    return PartialDerivativeCoefficients(coefficient_dict)


def test_getitem(coefficients: PartialDerivativeCoefficients):
    assert coefficients[(0, 0)] == {((1,), 0): 1.0, ((2,), 0): 2.0}
    assert coefficients[(1, 0)] == {((3,), 1): 3.0}


def test_partial_derivative_coefficients_len(
    coefficients: PartialDerivativeCoefficients,
):
    assert len(coefficients) == 2


def test_num_coefficients(coefficients: PartialDerivativeCoefficients):
    assert coefficients.num_entries == 3


def test_neg(coefficients: PartialDerivativeCoefficients):
    neg_coefficients = -coefficients
    assert neg_coefficients[(0, 0)] == {((1,), 0): -1.0, ((2,), 0): -2.0}
    assert neg_coefficients[(1, 0)] == {((3,), 1): -3.0}


def test_add(coefficients: PartialDerivativeCoefficients):
    added_coefficients = coefficients + coefficients
    assert added_coefficients[(0, 0)] == {((1,), 0): 2.0, ((2,), 0): 4.0}
    assert added_coefficients[(1, 0)] == {((3,), 1): 6.0}


def test_rmul(coefficients: PartialDerivativeCoefficients):
    multiplied_coefficients = 4.0 * coefficients
    assert multiplied_coefficients[(0, 0)] == {((1,), 0): 4.0, ((2,), 0): 8.0}
    assert multiplied_coefficients[(1, 0)] == {((3,), 1): 12.0}


def test_input_domain_shape_mismatch():
    coefficients_dict = {(): {((1,), 0): 1.0, ((2, 3), 0): 4.0}}
    with pytest.raises(ValueError):
        PartialDerivativeCoefficients(coefficients_dict)


def test_input_codomain_shape_mismatch():
    coefficients_dict = {(0,): {((1,), 0): 1.0}, (1, 1): {((2,), 0): 4.0}}
    with pytest.raises(ValueError):
        PartialDerivativeCoefficients(coefficients_dict)


def test_add_codomain_shape_mismatch():
    coefficients_dict1 = {(0, 0): {((1,), 0): 1.0}}
    coefficients_dict2 = {(0,): {((1,), 0): 1.0}}
    coefficients1 = PartialDerivativeCoefficients(coefficients_dict1)
    coefficients2 = PartialDerivativeCoefficients(coefficients_dict2)
    with pytest.raises(ValueError):
        coefficients1 + coefficients2  # pylint: disable=pointless-statement


def test_add_domain_shape_mismatch():
    coefficients_dict1 = {(0,): {((1,), 0): 1.0}}
    coefficients_dict2 = {(0,): {((1, 1), 0): 1.0}}
    coefficients1 = PartialDerivativeCoefficients(coefficients_dict1)
    coefficients2 = PartialDerivativeCoefficients(coefficients_dict2)
    with pytest.raises(ValueError):
        coefficients1 + coefficients2  # pylint: disable=pointless-statement
