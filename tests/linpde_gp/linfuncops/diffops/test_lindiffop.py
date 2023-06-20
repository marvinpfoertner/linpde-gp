import pytest
from pytest_cases import fixture

from linpde_gp.linfuncops.diffops import (
    LinearDifferentialOperator,
    PartialDerivativeCoefficients,
)


@fixture
def coefficients() -> PartialDerivativeCoefficients:
    return PartialDerivativeCoefficients({(1,): {(0, 0, 1): 1.0}})


def test_coefficients_domain_shape_mismatch(
    coefficients: PartialDerivativeCoefficients,
):
    with pytest.raises(ValueError):
        LinearDifferentialOperator(coefficients, input_shapes=((1,), (2,)))


def test_coefficients_codomain_shape_mismatch(
    coefficients: PartialDerivativeCoefficients,
):
    with pytest.raises(ValueError):
        LinearDifferentialOperator(coefficients, input_shapes=((3,), (1,)))
