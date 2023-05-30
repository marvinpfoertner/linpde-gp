import numpy as np

from pytest_cases import parametrize_with_cases

from .cases import CovarianceFunctionLinearFunctionalsTestCase, case_modules


@parametrize_with_cases("test_case", cases=case_modules)
def test_shape(test_case: CovarianceFunctionLinearFunctionalsTestCase):
    assert (
        test_case.L0kL1.shape == test_case.L0.output_shape + test_case.L1.output_shape
    )


@parametrize_with_cases("test_case", cases=case_modules)
def test_values(test_case: CovarianceFunctionLinearFunctionalsTestCase):
    np.testing.assert_allclose(
        test_case.L0kL1,
        test_case.L0kL1_fallback,
    )
