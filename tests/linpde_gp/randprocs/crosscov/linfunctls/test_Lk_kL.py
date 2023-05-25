import numpy as np

from pytest_cases import parametrize_with_cases

from .cases import CovarianceFunctionLinearFunctionalTestCase, case_modules


@parametrize_with_cases("test_case", cases=case_modules)
def test_expected_type(test_case: CovarianceFunctionLinearFunctionalTestCase):
    if test_case.expected_type is not None:
        # pylint: disable=unidiomatic-typecheck
        assert type(test_case.Lk) == test_case.expected_type
        assert type(test_case.kL) == test_case.expected_type


@parametrize_with_cases("test_case", cases=case_modules)
def test_Lk(test_case: CovarianceFunctionLinearFunctionalTestCase):
    Lk_X_test = test_case.Lk(test_case.X_test)
    Lk_fallback_X_test = test_case.Lk_fallback(test_case.X_test)

    np.testing.assert_allclose(Lk_X_test, Lk_fallback_X_test)


@parametrize_with_cases("test_case", cases=case_modules)
def test_kL(test_case: CovarianceFunctionLinearFunctionalTestCase):
    kL_X_test = test_case.Lk(test_case.X_test)
    kL_fallback_X_test = test_case.kL_fallback(test_case.X_test)

    np.testing.assert_allclose(kL_X_test, kL_fallback_X_test)
