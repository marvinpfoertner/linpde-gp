from pytest_cases import fixture

from linpde_gp.linfuncops.diffops import DirectionalDerivative, MultiIndex


@fixture
def dir_deriv() -> DirectionalDerivative:
    return DirectionalDerivative([1.0, 2.0, 3.0])


def test_coefficients(dir_deriv: DirectionalDerivative):
    assert len(dir_deriv.coefficients) == 1
    assert dir_deriv.coefficients[()] == {
        MultiIndex((1, 0, 0)): 1.0,
        MultiIndex((0, 1, 0)): 2.0,
        MultiIndex((0, 0, 1)): 3.0,
    }
