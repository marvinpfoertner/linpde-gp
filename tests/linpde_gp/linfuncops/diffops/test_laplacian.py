from pytest_cases import fixture

from linpde_gp.linfuncops.diffops import WeightedLaplacian


@fixture
def laplacian() -> WeightedLaplacian:
    return WeightedLaplacian([1.0, 2.0, 3.0])


def test_coefficients(laplacian: WeightedLaplacian) -> WeightedLaplacian:
    assert len(laplacian.coefficients) == 1
    assert laplacian.coefficients[()] == {
        (2, 0, 0): 1.0,
        (0, 2, 0): 2.0,
        (0, 0, 2): 3.0,
    }
