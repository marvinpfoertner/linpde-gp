import pathlib

from ._test_case import CovarianceFunctionLinearFunctionalTestCase

case_modules = [
    ".cases." + path.stem for path in pathlib.Path(__file__).parent.glob("cases_*.py")
]
