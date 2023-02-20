import pathlib

from ._test_case import CovarianceFunctionDiffOpTestCase

case_modules = [
    ".cases." + path.stem for path in pathlib.Path(__file__).parent.glob("cases_*.py")
]
