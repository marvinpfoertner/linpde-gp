import pathlib

from ._test_case import KernelDiffOpTestCase

case_modules = [
    ".cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "cases").glob("cases_*.py")
]
