from typing import Union

import numpy as np
from probnum.typing import ShapeType

import pytest_cases


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize(
    "input_shape",
    ((), (1,), (2,), (3,)),
    ids=lambda input_shape: f"inshape={input_shape}",
)
def input_shape(input_shape: ShapeType) -> ShapeType:
    return input_shape


@pytest_cases.parametrize(lengthscale=(0.4, 1.0, 3.0))
def case_lengthscales_scalar(lengthscale) -> float:
    return lengthscale


def case_lengthscales_diagonal(input_shape: ShapeType) -> np.ndarray:
    seed = abs(hash(input_shape) + 34876)

    return np.random.default_rng(seed).uniform(0.4, 3.0, size=input_shape)


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize_with_cases(
    "lengthscales_",
    cases=pytest_cases.THIS_MODULE,
    glob="lengthscales_*",
    scope="package",
)
def lengthscales(lengthscales_: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return lengthscales_


@pytest_cases.fixture(scope="package")
@pytest_cases.parametrize(output_scale=(0.1, 1.0, 2.8))
def output_scale(output_scale: float) -> float:
    return output_scale


@pytest_cases.fixture(scope="package")
def X(input_shape: ShapeType) -> np.ndarray:
    rng = np.random.default_rng(198748)

    return rng.normal(scale=2.0, size=(100,) + input_shape)
