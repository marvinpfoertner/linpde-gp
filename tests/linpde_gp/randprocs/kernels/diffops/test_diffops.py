import pathlib

import numpy as np
from probnum.typing import ShapeType

from pytest_cases import parametrize_with_cases

from .cases import KernelLinFuncOpTestCase

case_modules = [
    ".cases." + path.stem
    for path in (pathlib.Path(__file__).parent / "cases").glob("cases_*.py")
]


def X(input_shape: ShapeType) -> np.ndarray:
    rng = np.random.default_rng(897523)

    return rng.normal(scale=2.0, size=(100,) + input_shape)


@parametrize_with_cases("test_case", cases=case_modules)
def test_L0_k_L1_adj(test_case: KernelLinFuncOpTestCase):
    Xs = X(test_case.k.input_shape)

    L0_k_L1_adj = test_case.L0_k_L1_adj(Xs[:, None], Xs[None, :])
    L0_k_L1_adj_jax = test_case.L0_k_L1_adj_jax(Xs[:, None], Xs[None, :])

    nan_mask = np.isnan(L0_k_L1_adj_jax)

    if np.any(nan_mask):
        L0_k_L1_adj_jax[nan_mask] = L0_k_L1_adj[nan_mask]

    np.testing.assert_allclose(L0_k_L1_adj, L0_k_L1_adj_jax, atol=1e-14)
