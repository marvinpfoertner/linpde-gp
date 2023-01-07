import pytest
import numpy as np
import linpde_gp
from linpde_gp.linfunctls import StackedLinearFunctional, DiracFunctional
from linpde_gp.linops import SymmetricBlockMatrix


def test_block_matrix():
    rng = np.random.default_rng(7398432)
    mip = linpde_gp.randprocs.kernels.MultiIndependentProductMatern(
        2, 3, rng.random(size=(3, 2))
    )
    X1 = rng.random(size=(50, 2))
    X2 = rng.random(size=(100, 2))
    d1 = DiracFunctional((2,), (3,), X1)
    d2 = DiracFunctional((2,), (3,), X2)
    S = StackedLinearFunctional(d1, d2)

    cov_d1 = d1(d1(mip, argnum=1))
    assert cov_d1.shape == (50, 3, 50, 3)
    cov_d1_flattened = cov_d1.reshape((150, 150), order="C")
    cov_d2 = d2(d2(mip, argnum=1))
    assert cov_d2.shape == (100, 3, 100, 3)
    cov_d2_flattened = cov_d2.reshape((300, 300), order="C")

    cov_d1_d2 = d1(d2(mip, argnum=1))
    assert cov_d1_d2.shape == (50, 3, 100, 3)
    cov_d1_d2_flattened = cov_d1_d2.reshape((150, 300), order="C")

    block_cov = S(S(mip, argnum=1))
    assert isinstance(block_cov, SymmetricBlockMatrix)
    assert block_cov.shape == (450, 450)
    np.testing.assert_allclose(block_cov.A.todense(), cov_d1_flattened)  # Top left block
    np.testing.assert_allclose(
        block_cov.D.todense(), cov_d2_flattened
    )  # Bottom right block
    np.testing.assert_allclose(
        block_cov.B.todense(), cov_d1_d2_flattened
    )  # Top right block
