import numpy as np
import probnum as pn

import pytest

from linpde_gp.linops import BlockMatrix2x2


@pytest.fixture
def symm_3x3():
    x = np.array([1.0, 2.0, 3.0])
    K = pn.randprocs.kernels.ExpQuad(())
    return K.matrix(x, x)


@pytest.fixture
def sbm_3x3(symm_3x3):
    A = pn.linops.Matrix(symm_3x3[:2, :2])
    D = pn.linops.Matrix(symm_3x3[2:, 2:])
    B = pn.linops.Matrix(symm_3x3[:2, 2:])
    assert np.allclose(B.T.todense(), symm_3x3[2:, :2])
    A.is_symmetric = True
    D.is_symmetric = True
    A.is_positive_definite = True
    D.is_positive_definite = True
    return BlockMatrix2x2(A, B, None, D, is_spd=True)


@pytest.fixture
def mat_3x3(symm_3x3):
    M = pn.linops.Matrix(symm_3x3)
    M.is_symmetric = True
    M.is_positive_definite = True
    return M


def test_cholesky(mat_3x3, sbm_3x3):
    L_full = mat_3x3.cholesky()
    L_block = sbm_3x3.cholesky()
    assert L_full.shape == (3, 3)
    assert L_block.shape == (3, 3)
    some_vec = np.array([10.0, 11.0, 12.0])
    assert np.allclose(L_full @ some_vec, L_block @ some_vec)


def test_inverse(mat_3x3, sbm_3x3):
    some_vec = np.array([10.0, 11.0, 12.0])
    assert np.allclose(mat_3x3.inv() @ some_vec, sbm_3x3.inv() @ some_vec)


@pytest.fixture
def symm_5x5():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    K = pn.randprocs.kernels.ExpQuad(())
    return K.matrix(x, x)


@pytest.fixture
def sbm_nested(symm_5x5):
    # SBM_1 = [A B
    #         B.T D]
    # SBM_2 = [SBM_1 E
    #         E.T    F]
    A = pn.linops.Matrix(symm_5x5[:2, :2])
    D = pn.linops.Matrix(symm_5x5[2:4, 2:4])
    B = pn.linops.Matrix(symm_5x5[:2, 2:4])
    A.is_symmetric = True
    D.is_symmetric = True
    A.is_positive_definite = True
    D.is_positive_definite = True
    SBM_1 = BlockMatrix2x2(A, B, None, D, is_spd=True)

    E = pn.linops.Matrix(symm_5x5[:4, 4:])
    F = pn.linops.Matrix(symm_5x5[4:, 4:])
    F.is_symmetric = True
    F.is_positive_definite = True
    SBM_2 = BlockMatrix2x2(SBM_1, E, None, F, is_spd=True)
    return SBM_2


@pytest.fixture
def mat_5x5(symm_5x5):
    M = pn.linops.Matrix(symm_5x5)
    M.is_symmetric = True
    M.is_positive_definite = True
    return M


def test_cholesky_nested(mat_5x5, sbm_nested):
    L_full = mat_5x5.cholesky()
    L_block = sbm_nested.cholesky()
    assert L_full.shape == (5, 5)
    assert L_block.shape == (5, 5)
    some_vec = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    assert np.allclose(L_full @ some_vec, L_block @ some_vec)


def test_inverse_nested(mat_5x5, sbm_nested):
    some_vec = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    assert np.allclose(mat_5x5.inv() @ some_vec, sbm_nested.inv() @ some_vec)
