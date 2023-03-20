import numpy as np
from probnum.linops import Identity, LinearOperator, Matrix, Zero

import pytest

from linpde_gp.linops import BlockMatrix


@pytest.fixture(params=[2349023, 896934, 12983, 5492538])
def linop_blocks(request) -> np.ndarray:
    np.random.seed(request.param)
    shape = (np.random.randint(5) + 1, np.random.randint(5) + 1)
    row_dims = [np.random.randint(3) + 1 for i in range(shape[0])]
    col_dims = [np.random.randint(3) + 1 for j in range(shape[1])]

    def pull_linop(i: int, j: int) -> LinearOperator:
        shape = (row_dims[i], col_dims[j])
        linops = [Matrix(np.random.rand(*shape)), Zero(shape)]
        if row_dims[i] == col_dims[j]:
            linops.append(Identity(shape))
        return np.random.choice(linops)

    return np.block(
        [[pull_linop(i, j) for j in range(shape[1])] for i in range(shape[0])]
    )


@pytest.fixture
def block_linop(linop_blocks: np.ndarray):
    return BlockMatrix(linop_blocks.tolist())


@pytest.fixture
def dense_matrix(linop_blocks: np.ndarray):
    dense_mat = np.copy(linop_blocks)
    for i, j in np.ndindex(dense_mat.shape):
        dense_mat[i, j] = dense_mat[i, j].todense()
    return np.block(dense_mat.tolist())


def test_matmul(block_linop: BlockMatrix, dense_matrix: np.ndarray):
    np.random.seed(12309124)
    assert block_linop.shape == dense_matrix.shape
    x = np.random.rand(block_linop.shape[1])
    np.testing.assert_allclose(block_linop @ x, dense_matrix @ x)


def test_todense(block_linop: BlockMatrix, dense_matrix: np.ndarray):
    np.testing.assert_allclose(block_linop.todense(), dense_matrix)


def test_transpose(block_linop: BlockMatrix, dense_matrix: np.ndarray):
    np.testing.assert_allclose(block_linop.T.todense(), dense_matrix.T)
