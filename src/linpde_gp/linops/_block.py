import numpy as np
import probnum as pn


class BlockMatrix(pn.linops.LambdaLinearOperator):
    def __init__(self, A, B, C, D):
        self._A = pn.linops.aslinop(A)
        self._B = pn.linops.aslinop(B)
        self._C = pn.linops.aslinop(C)
        self._D = pn.linops.aslinop(D)

        assert A.shape[0] == B.shape[0]
        assert A.shape[1] == C.shape[1]
        assert D.shape[0] == C.shape[0]
        assert D.shape[1] == B.shape[1]

        super().__init__(
            shape=(A.shape[0] + D.shape[0], A.shape[1] + D.shape[1]),
            dtype=np.promote_types(
                np.promote_types(self._A.dtype, self._B.dtype),
                np.promote_types(self._C.dtype, self._D.dtype),
            ),
            matmul=self._matmul,
            rmatmul=self._rmatmul,
            apply=self._apply,
            todense=lambda: np.block(
                [
                    [self._A.todense(cache=False), self._B.todense(cache=False)],
                    [self._C.todense(cache=False), self._D.todense(cache=False)],
                ]
            ),
            transpose=lambda: BlockMatrix(
                A=self._A.T, B=self._C.T, C=self._B.T, D=self._D.T
            ),
            inverse=lambda: BlockInverse(self),
            # det= TODO
            trace=lambda: self._A.trace + self._D.trace,
        )

    def _matmul(self, x):
        x0, x1 = self._split_input(x, axis=-2)

        return np.concatenate(
            (
                self._A @ x0 + self._B @ x1,
                self._C @ x0 + self._D @ x1,
            ),
            axis=-2,
        )

    def _rmatmul(self, x):
        x0, x1 = self._split_input(x, axis=-1)

        return np.concatenate(
            (
                x0 @ self._A + x1 @ self._C,
                x0 @ self._B + x1 @ self._D,
            ),
            axis=-1,
        )

    def _apply(self, x, axis):
        x0, x1 = self._split_input(x, axis)

        return np.concatenate(
            (
                self._A(x0, axis=axis) + self._B(x1, axis=axis),
                self._C(x0, axis=axis) + self._D(x1, axis=axis),
            ),
            axis=axis,
        )

    def _split_input(self, x: np.ndarray, axis: int):
        return np.split(x, [self._A.shape[1]], axis=axis)


class BlockInverse(BlockMatrix):
    def __init__(self, block_matrix: BlockMatrix):
        self._bm = block_matrix
        self._Ainv_B = self._bm._A.inv() @ self._bm._B
        self._S = self._bm._D - self._bm._C @ self._Ainv_B

        S_inv_C_A_inv = self._S.inv() @ self._bm._C @ self._bm._A.inv()
        A = self._bm._A.inv() + self._Ainv_B @ S_inv_C_A_inv
        B = -self._Ainv_B @ self._S.inv()
        C = -S_inv_C_A_inv
        D = self._S.inv()

        super().__init__(
            A=A, B=B, C=C, D=D
        )

    @property
    def schur(self):
        return self._S

    def schur_update(self, Ainv_u, v):
        y = self.schur.inv() @ (v - self._bm._C @ Ainv_u)
        x = Ainv_u - self._Ainv_B @ y

        return np.concatenate((x, y), axis=0)

class SymmetricBlockMatrix(BlockMatrix):
    def __init__(self, A, B, D):
        self._A = pn.linops.aslinop(A)
        self._B = pn.linops.aslinop(B)
        self._D = pn.linops.aslinop(D)

        assert self._A.is_symmetric
        assert self._D.is_symmetric

        super().__init__(self._A, self._B, self._B.T, self._D)

        self.is_symmetric = True
        self.is_positive_definite = True

        self._schur = None

    
    @property
    def A(self):
        return self._A
    
    @property
    def B(self):
        return self._B
    
    @property
    def D(self):
        return self._D

    def _cholesky(self, lower: bool) -> pn.linops.LinearOperator:
        A = self.A
        B = self.B
        D = self.D

        A_sqrt = A.cholesky(True)
        A_sqrt.is_lower_triangular = True

        L_A_inv_B = A_sqrt.inv() @ B

        # Compute the Schur complement of A in the block matrix
        if self._schur is None:
            self._schur = pn.linops.Matrix((D - L_A_inv_B.T @ L_A_inv_B).todense())
        self._schur.is_symmetric = True
        self._schur.is_positive_definite = True
        S_sqrt = self._schur.cholesky(True)
        S_sqrt.is_lower_triangular = True

        if lower:
            block_sqrt = BlockTriangularMatrix(A_sqrt, L_A_inv_B.T, S_sqrt, lower)
        else:
            block_sqrt = BlockTriangularMatrix(A_sqrt.T, L_A_inv_B, S_sqrt.T, lower)
        return block_sqrt

    def _inverse(self) -> pn.linops.LinearOperator:
        return SymmetricBlockInverse(self)

    @property
    def schur(self):
        if self._schur is None:
            self._schur = self.D - self.B.T @ self.A.inv() @ self.B
            self._schur.is_symmetric = True
            self._schur.is_positive_definite = True
        return self._schur

    def schur_update(self, A_inv_u, v):
        A = self.A
        B = self.B
        y = self.schur.inv() @ (v - B.T @ A_inv_u)
        x = A_inv_u - A.inv() @ B @ y
        return np.concatenate((x, y))

    def __add__(self, other):
        if isinstance(other, pn.linops.Matrix) or isinstance(other, np.ndarray):
            other = pn.linops.aslinop(other).todense()
            assert other.shape == self.shape
            A_i_end, A_j_end = self.A.shape
            other_A = other[:A_i_end, :A_j_end]
            other_B = other[:A_i_end, A_j_end:]
            other_D = other[A_i_end:, A_j_end:]
            if np.allclose(other.T, other):
                return SymmetricBlockMatrix(self.A + other_A, self.B + other_B, self.D + other_D)
            else:
                other_C = other[A_i_end:, :A_j_end]
                return BlockMatrix(
                    self.A + other_A,
                    self.B + other_B,
                    self.B.T + other_C,
                    self.D + other_D
                )
        return super().__add__(other)

class SymmetricBlockInverse(BlockInverse):
    def __init__(self, symmetric_block_matrix: SymmetricBlockMatrix):
        self._sbm = symmetric_block_matrix

        super().__init__(self._sbm)

        self.is_symmetric = True
        self.is_positive_definite = True

    def _matmul(self, x):
        L = self._sbm.cholesky(True)
        return L.T.inv() @ (L.inv() @ x)

class BlockTriangularMatrix(BlockMatrix):
    def __init__(self, top_left, off_diagonal, bottom_right, lower=False):
        top_left = pn.linops.aslinop(top_left)
        off_diagonal = pn.linops.aslinop(off_diagonal)
        bottom_right = pn.linops.aslinop(bottom_right)

        self._top_left = top_left
        self._off_diagonal = off_diagonal
        self._bottom_right = bottom_right

        lower_left = off_diagonal if lower else pn.linops.Zero(off_diagonal.T.shape, off_diagonal.dtype)
        top_right = pn.linops.Zero(off_diagonal.T.shape, off_diagonal.T.dtype) if lower else off_diagonal

        super().__init__(top_left, top_right, lower_left, bottom_right)

        if lower:
            assert top_left.is_lower_triangular
            assert bottom_right.is_lower_triangular
            self.is_lower_triangular = True
        else:
            assert top_left.is_upper_triangular
            assert bottom_right.is_upper_triangular
            self.is_upper_triangular = True
    
    @property
    def top_left(self):
        return self._top_left
    
    @property
    def off_diagonal(self):
        return self._off_diagonal
    
    @property
    def bottom_right(self):
        return self._bottom_right

    def _transpose(self) -> pn.linops.LinearOperator:
        return BlockTriangularMatrix(self.top_left.T, self.off_diagonal.T, self.bottom_right.T, not self.is_lower_triangular)

    def _inverse(self) -> pn.linops.LinearOperator:
        return InverseBlockTriangularMatrix(self)
    
class InverseBlockTriangularMatrix(pn.linops.LinearOperator):
    def __init__(self, btm: BlockTriangularMatrix):
        self._btm = btm

        super().__init__(self._btm.shape, self._btm.dtype)
        self.is_lower_triangular = self._btm.is_lower_triangular

    def _matmul(self, x: np.ndarray) -> np.ndarray:
        u, v = self._btm._split_input(x, axis=-2)

        if self._btm.is_lower_triangular:
            y0 = self._btm.top_left.inv() @ u
            y1 = self._btm.bottom_right.inv() @ (v - self._btm.off_diagonal @ y0)
        else:
            y1 = self._btm.bottom_right.inv() @ v
            y0 = self._btm.top_left.inv() @ (u - self._btm.off_diagonal @ y1)
        return np.concatenate((y0, y1), axis=-2)