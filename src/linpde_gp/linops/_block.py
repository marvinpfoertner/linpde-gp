import numpy as np
import probnum as pn


class BlockMatrix(pn.linops.LinearOperator):
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
        return np.split(x, np.array(self._A.shape[1]), axis=axis)


class BlockInverse(pn.linops.LinearOperator):
    def __init__(self, block_matrix):
        self._bm = block_matrix

        super().__init__(
            shape=self._bm.shape,
            dtype=self._bm.dtype,
        )

    def schur_update(self, Ainv_u, v):
        Ainv_B = self._bm._A.inv() @ self._bm._B

        S = self._bm._D - self._bm._C @ Ainv_B

        y = S.inv() @ (v - self._bm._C @ Ainv_u)
        x = Ainv_u - Ainv_B @ y

        return np.concatenate((x, y), axis=0)
