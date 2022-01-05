from functools import cached_property
from typing import Optional

import numpy as np
import probnum as pn
import scipy.linalg
from scipy.linalg.decomp import eigvals


def outer(u: np.ndarray, v: np.ndarray) -> pn.linops.LinearOperator:
    return pn.linops.aslinop(u[:, None]) @ pn.linops.aslinop(v[None, :])


class LowRankUpdate(pn.linops.LinearOperator):
    r""":math:`M := A + U C V`"""

    def __init__(
        self,
        A: pn.linops.LinearOperatorLike,
        U: pn.linops.LinearOperatorLike,
        C: Optional[pn.linops.LinearOperatorLike] = None,
        V: Optional[pn.linops.LinearOperatorLike] = None,
    ):
        self._A = pn.linops.aslinop(A)

        self._U = pn.linops.aslinop(U)

        if C is not None:
            C = pn.linops.aslinop(C)

        self._C = C if C is not None else pn.linops.Identity(self._U.shape[1])

        if V is not None:
            V = pn.linops.aslinop(V)

        self._V = V if V is not None else self._U.transpose()

        UV_equal = V is None

        self._AplusUCV = self._A + self._U @ self._C @ self._V

        # Inverse by matrix inversion lemma
        inverse = lambda: LowRankUpdate(
            A=self._A.inv(),
            U=self._A.inv() @ self._U,
            C=-self._schur_complement.inv(),
            V=self._V @ self._A.inv() if not UV_equal else None,
        )

        # Determinant by matrix determinant lemma
        det = lambda: self._A.det() * self._C.det() * self._schur_complement.det()

        super().__init__(
            self._AplusUCV.shape,
            dtype=self._AplusUCV.dtype,
            matmul=lambda x: self._AplusUCV @ x,
            rmatmul=lambda x: x @ self._AplusUCV,
            apply=self._AplusUCV.__call__,
            todense=self._AplusUCV.todense,
            transpose=self._AplusUCV.transpose,
            inverse=inverse,
            det=det,
        )

    @cached_property
    def _schur_complement(self) -> pn.linops.LinearOperator:
        return self._C.inv() + self._V @ self._A.inv() @ self._U


class LowRankMatrix(pn.linops.LinearOperator):
    def __init__(
        self,
        U: pn.linops.LinearOperatorLike,
    ):
        self._U = pn.linops.aslinop(U)

        self._linop = self._U @ self._U.T

        super().__init__(
            self._linop.shape,
            self._linop.dtype,
            matmul=self._linop.__matmul__,
            rmatmul=self._linop.__rmatmul__,
            apply=self._linop.__call__,
            todense=self._linop.todense,
            transpose=lambda: self,
        )

    @cached_property
    def svd(self) -> pn.linops.LinearOperator:
        Q1, R = scipy.linalg.qr(self._U.todense(), mode="economic")

        Q2, sqrt_svals, _ = scipy.linalg.svd(R)

        U_svd = pn.linops.aslinop(Q1) @ pn.linops.aslinop(Q2)
        svals = sqrt_svals ** 2

        return U_svd, svals, U_svd

    @cached_property
    def pinv(self) -> pn.linops.LinearOperator:
        U, svals, _ = self.svd

        pinv_svals = np.divide(
            1.0,
            svals,
            where=svals > 0,
            out=np.zeros_like(svals),
        )

        return LowRankMatrix(U @ pn.linops.Scaling(np.sqrt(pinv_svals)))
