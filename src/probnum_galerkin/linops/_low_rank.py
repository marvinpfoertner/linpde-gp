from functools import cached_property
from typing import Optional

import numpy as np
import probnum as pn


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

        self._V = V if V is not None else self._U.adjoint()

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
            self._A.shape,
            dtype=np.result_type(
                self._A.dtype, self._U.dtype, self._C.dtype, self._V.dtype
            ),
            matmul=lambda x: self._AplusUCV @ x,
            rmatmul=lambda x: x @ self._AplusUCV,
            apply=self._AplusUCV.__call__,
            todense=self._AplusUCV.todense,
            conjugate=self._AplusUCV.conjugate,
            transpose=self._AplusUCV.transpose,
            adjoint=self._AplusUCV.adjoint,
            inverse=inverse,
            det=det,
        )

    @cached_property
    def _schur_complement(self) -> pn.linops.LinearOperator:
        return self._C.inv() + self._V @ self._A.inv() @ self._U
