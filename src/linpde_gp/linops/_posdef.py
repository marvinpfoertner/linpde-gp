import functools

import numpy as np
import probnum as pn
import scipy.linalg


class PDMatrix(pn.linops.LinearOperator):
    pass


class InversePDMatrix(pn.linops.LinearOperator):
    def __init__(self, A):
        self._A = pn.linops.aslinop(A)

        super().__init__(
            shape=self._A.shape,
            dtype=self._A.dtype,
            matmul=pn.linops.LinearOperator.broadcast_matmat(
                lambda x: scipy.linalg.cho_solve(self._A_cho, x)
            ),
        )

    @functools.cached_property
    def _A_cho(self):
        return scipy.linalg.cho_factor(self._A.todense())
