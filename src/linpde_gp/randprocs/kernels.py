import jax
import numpy as np
import probnum as pn


class JaxKernel(pn.randprocs.kernels.Kernel):
    def __init__(self, k, input_dim: int, vectorize: bool = True):
        if vectorize:
            k = jax.numpy.vectorize(k, signature="(d),(d)->()")

        self._k = k

        super().__init__(input_dim=input_dim)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        if x1 is None:
            x1 = x0

        kernmat = self._k(x0, x1)

        return np.array(kernmat)

    @property
    def jax(self):
        return self._k
