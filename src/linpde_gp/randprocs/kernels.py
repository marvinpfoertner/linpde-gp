from typing import Optional

import jax
import numpy as np
import probnum as pn
from jax import numpy as jnp
from probnum.typing import ArrayLike


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


class ExpQuad(JaxKernel):
    def __init__(self, input_dim: int, lengthscales: ArrayLike, output_scale: float):
        self._lengthscales = np.asarray(lengthscales, dtype=np.double)
        self._output_scale = np.asarray(output_scale, dtype=np.double)

        super().__init__(self._jax, input_dim, vectorize=False)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            x1 = x0

        square_dists = np.sum((x0 - x1) / self._lengthscales, axis=-1)

        return self._output_scale * np.exp(-0.5 * square_dists)

    @jax.jit(static_argnums=0)
    def _jax(self, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        square_dists = jnp.sum((x0 - x1) / self._lengthscales, axis=-1)

        return self._output_scale * jnp.exp(-0.5 * square_dists)
