import functools
from typing import Optional

import jax
import numpy as np
from jax import numpy as jnp
from probnum.typing import ArrayLike

from ...problems.pde.diffops import LaplaceOperator
from ._jax import JaxKernel


class ExpQuad(JaxKernel):
    def __init__(
        self,
        input_dim: int,
        lengthscales: ArrayLike = 1.0,
        output_scale: float = 1.0,
    ):
        self._lengthscales = np.asarray(lengthscales, dtype=np.double)
        self._output_scale = np.asarray(output_scale, dtype=np.double)

        super().__init__(self._jax, input_dim, vectorize=False)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            x1 = x0

        square_dists = np.sum(((x0 - x1) / self._lengthscales) ** 2, axis=-1)

        return self._output_scale * np.exp(-0.5 * square_dists)

    @functools.partial(jax.jit, static_argnums=0)
    def _jax(self, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        square_dists = jnp.sum(((x0 - x1) / self._lengthscales) ** 2, axis=-1)

        return self._output_scale ** 2 * jnp.exp(-0.5 * square_dists)


class ExpQuadLaplacianCross(JaxKernel):
    def __init__(
        self,
        input_dim: int,
        argnum: int,
        lengthscale: float = 1.0,
        output_scale: float = 1.0,
    ):
        assert argnum in (0, 1)

        self._argnum = argnum

        self._lengthscale = float(lengthscale)
        self._output_scale = float(output_scale)

        super().__init__(self._jax, input_dim, vectorize=False)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            x1 = x0

        square_dists = np.sum(((x0 - x1) / self._lengthscale) ** 2, axis=-1)
        d = self.input_dim

        return (
            (self._output_scale / self._lengthscale) ** 2
            * (square_dists - d)
            * np.exp(-0.5 * square_dists)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _jax(self, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        square_dists = jnp.sum(((x0 - x1) / self._lengthscale) ** 2, axis=-1)
        d = self.input_dim

        return (
            self._sign
            * (self._output_scale / self._lengthscale) ** 2
            * (square_dists - d)
            * jnp.exp(-0.5 * square_dists)
        )


class ExpQuadLaplacian(JaxKernel):
    def __init__(
        self,
        input_dim: int,
        lengthscale: float = 1.0,
        output_scale: float = 1.0,
    ):
        self._lengthscale = float(lengthscale)
        self._output_scale = float(output_scale)

        super().__init__(self._jax, input_dim, vectorize=False)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        if x1 is None:
            x1 = x0

        square_dists = np.sum(((x0 - x1) / self._lengthscale) ** 2, axis=-1)
        d = self._input_dim

        return (
            (self._output_scale / self._lengthscale ** 2) ** 2
            * (square_dists ** 2 - 2 * (d + 2) * square_dists + d * (d + 2))
            * np.exp(-0.5 * square_dists)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _jax(self, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
        square_dists = jnp.sum(((x0 - x1) / self._lengthscale) ** 2, axis=-1)
        d = self._input_dim

        return (
            (self._output_scale / self._lengthscale ** 2) ** 2
            * (square_dists ** 2 - 2 * (d + 2) * square_dists + d * (d + 2))
            * jnp.exp(-0.5 * square_dists)
        )


@LaplaceOperator.__call__.register
def _(self, f: ExpQuad, *, argnum: int = 0, **kwargs):
    return ExpQuadLaplacianCross(
        input_dim=f.input_dim,
        argnum=argnum,
        lengthscale=f._lengthscales,
        output_scale=f._output_scale,
    )


@LaplaceOperator.__call__.register
def _(self, f: ExpQuadLaplacianCross, *, argnum: int = 0, **kwargs):
    if argnum != f._argnum:
        return ExpQuadLaplacian(
            input_dim=f.input_dim,
            lengthscale=f._lengthscale,
            output_scale=f._output_scale,
        )

    return super(LaplaceOperator, self).__call__(f, argnum=argnum, **kwargs)
