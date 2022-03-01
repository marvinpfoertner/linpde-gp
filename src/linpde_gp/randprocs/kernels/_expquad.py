import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike, ShapeLike

from ...problems.pde.diffops import ScaledLaplaceOperator
from ._jax import JaxKernel


class ExpQuad(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeLike,
        lengthscales: ArrayLike = 1.0,
        output_scale: float = 1.0,
    ):
        super().__init__(input_shape, output_shape=())

        self._lengthscales = np.asarray(lengthscales, dtype=np.double)
        self._output_scale = np.asarray(output_scale, dtype=np.double)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                self._output_scale**2,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        square_dists = ((x0 - x1) / self._lengthscales) ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists = np.sum(square_dists, axis=-1)

        return self._output_scale**2 * np.exp(-0.5 * square_dists)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                self._output_scale**2,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        square_dists = ((x0 - x1) / self._lengthscales) ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists = jnp.sum(square_dists, axis=-1)

        return self._output_scale**2 * jnp.exp(-0.5 * square_dists)


@ScaledLaplaceOperator.__call__.register
def _(self, f: ExpQuad, *, argnum: int = 0, **kwargs):
    return ExpQuadLaplacianCross(
        input_shape=f.input_shape,
        argnum=argnum,
        alpha=self._alpha,
        lengthscale=f._lengthscales,
        output_scale=f._output_scale,
    )


class ExpQuadLaplacianCross(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeLike,
        argnum: int,
        alpha: float,
        lengthscale: float = 1.0,
        output_scale: float = 1.0,
    ):
        super().__init__(input_shape, output_shape=())

        assert argnum in (0, 1)

        self._argnum = argnum

        self._alpha = alpha

        self._lengthscale = float(lengthscale)
        self._output_scale = float(output_scale)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        new_output_scale = self._alpha * (self._output_scale / self._lengthscale) ** 2

        d = 1 if self.input_shape == () else self.input_shape[0]

        if x1 is None:
            return np.full_like(
                x0,
                new_output_scale * (-d),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        square_dists = ((x0 - x1) / self._lengthscale) ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists = np.sum(square_dists, axis=-1)

        return new_output_scale * (square_dists - d) * np.exp(-0.5 * square_dists)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        new_output_scale = self._alpha * (self._output_scale / self._lengthscale) ** 2

        d = 1 if self.input_shape == () else self.input_shape[0]

        if x1 is None:
            return jnp.full_like(
                x0,
                new_output_scale * (-d),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        square_dists = ((x0 - x1) / self._lengthscale) ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists = jnp.sum(square_dists, axis=-1)

        return new_output_scale * (square_dists - d) * jnp.exp(-0.5 * square_dists)


@ScaledLaplaceOperator.__call__.register
def _(self, f: ExpQuadLaplacianCross, *, argnum: int = 0, **kwargs):
    if argnum != f._argnum:
        if argnum == 0:
            alpha0 = self._alpha
            alpha1 = f._alpha
        else:
            alpha0 = f._alpha
            alpha1 = self._alpha

        return ExpQuadLaplacian(
            input_shape=f.input_shape,
            alpha0=alpha0,
            alpha1=alpha1,
            lengthscale=f._lengthscale,
            output_scale=f._output_scale,
        )

    return super(ScaledLaplaceOperator, self).__call__(f, argnum=argnum, **kwargs)


class ExpQuadLaplacian(JaxKernel):
    def __init__(
        self,
        input_shape: int,
        alpha0: float,
        alpha1: float,
        lengthscale: float = 1.0,
        output_scale: float = 1.0,
    ):
        super().__init__(input_shape, output_shape=())

        self._alpha0 = alpha0
        self._alpha1 = alpha1

        self._lengthscale = float(lengthscale)
        self._output_scale = float(output_scale)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        new_output_scale = (
            self._alpha0
            * self._alpha1
            * (self._output_scale / self._lengthscale**2) ** 2
        )
        d = 1 if self.input_shape == () else self.input_shape[0]

        if x1 is None:
            return np.full_like(
                x0,
                new_output_scale * d * (d + 2),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        square_dists = ((x0 - x1) / self._lengthscale) ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists = np.sum(square_dists, axis=-1)

        return (
            new_output_scale
            * (square_dists**2 - 2 * (d + 2) * square_dists + d * (d + 2))
            * np.exp(-0.5 * square_dists)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        new_output_scale = (
            self._alpha0
            * self._alpha1
            * (self._output_scale / self._lengthscale**2) ** 2
        )
        d = 1 if self.input_shape == () else self.input_shape[0]

        if x1 is None:
            return jnp.full_like(
                x0,
                new_output_scale * d * (d + 2),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        square_dists = ((x0 - x1) / self._lengthscale) ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists = jnp.sum(square_dists, axis=-1)

        return (
            new_output_scale
            * (square_dists**2 - 2 * (d + 2) * square_dists + d * (d + 2))
            * jnp.exp(-0.5 * square_dists)
        )
