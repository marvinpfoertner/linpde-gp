import functools
from typing import Optional, Union

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike, ShapeType

from ._jax import JaxKernel


class ExpQuad(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeLike,
        lengthscales: Union[ArrayLike, pn.linops.LinearOperatorLike] = 1.0,
        output_scale: float = 1.0,
    ):
        super().__init__(input_shape, output_shape=())

        if np.ndim(lengthscales) == 0:
            self._lengthscales = np.asarray(lengthscales, dtype=np.double)
        elif np.ndim(lengthscales) == 1:
            self._lengthscales = np.asarray(lengthscales, dtype=np.double)

            if self._lengthscales.shape != self.input_shape:
                raise ValueError()
        else:
            assert np.ndim(lengthscales) == 2

            self._lengthscales = pn.linops.aslinop(lengthscales)

            if self._lengthscales.shape != 2 * self.input_shape:
                raise ValueError()

        self._output_scale = np.asarray(output_scale, dtype=np.double)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                self._output_scale ** 2,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        if self._lengthscales.ndim <= 1:
            square_dists_Lambda_sq_inv = (x0 - x1) / self._lengthscales
        else:
            square_dists_Lambda_sq_inv = self._lengthscales.inv()(x0 - x1, axis=-1)

        square_dists_Lambda_sq_inv = square_dists_Lambda_sq_inv ** 2

        if self.input_ndim > 0:
            assert self.input_ndim == 1

            square_dists_Lambda_sq_inv = np.sum(square_dists_Lambda_sq_inv, axis=-1)

        return self._output_scale ** 2 * np.exp(-0.5 * square_dists_Lambda_sq_inv)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                self._output_scale ** 2,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        if self._lengthscales.ndim <= 1:
            square_dists_Lambda_sq_inv = (x0 - x1) / self._lengthscales
        else:
            # TODO: Remove the `.todense` here, once `LinearOperator`s support Jax
            lambda_inv = self._lengthscales.inv().todense()

            square_dists_Lambda_sq_inv = (lambda_inv @ (x0 - x1)[..., None])[..., 0]

        square_dists_Lambda_sq_inv = square_dists_Lambda_sq_inv ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists_Lambda_sq_inv = jnp.sum(square_dists_Lambda_sq_inv, axis=-1)

        return self._output_scale ** 2 * jnp.exp(-0.5 * square_dists_Lambda_sq_inv)
