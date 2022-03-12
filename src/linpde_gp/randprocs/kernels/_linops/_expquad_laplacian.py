import functools
from typing import Optional, Union

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ShapeType

from ....problems.pde.diffops import ScaledLaplaceOperator
from .._expquad import ExpQuad
from .._jax import JaxKernel


class ExpQuadLaplacianCross(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeType,
        argnum: int,
        alpha: float,
        lengthscales: Union[np.ndarray, pn.linops.LinearOperator],
        output_scale: np.ndarray,
    ):
        super().__init__(input_shape, output_shape=())

        assert argnum in (0, 1)

        self._argnum = argnum

        self._alpha = alpha

        self._lengthscales = lengthscales
        self._output_scale = output_scale

    @functools.cached_property
    def _trace_Lambda_sq_inv(self):
        if self._lengthscales.ndim == 0:
            d = 1 if self.input_shape == () else self.input_shape[0]

            return d / self._lengthscales ** 2
        elif self._lengthscales.ndim == 1:
            return np.sum(1 / self._lengthscales ** 2)

        return (self._lengthscales.inv().T @ self._lengthscales.inv()).trace()

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                self._alpha * self._output_scale ** 2 * (-self._trace_Lambda_sq_inv),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        if self._lengthscales.ndim <= 1:
            square_dists_Lambda_sq_inv = (x0 - x1) / self._lengthscales
            square_dists_Lambda_4_inv = square_dists_Lambda_sq_inv / self._lengthscales
        else:
            square_dists_Lambda_sq_inv = self._lengthscales.inv()(x0 - x1, axis=-1)
            square_dists_Lambda_4_inv = self._lengthscales.inv().T(
                square_dists_Lambda_sq_inv, axis=-1
            )

        square_dists_Lambda_sq_inv = square_dists_Lambda_sq_inv ** 2
        square_dists_Lambda_4_inv = square_dists_Lambda_4_inv ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists_Lambda_sq_inv = np.sum(square_dists_Lambda_sq_inv, axis=-1)
            square_dists_Lambda_4_inv = np.sum(square_dists_Lambda_4_inv, axis=-1)

        return (
            self._alpha
            * self._output_scale ** 2
            * (square_dists_Lambda_4_inv - self._trace_Lambda_sq_inv)
            * np.exp(-0.5 * square_dists_Lambda_sq_inv)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                self._alpha * self._output_scale ** 2 * (-self._trace_Lambda_sq_inv),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        if self._lengthscales.ndim <= 1:
            square_dists_Lambda_sq_inv = (x0 - x1) / self._lengthscales
            square_dists_Lambda_4_inv = square_dists_Lambda_sq_inv / self._lengthscales
        else:
            # TODO: Remove the `.todense` here, once `LinearOperator`s support Jax
            lambda_inv = self._lengthscales.inv().todense()

            square_dists_Lambda_sq_inv = lambda_inv @ (x0 - x1)[..., None]
            square_dists_Lambda_4_inv = lambda_inv.T @ square_dists_Lambda_sq_inv

            square_dists_Lambda_sq_inv = square_dists_Lambda_sq_inv[..., 0]
            square_dists_Lambda_4_inv = square_dists_Lambda_4_inv[..., 0]

        square_dists_Lambda_sq_inv = square_dists_Lambda_sq_inv ** 2
        square_dists_Lambda_4_inv = square_dists_Lambda_4_inv ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists_Lambda_sq_inv = np.sum(square_dists_Lambda_sq_inv, axis=-1)
            square_dists_Lambda_4_inv = np.sum(square_dists_Lambda_4_inv, axis=-1)

        return (
            self._alpha
            * self._output_scale ** 2
            * (square_dists_Lambda_4_inv - self._trace_Lambda_sq_inv)
            * jnp.exp(-0.5 * square_dists_Lambda_sq_inv)
        )


@ScaledLaplaceOperator.__call__.register
def _(self, f: ExpQuad, *, argnum: int = 0, **kwargs):
    return ExpQuadLaplacianCross(
        input_shape=f.input_shape,
        argnum=argnum,
        alpha=self._alpha,
        lengthscales=f._lengthscales,
        output_scale=f._output_scale,
    )


class ExpQuadLaplacian(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeType,
        alpha0: float,
        alpha1: float,
        lengthscales: Union[np.ndarray, pn.linops.LinearOperator],
        output_scale: np.ndarray,
    ):
        super().__init__(input_shape, output_shape=())

        self._alpha0 = alpha0
        self._alpha1 = alpha1

        self._lengthscales = lengthscales
        self._output_scale = output_scale

    @functools.cached_property
    def _trace_Lambda_sq_inv(self):
        if self._lengthscales.ndim == 0:
            d = 1 if self.input_shape == () else self.input_shape[0]

            return d / self._lengthscales ** 2
        elif self._lengthscales.ndim == 1:
            return np.sum(1 / self._lengthscales ** 2)

        return (self._lengthscales.inv().T @ self._lengthscales.inv()).trace()

    @functools.cached_property
    def _trace_Lambda_4_inv(self):
        if self._lengthscales.ndim == 0:
            d = 1 if self.input_shape == () else self.input_shape[0]

            return d / self._lengthscales ** 4
        elif self._lengthscales.ndim == 1:
            return np.sum(1 / self._lengthscales ** 4)

        return (
            self._lengthscales.inv().T
            @ self._lengthscales.inv()
            @ self._lengthscales.inv().T
            @ self._lengthscales.inv()
        ).trace()

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                self._alpha0
                * self._alpha1
                * self._output_scale ** 2
                * (self._trace_Lambda_sq_inv ** 2 + 2 * self._trace_Lambda_4_inv),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        if self._lengthscales.ndim <= 1:
            square_dists_Lambda_sq_inv = (x0 - x1) / self._lengthscales
            square_dists_Lambda_4_inv = square_dists_Lambda_sq_inv / self._lengthscales
            square_dists_Lambda_6_inv = square_dists_Lambda_4_inv / self._lengthscales
        else:
            square_dists_Lambda_sq_inv = self._lengthscales.inv()(x0 - x1, axis=-1)
            square_dists_Lambda_4_inv = self._lengthscales.inv().T(
                square_dists_Lambda_sq_inv, axis=-1
            )
            square_dists_Lambda_6_inv = self._lengthscales.inv()(
                square_dists_Lambda_4_inv, axis=-1
            )

        square_dists_Lambda_sq_inv = square_dists_Lambda_sq_inv ** 2
        square_dists_Lambda_4_inv = square_dists_Lambda_4_inv ** 2
        square_dists_Lambda_6_inv = square_dists_Lambda_6_inv ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists_Lambda_sq_inv = np.sum(square_dists_Lambda_sq_inv, axis=-1)
            square_dists_Lambda_4_inv = np.sum(square_dists_Lambda_4_inv, axis=-1)
            square_dists_Lambda_6_inv = np.sum(square_dists_Lambda_6_inv, axis=-1)

        return (
            self._alpha0
            * self._alpha1
            * self._output_scale ** 2
            * (
                (square_dists_Lambda_4_inv - self._trace_Lambda_sq_inv) ** 2
                - 4 * square_dists_Lambda_6_inv
                + 2 * self._trace_Lambda_4_inv
            )
            * np.exp(-0.5 * square_dists_Lambda_sq_inv)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                self._alpha0
                * self._alpha1
                * self._output_scale ** 2
                * (self._trace_Lambda_sq_inv ** 2 + 2 * self._trace_Lambda_4_inv),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        if self._lengthscales.ndim <= 1:
            square_dists_Lambda_sq_inv = (x0 - x1) / self._lengthscales
            square_dists_Lambda_4_inv = square_dists_Lambda_sq_inv / self._lengthscales
            square_dists_Lambda_6_inv = square_dists_Lambda_4_inv / self._lengthscales
        else:
            square_dists_Lambda_sq_inv = self._lengthscales.inv()(x0 - x1, axis=-1)
            square_dists_Lambda_4_inv = self._lengthscales.inv().T(
                square_dists_Lambda_sq_inv, axis=-1
            )
            square_dists_Lambda_6_inv = self._lengthscales.inv()(
                square_dists_Lambda_4_inv, axis=-1
            )

        square_dists_Lambda_sq_inv = square_dists_Lambda_sq_inv ** 2
        square_dists_Lambda_4_inv = square_dists_Lambda_4_inv ** 2
        square_dists_Lambda_6_inv = square_dists_Lambda_6_inv ** 2

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            square_dists_Lambda_sq_inv = np.sum(square_dists_Lambda_sq_inv, axis=-1)
            square_dists_Lambda_4_inv = np.sum(square_dists_Lambda_4_inv, axis=-1)
            square_dists_Lambda_6_inv = np.sum(square_dists_Lambda_6_inv, axis=-1)

        return (
            self._alpha0
            * self._alpha1
            * self._output_scale ** 2
            * (
                (square_dists_Lambda_4_inv - self._trace_Lambda_sq_inv) ** 2
                - 4 * square_dists_Lambda_6_inv
                + 2 * self._trace_Lambda_4_inv
            )
            * np.exp(-0.5 * square_dists_Lambda_sq_inv)
        )


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
            lengthscales=f._lengthscales,
            output_scale=f._output_scale,
        )

    return super(ScaledLaplaceOperator, self).__call__(f, argnum=argnum, **kwargs)
