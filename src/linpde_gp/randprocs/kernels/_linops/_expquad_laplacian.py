import functools
from typing import Optional, Union

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ShapeType

from linpde_gp.linfuncops.diffops import (
    DirectionalDerivative,
    ScaledLaplaceOperator,
    ScaledSpatialLaplacian,
)

from .._expquad import ExpQuad
from .._jax import JaxKernel
from ._expquad_directional_derivative import ExpQuadDirectionalDerivativeCross


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
def _(self, k: ExpQuad, /, *, argnum: int = 0):
    return ExpQuadLaplacianCross(
        input_shape=k.input_shape,
        argnum=argnum,
        alpha=self._alpha,
        lengthscales=k._lengthscales,
        output_scale=k._output_scale,
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
def _(self, k: ExpQuadLaplacianCross, /, *, argnum: int = 0):
    if argnum != k._argnum:
        if argnum == 0:
            alpha0 = self._alpha
            alpha1 = k._alpha
        else:
            alpha0 = k._alpha
            alpha1 = self._alpha

        return ExpQuadLaplacian(
            input_shape=k.input_shape,
            alpha0=alpha0,
            alpha1=alpha1,
            lengthscales=k._lengthscales,
            output_scale=k._output_scale,
        )

    return super(ScaledLaplaceOperator, self).__call__(k, argnum=argnum)


class ExpQuadDirectionalDerivativeLaplacian(JaxKernel):
    def __init__(
        self,
        expquad_laplacian: ExpQuadLaplacianCross,
        direction: np.ndarray,
    ):
        self._expquad_laplacian = expquad_laplacian
        self._expquad = ExpQuad(
            input_shape=self._expquad_laplacian.input_shape,
            lengthscales=self._expquad_laplacian._lengthscales,
            output_scale=self._expquad_laplacian._output_scale,
        )

        super().__init__(
            self._expquad_laplacian.input_shape,
            output_shape=self._expquad_laplacian.output_shape,
        )

        self._direction = direction

        self._sign = 1.0 if self._expquad_laplacian._argnum == 1 else -1.0

    @functools.cached_property
    def _direction_Lambda_sq_inv(self) -> np.ndarray:
        return self._direction / self._expquad._lengthscales ** 2

    @functools.cached_property
    def _direction_Lambda_4_inv(self) -> np.ndarray:
        return self._direction_Lambda_sq_inv / self._expquad._lengthscales ** 2

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(
                x0,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        k_x0_x1 = self._expquad(x0, x1)
        laplacian_k_x0_x1 = self._expquad_laplacian(x0, x1)

        diffs = x0 - x1

        direction_diffs_inprod_Lambda_sq_inv = self._direction_Lambda_sq_inv * diffs
        direction_diffs_inprod_Lambda_4_inv = self._direction_Lambda_4_inv * diffs

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            direction_diffs_inprod_Lambda_sq_inv = np.sum(
                direction_diffs_inprod_Lambda_sq_inv, axis=-1
            )
            direction_diffs_inprod_Lambda_4_inv = np.sum(
                direction_diffs_inprod_Lambda_4_inv, axis=-1
            )

        return self._sign * (
            2
            * self._expquad_laplacian._alpha
            * direction_diffs_inprod_Lambda_4_inv
            * k_x0_x1
            - direction_diffs_inprod_Lambda_sq_inv * laplacian_k_x0_x1
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(
                x0,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        k_x0_x1 = self._expquad.jax(x0, x1)
        laplacian_k_x0_x1 = self._expquad_laplacian.jax(x0, x1)

        diffs = x0 - x1

        direction_diffs_inprod_Lambda_sq_inv = self._direction_Lambda_sq_inv * diffs
        direction_diffs_inprod_Lambda_4_inv = self._direction_Lambda_4_inv * diffs

        if self._input_ndim > 0:
            assert self._input_ndim == 1

            direction_diffs_inprod_Lambda_sq_inv = jnp.sum(
                direction_diffs_inprod_Lambda_sq_inv, axis=-1
            )
            direction_diffs_inprod_Lambda_4_inv = jnp.sum(
                direction_diffs_inprod_Lambda_4_inv, axis=-1
            )

        return self._sign * (
            2
            * self._expquad_laplacian._alpha
            * direction_diffs_inprod_Lambda_4_inv
            * k_x0_x1
            + direction_diffs_inprod_Lambda_sq_inv * laplacian_k_x0_x1
        )


@ScaledLaplaceOperator.__call__.register
def _(self, k: ExpQuadDirectionalDerivativeCross, /, *, argnum: int = 0):
    if k._argnum != argnum:
        return ExpQuadDirectionalDerivativeLaplacian(
            expquad_laplacian=ExpQuadLaplacianCross(
                input_shape=k.input_shape,
                argnum=argnum,
                alpha=self._alpha,
                lengthscales=k._expquad_lengthscales,
                output_scale=k._expquad_output_scale,
            ),
            direction=k._direction,
        )

    return super(ScaledLaplaceOperator, self).__call__(k, argnum=argnum)


@DirectionalDerivative.__call__.register
def _(self, k: ExpQuadLaplacianCross, /, *, argnum: int = 0):
    if k._argnum != argnum:
        return ExpQuadDirectionalDerivativeLaplacian(
            expquad_laplacian=k,
            direction=self._direction,
        )

    return super(DirectionalDerivative, self).__call__(k, argnum=argnum)


class ExpQuadSpatialLaplacianCross(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeType,
        argnum: int,
        alpha: float,
        lengthscales: Union[np.ndarray, pn.linops.LinearOperator],
        output_scale: np.ndarray,
    ):
        super().__init__(input_shape, output_shape=())

        (D,) = self._input_shape
        assert D > 1

        assert argnum in (0, 1)

        self._argnum = argnum

        self._alpha = alpha

        self._lengthscales = lengthscales
        self._output_scale = output_scale

        self._expquad_time = ExpQuad(
            input_shape=(),
            lengthscales=self._lengthscales
            if self._lengthscales.ndim == 0
            else self._lengthscales[0],
            output_scale=self._output_scale,
        )
        self._laplacian_expquad_space = ExpQuadLaplacianCross(
            input_shape=(D - 1,),
            argnum=self._argnum,
            alpha=self._alpha,
            lengthscales=self._lengthscales
            if self._lengthscales.ndim == 0
            else self._lengthscales[1:],
            output_scale=1.0,
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._expquad_time(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._laplacian_expquad_space(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._expquad_time.jax(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._laplacian_expquad_space.jax(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )


@ScaledSpatialLaplacian.__call__.register
def _(self, k: ExpQuad, /, *, argnum: int = 0):
    return ExpQuadSpatialLaplacianCross(
        input_shape=k.input_shape,
        argnum=argnum,
        alpha=self._alpha,
        lengthscales=k._lengthscales,
        output_scale=k._output_scale,
    )


class ExpQuadSpatialLaplacianBoth(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeType,
        alpha0: float,
        alpha1: float,
        lengthscales: Union[np.ndarray, pn.linops.LinearOperator],
        output_scale: np.ndarray,
    ):
        super().__init__(input_shape, output_shape=())

        (D,) = self._input_shape
        assert D > 1

        self._alpha0 = alpha0
        self._alpha1 = alpha1

        self._lengthscales = lengthscales
        self._output_scale = output_scale

        self._expquad_time = ExpQuad(
            input_shape=(),
            lengthscales=self._lengthscales
            if self._lengthscales.ndim == 0
            else self._lengthscales[0],
            output_scale=self._output_scale,
        )
        self._laplacian_expquad_space = ExpQuadLaplacian(
            input_shape=(D - 1,),
            alpha0=self._alpha0,
            alpha1=self._alpha1,
            lengthscales=self._lengthscales
            if self._lengthscales.ndim == 0
            else self._lengthscales[1:],
            output_scale=1.0,
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._expquad_time(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._laplacian_expquad_space(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._expquad_time.jax(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._laplacian_expquad_space.jax(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )


@ScaledSpatialLaplacian.__call__.register
def _(self, k: ExpQuadSpatialLaplacianCross, /, *, argnum: int = 0):
    if argnum != k._argnum:
        if argnum == 0:
            alpha0 = self._alpha
            alpha1 = k._alpha
        else:
            alpha0 = k._alpha
            alpha1 = self._alpha

        return ExpQuadSpatialLaplacianBoth(
            input_shape=k.input_shape,
            alpha0=alpha0,
            alpha1=alpha1,
            lengthscales=k._lengthscales,
            output_scale=k._output_scale,
        )

    return super(ScaledLaplaceOperator, self).__call__(k, argnum=argnum)


class ExpQuadDirectionalDerivativeSpatialLaplacian(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeType,
        lengthscales: np.ndarray,
        output_scale: np.ndarray,
        direction: np.ndarray,
        alpha: float,
        reverse: bool = False,
    ):
        super().__init__(input_shape, output_shape=())

        (D,) = self._input_shape
        assert D > 1

        self._k_temporal_directional_derivative = ExpQuadDirectionalDerivativeCross(
            input_shape=(),
            argnum=1 if reverse else 0,
            lengthscales=lengthscales[0] if lengthscales.ndim > 0 else lengthscales,
            output_scale=1.0,
            direction=direction[0],
        )

        self._k_spatial_laplacian = ExpQuadLaplacianCross(
            input_shape=(D - 1,),
            argnum=1 if reverse else 0,
            alpha=alpha,
            lengthscales=lengthscales[1:] if lengthscales.ndim > 0 else lengthscales,
            output_scale=output_scale,
        )

        self._k_temporal = ExpQuad(
            input_shape=(),
            lengthscales=lengthscales[0] if lengthscales.ndim > 0 else lengthscales,
            output_scale=output_scale,
        )

        self._k_spatial_cross = ExpQuadDirectionalDerivativeLaplacian(
            expquad_laplacian=ExpQuadLaplacianCross(
                input_shape=(D - 1,),
                argnum=0 if reverse else 1,
                alpha=alpha,
                lengthscales=(
                    lengthscales[1:] if lengthscales.ndim > 0 else lengthscales
                ),
                output_scale=1.0,
            ),
            direction=direction[1:],
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return (
            self._k_temporal_directional_derivative(
                x0[..., 0], None if x1 is None else x1[..., 0]
            )
            * self._k_spatial_laplacian(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        ) + (
            self._k_temporal(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._k_spatial_cross(x0[..., 1:], None if x1 is None else x1[..., 1:])
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return (
            self._k_temporal_directional_derivative.jax(
                x0[..., 0], None if x1 is None else x1[..., 0]
            )
            * self._k_spatial_laplacian.jax(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        ) + (
            self._k_temporal.jax(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._k_spatial_cross.jax(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        )


@ScaledSpatialLaplacian.__call__.register
def _(self, k: ExpQuadDirectionalDerivativeCross, /, *, argnum: int = 0):
    if k._argnum != argnum:
        return ExpQuadDirectionalDerivativeSpatialLaplacian(
            input_shape=k.input_shape,
            lengthscales=k._expquad_lengthscales,
            output_scale=k._expquad_output_scale,
            direction=k._direction,
            alpha=self._alpha,
            reverse=(argnum == 0),
        )

    return super(ScaledLaplaceOperator, self).__call__(k, argnum=argnum)


@DirectionalDerivative.__call__.register
def _(self, k: ExpQuadSpatialLaplacianCross, /, *, argnum: int = 0):
    if k._argnum != argnum:
        return ExpQuadDirectionalDerivativeSpatialLaplacian(
            input_shape=k.input_shape,
            lengthscales=k._lengthscales,
            output_scale=k._output_scale,
            direction=self._direction,
            alpha=k._alpha,
            reverse=(argnum == 1),
        )

    return super(DirectionalDerivative, self).__call__(k, argnum=argnum)
