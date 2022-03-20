from cmath import exp
import functools
from typing import Optional, Union

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ShapeType

from linpde_gp.linfuncops import diffops

from .._expquad import ExpQuad
from .._jax import JaxKernel
from ._expquad_directional_derivative import ExpQuad_Identity_DirectionalDerivative


class ExpQuad_Identity_Laplacian(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        alpha: float,
        reverse: bool = True,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._alpha = alpha

        self._reverse = bool(reverse)

        self._modified_output_scale_sq = self._alpha * self._expquad._output_scale_sq

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _lengthscales_sq(self) -> np.ndarray:
        return self._expquad.lengthscales ** 2

    @functools.cached_property
    def _trace_lengthscales_sq_inv(self):
        if self._expquad.lengthscales.ndim == 0:
            d = self.input_size

            return d / self._lengthscales_sq

        return np.sum(1 / self._lengthscales_sq)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                self._modified_output_scale_sq * (-self._trace_lengthscales_sq_inv),
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        dists_sq_lengthscales_sq_inv = self._batched_euclidean_norm_sq(
            diffs / self._expquad.lengthscales
        )
        dists_sq_lengthscales_4_inv = self._batched_euclidean_norm_sq(
            diffs / self._lengthscales_sq
        )

        return (
            self._modified_output_scale_sq
            * (dists_sq_lengthscales_4_inv - self._trace_lengthscales_sq_inv)
            * np.exp(-0.5 * dists_sq_lengthscales_sq_inv)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                self._modified_output_scale_sq * (-self._trace_lengthscales_sq_inv),
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        dists_sq_lengthscales_sq_inv = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad.lengthscales
        )
        dists_sq_lengthscales_4_inv = self._batched_euclidean_norm_sq_jax(
            diffs / self._lengthscales_sq
        )

        return (
            self._modified_output_scale_sq
            * (dists_sq_lengthscales_4_inv - self._trace_lengthscales_sq_inv)
            * jnp.exp(-0.5 * dists_sq_lengthscales_sq_inv)
        )


@diffops.Laplacian.__call__.register
def _(self, k: ExpQuad, /, *, argnum: int = 0):
    return ExpQuad_Identity_Laplacian(
        expquad=k,
        alpha=self._alpha,
        reverse=(argnum == 0),
    )


class ExpQuad_Laplacian_Laplacian(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        alpha0: float,
        alpha1: float,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._alpha0 = alpha0
        self._alpha1 = alpha1

        self._modified_output_scale_sq = (
            self._alpha0 * self._alpha1 * self._expquad._output_scale_sq
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @functools.cached_property
    def _lengthscales_sq(self) -> np.ndarray:
        return self._expquad.lengthscales ** 2

    @functools.cached_property
    def _lengthscales_3(self) -> np.ndarray:
        return self._lengthscales_sq * self._expquad.lengthscales

    @functools.cached_property
    def _lengthscales_4(self) -> np.ndarray:
        return self._lengthscales_sq ** 2

    @functools.cached_property
    def _trace_lengthscales_sq_inv(self):
        if self._expquad.lengthscales.ndim == 0:
            d = self.input_size

            return d / self._lengthscales_sq

        return np.sum(1 / self._lengthscales_sq)

    @functools.cached_property
    def _trace_lengthscales_4_inv(self):
        if self._expquad._lengthscales.ndim == 0:
            d = self.input_size

            return d / self._lengthscales_4

        return np.sum(1 / self._lengthscales_4)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(
                x0,
                (
                    self._modified_output_scale_sq
                    * (
                        self._trace_lengthscales_sq_inv ** 2
                        + 2 * self._trace_lengthscales_4_inv
                    )
                ),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        dists_sq_lengthscales_sq_inv = self._batched_euclidean_norm_sq(
            diffs / self._expquad.lengthscales
        )
        dists_sq_lengthscales_4_inv = self._batched_euclidean_norm_sq(
            diffs / self._lengthscales_sq
        )
        dists_sq_lengthscales_6_inv = self._batched_euclidean_norm_sq(
            diffs / self._lengthscales_3
        )

        return (
            self._modified_output_scale_sq
            * (
                (dists_sq_lengthscales_4_inv - self._trace_lengthscales_sq_inv) ** 2
                - 4 * dists_sq_lengthscales_6_inv
                + 2 * self._trace_lengthscales_4_inv
            )
            * np.exp(-0.5 * dists_sq_lengthscales_sq_inv)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(
                x0,
                (
                    self._modified_output_scale_sq
                    * (
                        self._trace_lengthscales_sq_inv ** 2
                        + 2 * self._trace_lengthscales_4_inv
                    )
                ),
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        dists_sq_lengthscales_sq_inv = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad.lengthscales
        )
        dists_sq_lengthscales_4_inv = self._batched_euclidean_norm_sq_jax(
            diffs / self._lengthscales_sq
        )
        dists_sq_lengthscales_6_inv = self._batched_euclidean_norm_sq_jax(
            diffs / self._lengthscales_3
        )

        return (
            self._modified_output_scale_sq
            * (
                (dists_sq_lengthscales_4_inv - self._trace_lengthscales_sq_inv) ** 2
                - 4 * dists_sq_lengthscales_6_inv
                + 2 * self._trace_lengthscales_4_inv
            )
            * jnp.exp(-0.5 * dists_sq_lengthscales_sq_inv)
        )


@diffops.Laplacian.__call__.register
def _(self, k: ExpQuad_Identity_Laplacian, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        alpha0 = self._alpha
        alpha1 = k._alpha
    elif argnum == 1 and k.reverse:
        alpha0 = k._alpha
        alpha1 = self._alpha
    else:
        return super(diffops.Laplacian, self).__call__(k, argnum=argnum)

    return ExpQuad_Laplacian_Laplacian(
        expquad=k.expquad,
        alpha0=alpha0,
        alpha1=alpha1,
    )


class ExpQuad_DirectionalDerivative_Laplacian(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction: np.ndarray,
        alpha: float,
        reverse: bool = False,
        expquad_laplacian: Optional[ExpQuad_Identity_Laplacian] = None,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction = direction
        self._alpha = alpha

        self._reverse = bool(reverse)

        if expquad_laplacian is None:
            expquad_laplacian = ExpQuad_Identity_Laplacian(
                self._expquad,
                alpha=self._alpha,
                reverse=self._reverse,
            )

        assert expquad_laplacian._alpha == self._alpha
        assert expquad_laplacian.reverse == self._reverse

        self._expquad_laplacian = expquad_laplacian

        modified_output_scale_sq = self._alpha * self._expquad._output_scale_sq

        if self._reverse:
            modified_output_scale_sq = -modified_output_scale_sq

        self._modified_output_scale_sq = modified_output_scale_sq

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @functools.cached_property
    def _direction_lengthscales_sq_inv(self) -> np.ndarray:
        return self._direction / self._expquad_laplacian._lengthscales_sq

    @functools.cached_property
    def _direction_lengthscales_4_inv(self) -> np.ndarray:
        return (
            self._direction_lengthscales_sq_inv
            / self._expquad_laplacian._lengthscales_sq
        )

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs_direction_lengthscales_sq_inv = self._batched_sum(
            self._direction_lengthscales_sq_inv * diffs
        )
        proj_diffs_direction_lengthscales_4_inv = self._batched_sum(
            self._direction_lengthscales_4_inv * diffs
        )

        dists_sq_lengthscales_sq_inv = self._batched_euclidean_norm_sq(
            diffs / self._expquad.lengthscales
        )
        dists_sq_lengthscales_4_inv = self._batched_euclidean_norm_sq(
            diffs / self._expquad_laplacian._lengthscales_sq
        )

        return (
            self._modified_output_scale_sq
            * (
                2 * proj_diffs_direction_lengthscales_4_inv
                - proj_diffs_direction_lengthscales_sq_inv
                * (
                    dists_sq_lengthscales_4_inv
                    - self._expquad_laplacian._trace_lengthscales_sq_inv
                )
            )
            * np.exp(-0.5 * dists_sq_lengthscales_sq_inv)
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs_direction_lengthscales_sq_inv = self._batched_sum_jax(
            self._direction_lengthscales_sq_inv * diffs
        )
        proj_diffs_direction_lengthscales_4_inv = self._batched_sum_jax(
            self._direction_lengthscales_4_inv * diffs
        )

        dists_sq_lengthscales_sq_inv = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad.lengthscales
        )
        dists_sq_lengthscales_4_inv = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad_laplacian._lengthscales_sq
        )

        return (
            self._modified_output_scale_sq
            * (
                2 * proj_diffs_direction_lengthscales_4_inv
                - proj_diffs_direction_lengthscales_sq_inv
                * (
                    dists_sq_lengthscales_4_inv
                    - self._expquad_laplacian._trace_lengthscales_sq_inv
                )
            )
            * jnp.exp(-0.5 * dists_sq_lengthscales_sq_inv)
        )


@diffops.DirectionalDerivative.__call__.register
def _(self, k: ExpQuad_Identity_Laplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_Laplacian(
            expquad=k.expquad,
            direction=self.direction,
            alpha=k._alpha,
            reverse=(argnum == 1),
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.Laplacian.__call__.register
def _(self, k: ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_Laplacian(
            expquad=k.expquad,
            direction=k.direction,
            alpha=self._alpha,
            reverse=(argnum == 0),
        )

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)


class ExpQuad_Identity_SpatialLaplacian(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        alpha: float,
        reverse: bool,
    ):
        if expquad.input_ndim != 1 or expquad.input_size < 2:
            raise ValueError()

        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._alpha = alpha

        self._reverse = reverse

        self._expquad_time = ExpQuad(
            input_shape=(),
            lengthscales=(
                self._expquad.lengthscales
                if self._expquad.lengthscales.ndim == 0
                else self._expquad.lengthscales[0]
            ),
            output_scale=self._expquad._output_scale,
        )

        self._expquad_space = ExpQuad(
            input_shape=(self.input_size - 1,),
            lengthscales=(
                self._expquad.lengthscales
                if self._expquad.lengthscales.ndim == 0
                else self._expquad.lengthscales[1:]
            ),
            output_scale=1.0,
        )

        self._expquad_laplacian_space = ExpQuad_Identity_Laplacian(
            expquad=self._expquad_space,
            alpha=self._alpha,
            reverse=self._reverse,
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def reverse(self) -> bool:
        return self._reverse

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._expquad_time(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_space(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._expquad_time.jax(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_space.jax(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )


@diffops.SpatialLaplacian.__call__.register
def _(self, k: ExpQuad, /, *, argnum: int = 0):
    return ExpQuad_Identity_SpatialLaplacian(
        expquad=k,
        alpha=self._alpha,
        reverse=(argnum == 0),
    )


class ExpQuad_SpatialLaplacian_SpatialLaplacian(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        alpha0: float,
        alpha1: float,
    ):
        if expquad.input_ndim != 1 or expquad.input_size < 2:
            raise ValueError()

        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._alpha0 = alpha0
        self._alpha1 = alpha1

        self._expquad_time = ExpQuad(
            input_shape=(),
            lengthscales=(
                self._expquad.lengthscales
                if self._expquad.lengthscales.ndim == 0
                else self._expquad.lengthscales[0]
            ),
            output_scale=self._expquad._output_scale,
        )

        self._expquad_laplacian_laplacian_space = ExpQuad_Laplacian_Laplacian(
            expquad=ExpQuad(
                input_shape=(self.input_size - 1,),
                lengthscales=(
                    self._expquad.lengthscales
                    if self._expquad.lengthscales.ndim == 0
                    else self._expquad.lengthscales[1:]
                ),
                output_scale=1.0,
            ),
            alpha0=self._alpha0,
            alpha1=self._alpha1,
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return self._expquad_time(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_laplacian_space(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return self._expquad_time.jax(
            x0[..., 0], None if x1 is None else x1[..., 0]
        ) * self._expquad_laplacian_laplacian_space.jax(
            x0[..., 1:], None if x1 is None else x1[..., 1:]
        )


@diffops.SpatialLaplacian.__call__.register
def _(self, k: ExpQuad_Identity_SpatialLaplacian, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        alpha0 = self._alpha
        alpha1 = k._alpha
    elif argnum == 1 and k.reverse:
        alpha0 = k._alpha
        alpha1 = self._alpha
    else:
        return super(diffops.Laplacian, self).__call__(k, argnum=argnum)

    return ExpQuad_SpatialLaplacian_SpatialLaplacian(
        expquad=k.expquad,
        alpha0=alpha0,
        alpha1=alpha1,
    )


class ExpQuad_DirectionalDerivative_SpatialLaplacian(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction: np.ndarray,
        alpha: float,
        reverse: bool = False,
        expquad_spatial_laplacian: Optional[ExpQuad_Identity_SpatialLaplacian] = None,
    ):
        if expquad.input_ndim != 1 or expquad.input_size < 2:
            raise ValueError()

        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction = direction
        self._alpha = alpha

        self._reverse = bool(reverse)

        if expquad_spatial_laplacian is None:
            expquad_spatial_laplacian = ExpQuad_Identity_SpatialLaplacian(
                expquad=self._expquad,
                alpha=self._alpha,
                reverse=self._reverse,
            )

        assert expquad_spatial_laplacian._alpha == self._alpha
        assert expquad_spatial_laplacian._reverse == self._reverse

        self._expquad_spatial_laplacian = expquad_spatial_laplacian

        self._expquad_time = self._expquad_spatial_laplacian._expquad_time

        self._expquad_ddv_time = ExpQuad_Identity_DirectionalDerivative(
            expquad=self._expquad_time,
            direction=self._direction[0],
            reverse=not reverse,
        )

        self._expquad_laplacian_space = (
            self._expquad_spatial_laplacian._expquad_laplacian_space
        )

        self._expquad_ddv_laplacian_space = ExpQuad_DirectionalDerivative_Laplacian(
            expquad=self._expquad_spatial_laplacian._expquad_space,
            direction=self._direction[1:],
            alpha=self._alpha,
            reverse=self._reverse,
            expquad_laplacian=self._expquad_spatial_laplacian._expquad_laplacian_space,
        )

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def reverse(self) -> bool:
        return self._reverse

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return (
            self._expquad_ddv_time(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_laplacian_space(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        ) + (
            self._expquad_time(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_ddv_laplacian_space(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return (
            self._expquad_ddv_time.jax(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_laplacian_space.jax(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        ) + (
            self._expquad_time.jax(x0[..., 0], None if x1 is None else x1[..., 0])
            * self._expquad_ddv_laplacian_space.jax(
                x0[..., 1:], None if x1 is None else x1[..., 1:]
            )
        )


@diffops.DirectionalDerivative.__call__.register
def _(self, k: ExpQuad_Identity_SpatialLaplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_SpatialLaplacian(
            expquad=k.expquad,
            direction=self.direction,
            alpha=k._alpha,
            reverse=(argnum == 1),
            expquad_spatial_laplacian=k,
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.SpatialLaplacian.__call__.register
def _(self, k: ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ExpQuad_DirectionalDerivative_SpatialLaplacian(
            expquad=k.expquad,
            direction=k.direction,
            alpha=self._alpha,
            reverse=(argnum == 0),
        )

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)
