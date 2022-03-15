import functools
from typing import Optional

from jax import numpy as jnp
import numpy as np
from probnum.typing import ShapeType

from ....problems.pde import diffops
from .._expquad import ExpQuad
from .._jax import JaxKernel


class ExpQuadDirectionalDerivativeCross(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeType,
        argnum: int,
        lengthscales: np.ndarray,
        output_scale: np.ndarray,
        direction: np.ndarray,
    ):
        super().__init__(input_shape, output_shape=())

        assert argnum in (0, 1)

        self._argnum = argnum

        self._expquad_lengthscales = lengthscales
        self._expquad_output_scale = output_scale

        self._direction = direction

        self._output_scale_sq = (
            -(self._expquad_output_scale ** 2)
            if self._argnum == 0
            else self._expquad_output_scale ** 2
        )

    @functools.cached_property
    def _rescaled_direction(self) -> np.ndarray:
        return self._direction / self._expquad_lengthscales ** 2

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._rescaled_direction * diffs
        dists_sq = (diffs / self._expquad_lengthscales) ** 2

        if self.input_ndim > 0:
            assert self.input_ndim == 1

            proj_diffs = np.sum(proj_diffs, axis=-1)
            dists_sq = np.sum(dists_sq, axis=-1)

        return self._output_scale_sq * proj_diffs * np.exp(-0.5 * dists_sq)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._rescaled_direction * diffs
        dists_sq = (diffs / self._expquad_lengthscales) ** 2

        if self.input_ndim > 0:
            assert self.input_ndim == 1

            proj_diffs = jnp.sum(proj_diffs, axis=-1)
            dists_sq = jnp.sum(dists_sq, axis=-1)

        return self._output_scale_sq * proj_diffs * jnp.exp(-0.5 * dists_sq)


@diffops.DirectionalDerivative.__call__.register
def _(self, f: ExpQuad, *, argnum: int = 0, **kwargs):
    return ExpQuadDirectionalDerivativeCross(
        input_shape=self.output_domain_shape,
        argnum=argnum,
        lengthscales=f._lengthscales,
        output_scale=f._output_scale,
        direction=self._direction,
    )


class ExpQuadDirectionalDerivativeBoth(JaxKernel):
    def __init__(
        self,
        input_shape: ShapeType,
        lengthscales: np.ndarray,
        output_scale: np.ndarray,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        super().__init__(input_shape, output_shape=())

        self._expquad_lengthscales = lengthscales
        self._expquad_output_scale = output_scale

        self._direction0 = direction0
        self._direction1 = direction1

    @functools.cached_property
    def _directions_inprod(self) -> np.ndarray:
        inprod = (self._direction0 * self._direction1) / self._expquad_lengthscales ** 2

        if self.input_ndim > 0:
            assert self.input_ndim == 1

            inprod = np.sum(inprod, axis=-1)

        return inprod

    @functools.cached_property
    def _rescaled_direction0(self) -> np.ndarray:
        return self._direction0 / self._expquad_lengthscales ** 2

    @functools.cached_property
    def _rescaled_direction1(self) -> np.ndarray:
        return self._direction1 / self._expquad_lengthscales ** 2

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._expquad_output_scale ** 2 * self._directions_inprod,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        proj_diffs0 = self._rescaled_direction0 * diffs
        proj_diffs1 = self._rescaled_direction1 * diffs

        dists_sq = (diffs / self._expquad_lengthscales) ** 2

        if self.input_ndim > 0:
            assert self.input_ndim == 1

            proj_diffs0 = np.sum(proj_diffs0, axis=-1)
            proj_diffs1 = np.sum(proj_diffs1, axis=-1)

            dists_sq = np.sum(dists_sq, axis=-1)

        return (
            self._expquad_output_scale ** 2
            * (self._directions_inprod - proj_diffs0 * proj_diffs1)
            * np.exp(-0.5 * dists_sq)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._expquad_output_scale ** 2 * self._directions_inprod,
                shape=x0.shape[: x0.ndim - self._input_ndim],
            )

        diffs = x0 - x1

        proj_diffs0 = self._rescaled_direction0 * diffs
        proj_diffs1 = self._rescaled_direction1 * diffs
        dists_sq = (diffs / self._expquad_lengthscales) ** 2

        if self.input_ndim > 0:
            assert self.input_ndim == 1

            proj_diffs0 = jnp.sum(proj_diffs0, axis=-1)
            proj_diffs1 = jnp.sum(proj_diffs1, axis=-1)
            dists_sq = jnp.sum(dists_sq, axis=-1)

        return (
            self._expquad_output_scale ** 2
            * (self._directions_inprod - proj_diffs0 * proj_diffs1)
            * jnp.exp(-0.5 * dists_sq)
        )


@diffops.DirectionalDerivative.__call__.register
def _(self, f: ExpQuadDirectionalDerivativeCross, *, argnum: int = 0, **kwargs):
    if f._argnum == 1 and argnum == 0:
        direction0 = self._direction
        direction1 = f._direction
    elif f._argnum == 0 and argnum == 1:
        direction0 = f._direction
        direction1 = self._direction
    else:
        return super(diffops.DirectionalDerivative, self).__call__(
            f, argnum=argnum, **kwargs
        )

    return ExpQuadDirectionalDerivativeBoth(
        input_shape=self.output_domain_shape,
        lengthscales=f._expquad_lengthscales,
        output_scale=f._expquad_output_scale,
        direction0=direction0,
        direction1=direction1,
    )
