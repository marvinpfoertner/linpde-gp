import functools
from typing import Optional

from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from .._expquad import ExpQuad
from .._jax import JaxKernel


class ExpQuad_Identity_DirectionalDerivative(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction: np.ndarray,
        reverse: bool = False,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction = direction

        self._reverse = reverse

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _rescaled_direction(self) -> np.ndarray:
        rescaled_dir = self._direction / self._expquad.lengthscales ** 2

        return -rescaled_dir if self._reverse else rescaled_dir

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._batched_sum(self._rescaled_direction * diffs)
        dists_sq = self._batched_euclidean_norm_sq(diffs / self._expquad.lengthscales)

        return proj_diffs * np.exp(-0.5 * dists_sq)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs = self._batched_sum_jax(self._rescaled_direction * diffs)
        dists_sq = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad.lengthscales
        )

        return proj_diffs * jnp.exp(-0.5 * dists_sq)


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad, /, *, argnum: int = 0):
    return ExpQuad_Identity_DirectionalDerivative(
        expquad=k,
        direction=self.direction,
        reverse=(argnum == 0),
    )


class ExpQuad_DirectionalDerivative_DirectionalDerivative(JaxKernel):
    def __init__(
        self,
        expquad: ExpQuad,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        self._expquad = expquad

        super().__init__(self._expquad.input_shape, output_shape=())

        self._direction0 = direction0
        self._direction1 = direction1

    @property
    def expquad(self) -> ExpQuad:
        return self._expquad

    @functools.cached_property
    def _rescaled_direction0(self) -> np.ndarray:
        return self._direction0 / self._expquad.lengthscales ** 2

    @functools.cached_property
    def _rescaled_direction1(self) -> np.ndarray:
        return self._direction1 / self._expquad.lengthscales ** 2

    @functools.cached_property
    def _directions_inprod(self) -> np.ndarray:
        return self._batched_sum(self._direction0 * self._rescaled_direction1)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs0 = self._batched_sum(self._rescaled_direction0 * diffs)
        proj_diffs1 = self._batched_sum(self._rescaled_direction1 * diffs)
        dists_sq = self._batched_euclidean_norm_sq(diffs / self._expquad.lengthscales)

        return (self._directions_inprod - proj_diffs0 * proj_diffs1) * np.exp(
            -0.5 * dists_sq
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_diffs0 = self._batched_sum_jax(self._rescaled_direction0 * diffs)
        proj_diffs1 = self._batched_sum_jax(self._rescaled_direction1 * diffs)
        dists_sq = self._batched_euclidean_norm_sq_jax(
            diffs / self._expquad.lengthscales
        )

        return (self._directions_inprod - proj_diffs0 * proj_diffs1) * jnp.exp(
            -0.5 * dists_sq
        )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: ExpQuad_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        direction0 = self.direction
        direction1 = k.direction
    elif argnum == 1 and k.reverse:
        direction0 = k.direction
        direction1 = self.direction
    else:
        return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)

    return ExpQuad_DirectionalDerivative_DirectionalDerivative(
        expquad=k.expquad,
        direction0=direction0,
        direction1=direction1,
    )
