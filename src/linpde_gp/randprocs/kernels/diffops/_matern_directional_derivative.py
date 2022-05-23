import functools
from typing import Optional

from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from .._jax import JaxKernel
from .._matern import Matern
from .._stationary import JaxStationaryMixin


class Matern_Identity_DirectionalDerivative(JaxKernel, JaxStationaryMixin):
    def __init__(
        self,
        matern: Matern,
        direction: np.ndarray,
        reverse: bool = False,
    ):
        self._matern = matern

        super().__init__(self._matern.input_shape, output_shape=())

        self._direction = direction

        self._reverse = reverse

        self._scale_factor = np.sqrt(2 * self._matern.p + 1) / self._matern.lengthscale

    @property
    def matern(self) -> Matern:
        return self._matern

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _rescaled_direction(self) -> np.ndarray:
        rescaled_dir = self._scale_factor**2 * self._direction

        return -rescaled_dir if self._reverse else rescaled_dir

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_scaled_diffs = self._batched_sum(self._rescaled_direction * diffs)
        scaled_dists = self._scale_factor * self._batched_euclidean_norm(diffs)

        return (
            np.exp(-scaled_dists)
            * (1 / 15)
            * (3 + scaled_dists * (3 + scaled_dists))
            * proj_scaled_diffs
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_scaled_diffs = self._batched_sum_jax(self._rescaled_direction * diffs)
        scaled_dists = self._scale_factor * self._batched_euclidean_norm_jax(diffs)

        return (
            jnp.exp(-scaled_dists)
            * (1 / 15)
            * (3 + scaled_dists * (3 + scaled_dists))
            * proj_scaled_diffs
        )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: Matern, /, *, argnum: int = 0):
    return Matern_Identity_DirectionalDerivative(
        matern=k,
        direction=self.direction,
        reverse=(argnum == 0),
    )


class Matern_DirectionalDerivative_DirectionalDerivative(JaxKernel):
    def __init__(
        self,
        matern: Matern,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        self._matern = matern

        super().__init__(self._matern.input_shape, output_shape=())

        self._direction0 = direction0
        self._direction1 = direction1

        self._scale_factor = np.sqrt(2 * self._matern.p + 1) / self._matern.lengthscale

    @property
    def matern(self) -> Matern:
        return self._matern

    @functools.cached_property
    def _rescaled_direction0(self) -> np.ndarray:
        return self._scale_factor**2 * self._direction0

    @functools.cached_property
    def _rescaled_direction1(self) -> np.ndarray:
        return self._scale_factor**2 * self._direction1

    @functools.cached_property
    def _directions_inprod(self) -> np.ndarray:
        return self._batched_sum(self._direction0 * self._rescaled_direction1)

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                0.2 * self._directions_inprod,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_scaled_diffs0 = self._batched_sum(self._rescaled_direction0 * diffs)
        proj_scaled_diffs1 = self._batched_sum(self._rescaled_direction1 * diffs)
        scaled_dists = self._scale_factor * self._batched_euclidean_norm(diffs)

        return (
            np.exp(-scaled_dists)
            * (1 / 15)
            * (
                (3 + scaled_dists * (3 + scaled_dists)) * self._directions_inprod
                - (1 + scaled_dists) * proj_scaled_diffs0 * proj_scaled_diffs1
            )
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                0.2 * self._directions_inprod,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_scaled_diffs0 = self._batched_sum_jax(self._rescaled_direction0 * diffs)
        proj_scaled_diffs1 = self._batched_sum_jax(self._rescaled_direction1 * diffs)
        scaled_dists = self._scale_factor * self._batched_euclidean_norm_jax(diffs)

        return (
            jnp.exp(-scaled_dists)
            * (1 / 15)
            * (
                (3 + scaled_dists * (3 + scaled_dists)) * self._directions_inprod
                - (1 + scaled_dists) * proj_scaled_diffs0 * proj_scaled_diffs1
            )
        )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: Matern_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        direction0 = self.direction
        direction1 = k.direction
    elif argnum == 1 and k.reverse:
        direction0 = k.direction
        direction1 = self.direction
    else:
        return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)

    return Matern_DirectionalDerivative_DirectionalDerivative(
        matern=k.matern,
        direction0=direction0,
        direction1=direction1,
    )
