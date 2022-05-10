import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from .._jax import JaxKernel
from .._matern import Matern
from .._stationary import JaxStationaryMixin


class Matern_Identity_Laplacian(JaxKernel, JaxStationaryMixin):
    def __init__(self, matern: Matern, reverse: bool = True):

        self._matern = matern

        super().__init__(self._matern.input_shape, output_shape=())

        self._reverse = bool(reverse)

    @property
    def matern(self) -> Matern:
        return self._matern

    @property
    def reverse(self) -> bool:
        return self._reverse

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        dists = self._euclidean_distances(x0, x1) / self._matern.lengthscale
        scaled_dists = np.sqrt(2 * self._matern.p + 1) * dists

        if self._matern.p == 3:
            return (
                (2 * self._matern.p + 1)
                / self._matern.lengthscale**2
                * np.exp(-scaled_dists)
                * ((1.0 / 15.0 * scaled_dists**2 - 0.2) * scaled_dists - 0.2)
            )

        raise ValueError()

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        dists = self._euclidean_distances_jax(x0, x1) / self._matern.lengthscale
        scaled_dists = jnp.sqrt(2 * self._matern.p + 1) * dists

        if self._matern.p == 3:
            return (
                (2 * self._matern.p + 1)
                / self._matern.lengthscale**2
                * jnp.exp(-scaled_dists)
                * ((1.0 / 15.0 * scaled_dists**2 - 0.2) * scaled_dists - 0.2)
            )

        raise ValueError()


@diffops.Laplacian.__call__.register  # pylint: disable=no-member
def _(self, k: Matern, /, *, argnum: int = 0):  # pylint: disable=unused-argument
    return Matern_Identity_Laplacian(
        matern=k,
        reverse=(argnum == 0),
    )


class Matern_Laplacian_Laplacian(JaxKernel, JaxStationaryMixin):
    def __init__(self, matern: Matern):
        self._matern = matern

        super().__init__(self._matern.input_shape, output_shape=())

    @property
    def matern(self) -> Matern:
        return self._matern

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        dists = self._euclidean_distances(x0, x1) / self._matern.lengthscale
        scaled_dists = np.sqrt(2 * self._matern.p + 1) * dists

        if self._matern.p == 3:
            return (
                ((2 * self._matern.p + 1) / self._matern.lengthscale**2) ** 2
                * np.exp(-scaled_dists)
                * (
                    ((1.0 / 15.0 * scaled_dists - 0.4) * scaled_dists + 0.2)
                    * scaled_dists
                    + 0.2
                )
            )

        raise ValueError()

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        dists = self._euclidean_distances_jax(x0, x1) / self._matern.lengthscale
        scaled_dists = np.sqrt(2 * self._matern.p + 1) * dists

        if self._matern.p == 3:
            return (
                ((2 * self._matern.p + 1) / self._matern.lengthscale**2) ** 2
                * jnp.exp(-scaled_dists)
                * (
                    ((1.0 / 15.0 * scaled_dists - 0.4) * scaled_dists + 0.2)
                    * scaled_dists
                    + 0.2
                )
            )

        raise ValueError()


@diffops.Laplacian.__call__.register  # pylint: disable=no-member
def _(self, k: Matern_Identity_Laplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return Matern_Laplacian_Laplacian(matern=k.matern)

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)
