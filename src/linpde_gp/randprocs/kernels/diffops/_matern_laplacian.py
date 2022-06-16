import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from .._jax import JaxKernel
from .._matern import Matern, ProductMatern
from .._stationary import JaxStationaryMixin
from ._matern_directional_derivative import Matern_Identity_DirectionalDerivative


class Matern_Identity_Laplacian(JaxKernel, JaxStationaryMixin):
    def __init__(self, matern: Matern, reverse: bool = True):

        self._matern = matern

        if self._matern.input_ndim != 0:
            raise ValueError()

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


class ProductMatern_Identity_Laplacian(JaxKernel, JaxStationaryMixin):
    def __init__(self, prod_matern: ProductMatern, reverse: bool = True):
        self._prod_matern = prod_matern

        super().__init__(self._prod_matern.input_shape, output_shape=())

        self._reverse = bool(reverse)

    @property
    def prod_matern(self) -> Matern:
        return self._prod_matern

    @property
    def reverse(self) -> bool:
        return self._reverse

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        return np.sum(
            np.prod(
                np.where(
                    np.eye(self.input_shape[0], dtype=np.bool_),
                    self._evaluate_factors(x0, x1)[..., None, :],
                    self._prod_matern._evaluate_factors(x0, x1)[..., None, :],
                ),
                axis=-1,
            ),
            axis=-1,
        )

    def _evaluate_factors(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            scaled_dists = np.zeros_like(x0)
        else:
            scaled_dists = self._prod_matern._scale_factors * np.abs(x0 - x1)

        if self._prod_matern.p == 3:
            return (
                self._prod_matern._scale_factors**2
                * np.exp(-scaled_dists)
                * ((1.0 / 15.0 * scaled_dists**2 - 0.2) * scaled_dists - 0.2)
            )

        raise ValueError()

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return jnp.sum(
            jnp.prod(
                jnp.where(
                    jnp.eye(self.input_shape[0], dtype=jnp.bool_),
                    self._evaluate_factors(x0, x1)[..., None, :],
                    self._prod_matern._evaluate_factors(x0, x1)[..., None, :],
                ),
                axis=-1,
            ),
            axis=-1,
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_factors_jax(
        self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        if x1 is None:
            scaled_dists = jnp.zeros_like(x0, shape=x0.shape)
        else:
            scaled_dists = self._prod_matern._scale_factors * jnp.abs(x0 - x1)

        if self._prod_matern.p == 3:
            return (
                self._prod_matern._scale_factors**2
                * jnp.exp(-scaled_dists)
                * ((1.0 / 15.0 * scaled_dists**2 - 0.2) * scaled_dists - 0.2)
            )

        raise ValueError()


@diffops.Laplacian.__call__.register  # pylint: disable=no-member
def _(self, k: ProductMatern, /, *, argnum: int = 0):  # pylint: disable=unused-argument
    return ProductMatern_Identity_Laplacian(
        prod_matern=k,
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


class ProductMatern_Laplacian_Laplacian(JaxKernel, JaxStationaryMixin):
    def __init__(self, prod_matern: ProductMatern):
        self._prod_matern = prod_matern
        self._prod_matern_id_lap = ProductMatern_Identity_Laplacian(
            self._prod_matern,
            reverse=False,
        )

        super().__init__(self._prod_matern.input_shape, output_shape=())

    @property
    def prod_matern(self) -> ProductMatern:
        return self._prod_matern

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        idx_equal_mask = np.eye(self.input_shape[0], dtype=np.bool)

        ks_x0_x1 = np.where(
            idx_equal_mask[None, :, :] ^ idx_equal_mask[:, None, :],
            self._prod_matern_id_lap._evaluate_factors(x0, x1)[..., None, None, :],
            self._prod_matern._evaluate_factors(x0, x1)[..., None, None, :],
        )

        ks_x0_x1 = np.where(
            idx_equal_mask[None, :, :] & idx_equal_mask[:, None, :],
            self._evaluate_factors(x0, x1)[..., None, None, :],
            ks_x0_x1,
        )

        return np.sum(
            np.prod(ks_x0_x1, axis=-1),
            axis=(-2, -1),
        )

    def _evaluate_factors(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            scaled_dists = np.zeros_like(x0)
        else:
            scaled_dists = self._prod_matern._scale_factors * np.abs(x0 - x1)

        if self._prod_matern.p == 3:
            return (
                self._prod_matern._scale_factors**4
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
        idx_equal_mask = np.eye(self.input_shape[0], dtype=np.bool)

        ks_x0_x1 = np.where(
            idx_equal_mask[None, :, :] ^ idx_equal_mask[:, None, :],
            self._prod_matern_id_lap._evaluate_factors(x0, x1)[..., None, None, :],
            self._prod_matern._evaluate_factors(x0, x1)[..., None, None, :],
        )

        ks_x0_x1 = np.where(
            idx_equal_mask[None, :, :] & idx_equal_mask[:, None, :],
            self._evaluate_factors(x0, x1)[..., None, None, :],
            ks_x0_x1,
        )

        return np.sum(
            np.prod(ks_x0_x1, axis=-1),
            axis=(-2, -1),
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_factors_jax(
        self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        if x1 is None:
            scaled_dists = jnp.zeros_like(x0)
        else:
            scaled_dists = self._prod_matern._scale_factors * jnp.abs(x0 - x1)

        if self._prod_matern.p == 3:
            return (
                self._prod_matern._scale_factors**4
                * jnp.exp(-scaled_dists)
                * (
                    ((1.0 / 15.0 * scaled_dists - 0.4) * scaled_dists + 0.2)
                    * scaled_dists
                    + 0.2
                )
            )

        raise ValueError()


@diffops.Laplacian.__call__.register  # pylint: disable=no-member
def _(self, k: ProductMatern_Identity_Laplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return ProductMatern_Laplacian_Laplacian(prod_matern=k.prod_matern)

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)


class Matern_DirectionalDerivative_Laplacian(JaxKernel, JaxStationaryMixin):
    def __init__(
        self,
        matern: Matern,
        direction: np.ndarray,
        reverse: bool = False,
    ):
        self._matern = matern

        super().__init__(self._matern.input_shape, output_shape=())

        self._direction = direction

        self._reverse = bool(reverse)

    @property
    def matern(self) -> Matern:
        return self._matern

    @functools.cached_property
    def _rescaled_direction(self) -> np.ndarray:
        rescaled_dir = self._matern._scale_factor**4 * self._direction

        return -rescaled_dir if self._reverse else rescaled_dir

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_scaled_diffs = self._batched_sum(self._rescaled_direction * diffs)
        scaled_dists = self._matern._scale_factor * self._batched_euclidean_norm(diffs)

        return (
            (1 / 15)
            * np.exp(-scaled_dists)
            * (3 + scaled_dists * (3 - scaled_dists))
            * proj_scaled_diffs
        )

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        diffs = x0 - x1

        proj_scaled_diffs = self._batched_sum_jax(self._rescaled_direction * diffs)
        scaled_dists = self._matern._scale_factor * self._batched_euclidean_norm_jax(
            diffs
        )

        return (
            (1 / 15)
            * jnp.exp(-scaled_dists)
            * (3 + scaled_dists * (3 - scaled_dists))
            * proj_scaled_diffs
        )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: Matern_Identity_Laplacian, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return Matern_DirectionalDerivative_Laplacian(
            matern=k.matern,
            direction=self.direction,
            reverse=(argnum == 1),
        )

    return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)


@diffops.Laplacian.__call__.register  # pylint: disable=no-member
def _(self, k: Matern_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if (argnum == 0 and not k.reverse) or (argnum == 1 and k.reverse):
        return Matern_DirectionalDerivative_Laplacian(
            matern=k.matern,
            direction=k.direction,
            reverse=(argnum == 0),
        )

    return super(diffops.Laplacian, self).__call__(k, argnum=argnum)
