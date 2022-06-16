import functools

from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import diffops

from .._jax import JaxKernel
from .._product_matern import ProductMatern
from .._stationary import JaxStationaryMixin


class ProductMatern_Identity_DirectionalDerivative(JaxKernel, JaxStationaryMixin):
    def __init__(
        self,
        prod_matern: ProductMatern,
        direction: np.ndarray,
        reverse: bool = False,
    ):
        self._prod_matern = prod_matern

        super().__init__(self._prod_matern.input_shape, output_shape=())

        self._direction = direction

        self._reverse = reverse

    @property
    def prod_matern(self) -> ProductMatern:
        return self._prod_matern

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _rescaled_direction(self) -> np.ndarray:
        rescaled_dir = self._prod_matern._scale_factors**2 * self._direction

        return -rescaled_dir if self._reverse else rescaled_dir

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        ks_x0_x1 = np.where(
            np.eye(self.input_shape[0], dtype=np.bool_),
            self._evaluate_factors(x0, x1)[..., None, :],
            self._prod_matern._evaluate_factors(x0, x1)[..., None, :],
        )

        return np.sum(np.prod(ks_x0_x1, axis=-1), axis=-1)

    def _evaluate_factors(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(x0)

        diffs = x0 - x1

        scaled_dists = self._prod_matern._scale_factors * np.abs(diffs)
        proj_scaled_diffs = self._rescaled_direction * diffs

        return (
            np.exp(-scaled_dists)
            * (1 / 15)
            * (3 + scaled_dists * (3 + scaled_dists))
            * proj_scaled_diffs
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        ks_x0_x1 = jnp.where(
            jnp.eye(self.input_shape[0], dtype=jnp.bool_),
            self._evaluate_factors_jax(x0, x1)[..., None, :],
            self._prod_matern._evaluate_factors_jax(x0, x1)[..., None, :],
        )

        return jnp.sum(jnp.prod(ks_x0_x1, axis=-1), axis=-1)

    def _evaluate_factors_jax(
        self, x0: jnp.ndarray, x1: jnp.ndarray | None
    ) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(x0)

        diffs = x0 - x1

        scaled_dists = self._prod_matern._scale_factors * jnp.abs(diffs)
        proj_scaled_diffs = self._rescaled_direction * diffs

        return (
            jnp.exp(-scaled_dists)
            * (1 / 15)
            * (3 + scaled_dists * (3 + scaled_dists))
            * proj_scaled_diffs
        )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: ProductMatern, /, *, argnum: int = 0):
    return ProductMatern_Identity_DirectionalDerivative(
        prod_matern=k,
        direction=self.direction,
        reverse=(argnum == 0),
    )


class ProductMatern_DirectionalDerivative_DirectionalDerivative(JaxKernel):
    def __init__(
        self,
        prod_matern: ProductMatern,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        self._prod_matern = prod_matern

        super().__init__(self._prod_matern.input_shape, output_shape=())

        self._direction0 = direction0
        self._direction1 = direction1

        self._prod_matern_dderiv_id = ProductMatern_Identity_DirectionalDerivative(
            self._prod_matern,
            direction=self._direction0,
            reverse=True,
        )

        self._prod_matern_id_dderiv = ProductMatern_Identity_DirectionalDerivative(
            self._prod_matern,
            direction=self._direction1,
            reverse=False,
        )

    @property
    def prod_matern(self) -> ProductMatern:
        return self._prod_matern

    @functools.cached_property
    def _rescaled_direction0(self) -> np.ndarray:
        return self._prod_matern._scale_factors**2 * self._direction0

    @functools.cached_property
    def _rescaled_direction1(self) -> np.ndarray:
        return self._prod_matern._scale_factors**2 * self._direction1

    @functools.cached_property
    def _directions_prod(self) -> np.ndarray:
        return self._direction0 * self._rescaled_direction1

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        idx_equal_mask = np.eye(self.input_shape[0], dtype=np.bool_)

        ks_dderiv_dderiv_x0_x1 = self._prod_matern._evaluate_factors(x0, x1)[
            ..., None, None, :
        ]

        ks_dderiv_dderiv_x0_x1 = np.where(
            idx_equal_mask[None, :, :],
            self._prod_matern_id_dderiv._evaluate_factors(x0, x1)[..., None, None, :],
            ks_dderiv_dderiv_x0_x1,
        )

        ks_dderiv_dderiv_x0_x1 = np.where(
            idx_equal_mask[:, None, :],
            self._prod_matern_dderiv_id._evaluate_factors(x0, x1)[..., None, None, :],
            ks_dderiv_dderiv_x0_x1,
        )

        ks_dderiv_dderiv_x0_x1 = np.where(
            idx_equal_mask[:, None, :] & idx_equal_mask[None, :, :],
            self._evaluate_factors(x0, x1)[..., None, None, :],
            ks_dderiv_dderiv_x0_x1,
        )

        return np.sum(
            np.prod(ks_dderiv_dderiv_x0_x1, axis=-1),
            axis=(-2, -1),
        )

    def _evaluate_factors(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.broadcast_to(0.2 * self._directions_prod, shape=x0.shape)

        diffs = x0 - x1

        proj_scaled_diffs0 = self._rescaled_direction0 * diffs
        proj_scaled_diffs1 = self._rescaled_direction1 * diffs
        scaled_dists = self._prod_matern._scale_factors * np.abs(diffs)

        return (
            np.exp(-scaled_dists)
            * (1 / 15)
            * (
                (3 + scaled_dists * (3 + scaled_dists)) * self._directions_prod
                - (1 + scaled_dists) * proj_scaled_diffs0 * proj_scaled_diffs1
            )
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        idx_equal_mask = np.eye(self.input_shape[0], dtype=np.bool_)

        ks_dderiv_dderiv_x0_x1 = self._prod_matern._evaluate_factors_jax(x0, x1)[
            ..., None, None, :
        ]

        ks_dderiv_dderiv_x0_x1 = jnp.where(
            idx_equal_mask[None, :, :],
            self._prod_matern_id_dderiv._evaluate_factors_jax(x0, x1)[
                ..., None, None, :
            ],
            ks_dderiv_dderiv_x0_x1,
        )

        ks_dderiv_dderiv_x0_x1 = jnp.where(
            idx_equal_mask[:, None, :],
            self._prod_matern_dderiv_id._evaluate_factors_jax(x0, x1)[
                ..., None, None, :
            ],
            ks_dderiv_dderiv_x0_x1,
        )

        ks_dderiv_dderiv_x0_x1 = jnp.where(
            idx_equal_mask[:, None, :] & idx_equal_mask[None, :, :],
            self._evaluate_factors_jax(x0, x1)[..., None, None, :],
            ks_dderiv_dderiv_x0_x1,
        )

        return jnp.sum(
            jnp.prod(ks_dderiv_dderiv_x0_x1, axis=-1),
            axis=(-2, -1),
        )

    def _evaluate_factors_jax(
        self, x0: jnp.ndarray, x1: jnp.ndarray | None
    ) -> jnp.ndarray:
        if x1 is None:
            return jnp.broadcast_to(0.2 * self._directions_prod, shape=x0.shape)

        diffs = x0 - x1

        proj_scaled_diffs0 = self._rescaled_direction0 * diffs
        proj_scaled_diffs1 = self._rescaled_direction1 * diffs
        scaled_dists = self._prod_matern._scale_factors * np.abs(diffs)

        return (
            jnp.exp(-scaled_dists)
            * (1 / 15)
            * (
                (3 + scaled_dists * (3 + scaled_dists)) * self._directions_prod
                - (1 + scaled_dists) * proj_scaled_diffs0 * proj_scaled_diffs1
            )
        )


@diffops.DirectionalDerivative.__call__.register  # pylint: disable=no-member
def _(self, k: ProductMatern_Identity_DirectionalDerivative, /, *, argnum: int = 0):
    if argnum == 0 and not k.reverse:
        direction0 = self.direction
        direction1 = k.direction
    elif argnum == 1 and k.reverse:
        direction0 = k.direction
        direction1 = self.direction
    else:
        return super(diffops.DirectionalDerivative, self).__call__(k, argnum=argnum)

    return ProductMatern_DirectionalDerivative_DirectionalDerivative(
        prod_matern=k.prod_matern,
        direction0=direction0,
        direction1=direction1,
    )
