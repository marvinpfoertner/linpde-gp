from __future__ import annotations

import functools
from typing import Optional

try:
    from pykeops.numpy import LazyTensor, Pm, Vi, Vj
except ImportError:
    pass

import numpy as np
from jax import numpy as jnp
from linpde_gp.functions import Monomial, RationalPolynomial
from linpde_gp.linfuncops import diffops
from probnum.randprocs import covfuncs

from ..._jax import JaxCovarianceFunction, JaxIsotropicMixin


class HalfIntegerMatern_Identity_DirectionalDerivative(JaxCovarianceFunction):
    def __init__(
        self,
        matern: covfuncs.Matern,
        *,
        direction: np.ndarray,
        reverse: bool = False,
    ):
        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._direction = direction
        self._reverse = reverse

        self._poly = half_integer_matern_derivative_polynomial(
            self.matern.p, 1
        ) // Monomial(1)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _scaled_direction(self) -> np.ndarray:
        scaled_direction = self._matern._scale_factors
        scaled_direction *= self.direction

        if not self._reverse:
            scaled_direction *= -1

        return scaled_direction

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern._scale_factors

        proj_scaled_diffs = self._batched_sum(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm(scaled_diffs)

        # Polynomial part
        res = self._poly(scaled_dists)

        # Exponential part
        res *= np.exp(-scaled_dists)

        # Chain Rule
        res *= proj_scaled_diffs

        return res

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern._scale_factors

        proj_scaled_diffs = self._batched_sum_jax(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm_jax(scaled_diffs)

        # Polynomial part
        res = self._poly.jax(scaled_dists)

        # Exponential part
        res *= jnp.exp(-scaled_dists)

        # Chain Rule
        res *= proj_scaled_diffs

        return res

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        if x1 is None:
            x1 = x0
        if len(x0.shape) < 2:
            x0 = x0.reshape(-1, 1)
        if len(x1.shape) < 2:
            x1 = x1.reshape(-1, 1)

        scaled_diffs = Vi(x0) - Vj(x1)
        scaled_diffs *= Pm(self._matern._scale_factors)

        proj_scaled_diffs = (Pm(self._scaled_direction) * scaled_diffs).sum()
        scaled_dists = scaled_diffs * scaled_diffs
        scaled_dists = scaled_dists.sum().sqrt()

        res = self._poly._evaluate_keops(scaled_dists)
        res *= (-scaled_dists).exp()
        res *= proj_scaled_diffs

        return res


class HalfIntegerMatern_DirectionalDerivative_DirectionalDerivative(
    JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern

        self._direction0 = direction0
        self._direction1 = direction1

        self._neg_poly_deriv = -half_integer_matern_derivative_polynomial(
            self._matern.p, 1
        ) // Monomial(1)

        self._poly_diff = (
            half_integer_matern_derivative_polynomial(self._matern.p, 2)
            + self._neg_poly_deriv
        ) // Monomial(2)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @functools.cached_property
    def _scaled_direction0(self) -> np.ndarray:
        return self._matern._scale_factors * self._direction0

    @functools.cached_property
    def _scaled_direction1(self) -> np.ndarray:
        return self._matern._scale_factors * self._direction1

    @functools.cached_property
    def _directions_inprod(self) -> np.floating:
        return np.sum(self._scaled_direction0 * self._scaled_direction1)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod * self._neg_poly_deriv.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern._scale_factors

        proj_scaled_diffs0 = self._batched_sum(self._scaled_direction0 * scaled_diffs)
        proj_scaled_diffs1 = self._batched_sum(self._scaled_direction1 * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm(scaled_diffs)

        res = self._directions_inprod * self._neg_poly_deriv(scaled_dists)
        res -= proj_scaled_diffs0 * proj_scaled_diffs1 * self._poly_diff(scaled_dists)

        return res * np.exp(-scaled_dists)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._directions_inprod * self._neg_poly_deriv.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern._scale_factors

        proj_scaled_diffs0 = self._batched_sum_jax(
            self._scaled_direction0 * scaled_diffs
        )
        proj_scaled_diffs1 = self._batched_sum_jax(
            self._scaled_direction1 * scaled_diffs
        )
        scaled_dists = self._batched_euclidean_norm_jax(scaled_diffs)

        res = self._directions_inprod * self._neg_poly_deriv.jax(scaled_dists)
        res -= (
            proj_scaled_diffs0 * proj_scaled_diffs1 * self._poly_diff.jax(scaled_dists)
        )

        return res * jnp.exp(-scaled_dists)

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        if x1 is None:
            x1 = x0
        if len(x0.shape) < 2:
            x0 = x0.reshape(-1, 1)
        if len(x1.shape) < 2:
            x1 = x1.reshape(-1, 1)

        scaled_diffs = Vi(x0) - Vj(x1)
        scaled_diffs *= Pm(self._matern._scale_factors)

        proj_scaled_diffs0 = (Pm(self._scaled_direction0) * scaled_diffs).sum()
        proj_scaled_diffs1 = (Pm(self._scaled_direction1) * scaled_diffs).sum()

        scaled_dists = scaled_diffs * scaled_diffs
        scaled_dists = scaled_dists.sum().sqrt()

        res = Pm(self._directions_inprod) * self._neg_poly_deriv._evaluate_keops(
            scaled_dists
        )
        res -= (
            proj_scaled_diffs0
            * proj_scaled_diffs1
            * self._poly_diff._evaluate_keops(scaled_dists)
        )
        res *= (-scaled_dists).exp()

        return res


class UnivariateHalfIntegerMatern_DirectionalDerivative_DirectionalDerivative(
    covfuncs.IsotropicMixin, JaxIsotropicMixin, JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        direction0: np.ndarray,
        direction1: np.ndarray,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern

        self._direction0 = direction0
        self._direction1 = direction1

        self._poly = -half_integer_matern_derivative_polynomial(self._matern.p, 2)

    @functools.cached_property
    def _scaled_directions_prod(self) -> np.floating:
        return np.squeeze(
            self._direction0 * self._direction1 * self._matern._scale_factors**2
        )[()]

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._scaled_directions_prod * self._poly.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_dists = self._euclidean_distances(
            x0,
            x1,
            scale_factors=self._matern._scale_factors,
        )

        return (
            self._scaled_directions_prod
            * self._poly(scaled_dists)
            * np.exp(-scaled_dists)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.full_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                self._scaled_directions_prod * self._poly.coefficients[0],
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_dists = self._euclidean_distances_jax(
            x0,
            x1,
            scale_factors=self._matern._scale_factors,
        )

        return (
            self._scaled_directions_prod
            * self._poly.jax(scaled_dists)
            * jnp.exp(-scaled_dists)
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        scaled_dists = self._euclidean_distances_keops(
            x0,
            x1,
            scale_factors=self._matern._scale_factors,
        )

        return (
            Pm(self._scaled_directions_prod)
            * self._poly._evaluate_keops(scaled_dists)
            * (-scaled_dists).exp()
        )


class UnivariateHalfIntegerMatern_Identity_WeightedLaplacian(
    covfuncs.IsotropicMixin, JaxIsotropicMixin, JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        L: diffops.WeightedLaplacian,
        reverse: bool = True,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._L = L
        self._reverse = bool(reverse)

        self._poly = half_integer_matern_derivative_polynomial(matern.p, 2)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @property
    def reverse(self) -> bool:
        return self._reverse

    @functools.cached_property
    def _output_scale_factor(self) -> np.floating:
        return np.squeeze(
            self._L.weights * self._matern._scale_factors * self._matern._scale_factors
        )[()]

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        scaled_dists = self._euclidean_distances(
            x0, x1, scale_factors=self._matern._scale_factors
        )

        return (
            self._output_scale_factor * np.exp(-scaled_dists) * self._poly(scaled_dists)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        scaled_dists = self._euclidean_distances_jax(
            x0, x1, scale_factors=self._matern._scale_factors
        )

        return (
            self._output_scale_factor
            * jnp.exp(-scaled_dists)
            * self._poly.jax(scaled_dists)
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        scaled_dists = self._euclidean_distances_keops(
            x0, x1, scale_factors=self._matern._scale_factors
        )

        return (
            Pm(self._output_scale_factor)
            * (-scaled_dists).exp()
            * self._poly._evaluate_keops(scaled_dists)
        )


class UnivariateHalfIntegerMatern_WeightedLaplacian_WeightedLaplacian(
    covfuncs.IsotropicMixin, JaxIsotropicMixin, JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        L0: diffops.WeightedLaplacian,
        L1: diffops.WeightedLaplacian,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern
        self._L0 = L0
        self._L1 = L1

        self._poly = half_integer_matern_derivative_polynomial(matern.p, 4)

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @functools.cached_property
    def _output_scale_factor(self) -> float:
        return np.squeeze(
            self._L0.weights * self._L1.weights * self._matern._scale_factors**4
        )[()]

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        scaled_dists = self._euclidean_distances(
            x0, x1, scale_factors=self._matern._scale_factors
        )

        return (
            self._output_scale_factor * np.exp(-scaled_dists) * self._poly(scaled_dists)
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        scaled_dists = self._euclidean_distances_jax(
            x0, x1, scale_factors=self._matern._scale_factors
        )

        return (
            self._output_scale_factor
            * jnp.exp(-scaled_dists)
            * self._poly.jax(scaled_dists)
        )

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        scaled_dists = self._euclidean_distances_keops(
            x0, x1, scale_factors=self._matern._scale_factors
        )

        return (
            Pm(self._output_scale_factor)
            * (-scaled_dists).exp()
            * self._poly._evaluate_keops(scaled_dists)
        )


class UnivariateHalfIntegerMatern_DirectionalDerivative_WeightedLaplacian(
    JaxCovarianceFunction
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        direction: np.ndarray,
        L1: diffops.WeightedLaplacian,
        reverse: bool = False,
    ):
        if matern.input_size != 1:
            raise ValueError("`matern` must be univariate.")

        if matern.p is None:
            raise ValueError(
                "`matern` must be a half-integer Matérn covariance function."
            )

        super().__init__(input_shape=matern.input_shape)

        self._matern = matern

        self._direction = direction
        self._L1 = L1

        self._reverse = bool(reverse)

        self._poly = half_integer_matern_derivative_polynomial(matern.p, 3) // Monomial(
            1
        )

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @functools.cached_property
    def _scaled_direction(self) -> np.ndarray:
        scaled_direction = self._L1.weights * self._matern._scale_factors**3
        scaled_direction *= self._direction

        if self._reverse:
            scaled_direction *= -1

        return scaled_direction

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        if x1 is None:
            return np.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern._scale_factors

        proj_scaled_diffs = self._batched_sum(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm(scaled_diffs)

        return np.exp(-scaled_dists) * self._poly(scaled_dists) * proj_scaled_diffs

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        if x1 is None:
            return jnp.zeros_like(  # pylint: disable=unexpected-keyword-arg
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        scaled_diffs = x0 - x1
        scaled_diffs *= self._matern._scale_factors

        proj_scaled_diffs = self._batched_sum_jax(self._scaled_direction * scaled_diffs)
        scaled_dists = self._batched_euclidean_norm_jax(scaled_diffs)

        return jnp.exp(-scaled_dists) * self._poly.jax(scaled_dists) * proj_scaled_diffs

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        if x1 is None:
            x1 = x0
        if len(x0.shape) < 2:
            x0 = x0.reshape(-1, 1)
        if len(x1.shape) < 2:
            x1 = x1.reshape(-1, 1)
        scaled_diffs = Vi(x0) - Vj(x1)
        scaled_diffs *= Pm(self._matern._scale_factors)

        proj_scaled_diffs = (Pm(self._scaled_direction) * scaled_diffs).sum()
        scaled_dists = scaled_diffs * scaled_diffs
        scaled_dists = scaled_dists.sum().sqrt()

        return (
            (-scaled_dists).exp()
            * self._poly._evaluate_keops(scaled_dists)
            * proj_scaled_diffs
        )


@functools.lru_cache(maxsize=None)
def half_integer_matern_polynomial(p: int) -> RationalPolynomial:
    return RationalPolynomial(covfuncs.Matern.half_integer_coefficients(p))


@functools.lru_cache(maxsize=None)
def half_integer_matern_derivative_polynomial(p: int, n: int) -> RationalPolynomial:
    r"""Polynomial coefficients for `n`-th derivatives of the Matérn covariance function
    with :math:`\nu = p + \frac{1}{2}`.

    We can express the Matérn covariance function as

    .. math::
        k_{\nu}(x_0, x_1) = \kappa_{\nu}(\sqrt{2 \nu} \lVert x_0 - x_1 \rVert_2).

    If :math:`\nu = p + \frac{1}{2}` for some nonnegative integer :math:`p`, then
    :math:`\kappa_\nu(r)` and all its derivatives are products of an exponential
    and a polynomial of degree :math:`p`.
    This function computes the coefficients of the polynomial.
    """

    if n == 0:
        return half_integer_matern_polynomial(p)

    poly = half_integer_matern_derivative_polynomial(p, n - 1)

    return poly.differentiate() - poly
