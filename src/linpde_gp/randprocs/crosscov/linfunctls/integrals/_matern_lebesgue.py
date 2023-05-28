from jax import numpy as jnp
import numpy as np
from probnum.typing import ScalarType

from linpde_gp import functions, linfunctls
from linpde_gp.randprocs import covfuncs

from .._base import LinearFunctionalProcessVectorCrossCovariance


class HalfIntegerMaternRadialAntiderivative(functions.JaxFunction):
    def __init__(self, matern: covfuncs.Matern) -> None:
        super().__init__(input_shape=(), output_shape=())

        self._matern = matern
        self._sqrt_2nu = np.sqrt(2 * self._matern.nu)
        self._neg_inv_sqrt_2nu = -(1.0 / self._sqrt_2nu)

        # Compute the polynomial part of the function
        p_i = functions.RationalPolynomial(
            covfuncs.Matern.half_integer_coefficients(matern.p)
        )

        poly = p_i

        for _ in range(matern.p):
            p_i = p_i.differentiate()

            poly += p_i

        self._poly = poly

    def _evaluate(  # pylint: disable=arguments-renamed
        self, r: np.ndarray
    ) -> np.ndarray:
        return self._neg_inv_sqrt_2nu * (
            np.exp(-self._sqrt_2nu * r) * self._poly(self._sqrt_2nu * r)
            - self._poly.coefficients[0]
        )

    def _evaluate_jax(  # pylint: disable=arguments-renamed
        self, r: jnp.ndarray
    ) -> jnp.ndarray:
        return self._neg_inv_sqrt_2nu * (
            jnp.exp(-self._sqrt_2nu * r) * self._poly.jax(self._sqrt_2nu * r)
            - self._poly.coefficients[0]
        )


class HalfIntegerMaternRadialSecondAntiderivative(functions.JaxFunction):
    def __init__(self, matern: covfuncs.Matern) -> None:
        super().__init__(input_shape=(), output_shape=())

        self._matern = matern

        self._sqrt_2nu = np.sqrt(2 * self._matern.nu)
        self._neg_inv_2nu = -(1.0 / (2 * self._matern.nu))

        # Compute the polynomial part of the function
        p_i = functions.RationalPolynomial(
            covfuncs.Matern.half_integer_coefficients(matern.p)
        )

        poly = p_i

        for i in range(1, matern.p + 1):
            p_i = p_i.differentiate()

            poly += (i + 1) * p_i

        self._poly = poly

    def _evaluate(  # pylint: disable=arguments-renamed
        self, r: np.ndarray
    ) -> np.ndarray:
        return self._neg_inv_2nu * (
            np.exp(-self._sqrt_2nu * r) * self._poly(self._sqrt_2nu * r)
            - self._poly.coefficients[0]
        )

    def _evaluate_jax(  # pylint: disable=arguments-renamed
        self, r: jnp.ndarray
    ) -> jnp.ndarray:
        return self._neg_inv_2nu * (
            jnp.exp(-self._sqrt_2nu * r) * self._poly.jax(self._sqrt_2nu * r)
            - self._poly.coefficients[0]
        )


class UnivariateHalfIntegerMaternLebesgueIntegral(
    LinearFunctionalProcessVectorCrossCovariance
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        integral: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        assert matern.input_shape == ()

        super().__init__(
            covfunc=matern,
            linfunctl=integral,
            reverse=reverse,
        )

        self._matern_radial_antideriv = HalfIntegerMaternRadialAntiderivative(
            self.matern
        )

    @property
    def matern(self) -> covfuncs.Matern:
        return self.covfunc

    @property
    def integral(self) -> linfunctls.LebesgueIntegral:
        return self.linfunctl

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        l = self.matern.lengthscale
        a, b = self.integral.domain

        return l * (
            (-1) ** (b < x) * self._matern_radial_antideriv(np.abs(b - x) / l)
            - (-1) ** (a < x) * self._matern_radial_antideriv(np.abs(a - x) / l)
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        l = self.matern.lengthscale
        a, b = self.integral.domain

        return l * (
            (-1) ** (b < x) * self._matern_radial_antideriv.jax(jnp.abs(b - x) / l)
            - (-1) ** (a < x) * self._matern_radial_antideriv.jax(jnp.abs(a - x) / l)
        )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    UnivariateHalfIntegerMaternLebesgueIntegral
)
def _(self, kL_or_Lk: UnivariateHalfIntegerMaternLebesgueIntegral, /) -> ScalarType:
    matern_radial_antideriv_2 = HalfIntegerMaternRadialSecondAntiderivative(
        kL_or_Lk.matern
    )

    l = kL_or_Lk.matern.lengthscales

    a, b = self.domain
    c, d = kL_or_Lk.integral.domain

    return l**2 * (
        matern_radial_antideriv_2(np.abs(b - c) / l)
        - matern_radial_antideriv_2(np.abs(a - c) / l)
        - matern_radial_antideriv_2(np.abs(b - d) / l)
        + matern_radial_antideriv_2(np.abs(a - d) / l)
    )
