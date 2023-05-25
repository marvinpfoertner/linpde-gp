from jax import numpy as jnp
import numpy as np
from probnum.typing import ScalarType

from linpde_gp import functions, linfunctls
from linpde_gp.randprocs import covfuncs

from ... import _pv_crosscov


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

    def _evaluate(self, r: np.ndarray) -> np.ndarray:
        return self._neg_inv_sqrt_2nu * (
            np.exp(-self._sqrt_2nu * r) * self._poly(self._sqrt_2nu * r)
            - self._poly.coefficients[0]
        )

    def _evaluate_jax(self, r: jnp.ndarray) -> jnp.ndarray:
        return self._neg_inv_sqrt_2nu * (
            jnp.exp(-self._sqrt_2nu * r) * self._poly.jax(self._sqrt_2nu * r)
            - self._poly.coefficients[0]
        )


class HalfIntegerMatern_Identity_LebesgueIntegral(
    _pv_crosscov.ProcessVectorCrossCovariance
):
    def __init__(
        self,
        matern: covfuncs.Matern,
        integral: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        self._matern = matern
        self._integral = integral
        self._reverse = bool(reverse)

        assert self._matern.input_shape == ()
        assert self._integral.input_domain_shape == ()
        assert self._integral.input_codomain_shape == ()
        assert self._integral.output_shape == ()

        super().__init__(
            randproc_input_shape=(),
            randproc_output_shape=(),
            randvar_shape=(),
            reverse=reverse,
        )

        self._matern_radial_antideriv = HalfIntegerMaternRadialAntiderivative(
            self._matern
        )

    @property
    def matern(self) -> covfuncs.Matern:
        return self._matern

    @property
    def integral(self) -> linfunctls.LebesgueIntegral:
        return self._integral

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        l = self._matern.lengthscale
        a, b = self._integral.domain

        return l * (
            (-1) ** (b < x) * self._matern_radial_antideriv(np.abs(b - x) / l)
            - (-1) ** (a < x) * self._matern_radial_antideriv(np.abs(a - x) / l)
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        l = self._matern.lengthscale
        a, b = self._integral.domain

        return l * (
            (-1) ** (b < x) * self._matern_radial_antideriv.jax(jnp.abs(b - x) / l)
            - (-1) ** (a < x) * self._matern_radial_antideriv.jax(jnp.abs(a - x) / l)
        )


@linfunctls.LebesgueIntegral.__call__.register(  # pylint: disable=no-member
    HalfIntegerMatern_Identity_LebesgueIntegral
)
def _(self, kL_or_Lk: HalfIntegerMatern_Identity_LebesgueIntegral, /) -> ScalarType:
    if self.domain != kL_or_Lk.integral.domain:
        import scipy.integrate

        return scipy.integrate.dblquad(
            kL_or_Lk.matern,
            *self.domain,
            *kL_or_Lk._L.domain,
        )[0]

    # adapted from `probnum.quad.kernel_embeddings._matern_lebesgue`
    ell = kL_or_Lk.matern.lengthscales
    a, b = self.domain

    match kL_or_Lk.matern.p:
        case 0:
            r = b - a

            return 2.0 * ell * (r + ell * (np.exp(-r / ell) - 1.0))
        case 3:
            c = np.sqrt(7.0) * (b - a)

            return (
                1.0
                / (105.0 * ell)
                * (
                    2.0
                    * np.exp(-c / ell)
                    * (
                        7.0 * np.sqrt(7.0) * (b**3 - a**3)
                        + 84.0 * b**2 * ell
                        + 57.0 * np.sqrt(7.0) * b * ell**2
                        + 105.0 * ell**3
                        + 21.0 * a**2 * (np.sqrt(7.0) * b + 4.0 * ell)
                        - 3.0
                        * a
                        * (
                            7.0 * np.sqrt(7.0) * b**2
                            + 56.0 * b * ell
                            + 19.0 * np.sqrt(7.0) * ell**2
                        )
                    )
                    - 6.0 * ell**2 * (35.0 * ell - 16.0 * c)
                )
            )

    raise NotImplementedError
