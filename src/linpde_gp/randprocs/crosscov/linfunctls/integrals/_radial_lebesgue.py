from jax import numpy as jnp
import numpy as np
from probnum.randprocs import covfuncs

from linpde_gp import functions, linfunctls

from .._base import LinearFunctionalProcessVectorCrossCovariance


class UnivariateRadialCovarianceFunctionLebesgueIntegral(
    LinearFunctionalProcessVectorCrossCovariance
):
    def __init__(
        self,
        radial_covfunc: covfuncs.CovarianceFunction,
        integral: linfunctls.LebesgueIntegral,
        radial_antideriv: functions.JaxFunction,
        reverse: bool = False,
    ):
        assert radial_covfunc.input_shape == ()

        super().__init__(
            covfunc=radial_covfunc,
            linfunctl=integral,
            reverse=reverse,
        )

        self._radial_antideriv = radial_antideriv

    @property
    def integral(self) -> linfunctls.LebesgueIntegral:
        return self.linfunctl

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        l = self.covfunc.lengthscale
        a, b = self.integral.domain

        return l * (
            (-1) ** (b < x) * self._radial_antideriv(np.abs(b - x) / l)
            - (-1) ** (a < x) * self._radial_antideriv(np.abs(a - x) / l)
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        l = self.covfunc.lengthscale
        a, b = self.integral.domain

        return l * (
            (-1) ** (b < x) * self._radial_antideriv.jax(jnp.abs(b - x) / l)
            - (-1) ** (a < x) * self._radial_antideriv.jax(jnp.abs(a - x) / l)
        )


def univariate_radial_covfunc_lebesgue_integral_lebesgue_integral(
    k: covfuncs.CovarianceFunction,
    integral0: linfunctls.LebesgueIntegral,
    integral1: linfunctls.LebesgueIntegral,
    radial_antideriv_2: functions.JaxFunction,
):
    l = k.lengthscale

    a, b = integral0.domain
    c, d = integral1.domain

    return l**2 * (
        radial_antideriv_2(np.abs(b - c) / l)
        - radial_antideriv_2(np.abs(a - c) / l)
        - radial_antideriv_2(np.abs(b - d) / l)
        + radial_antideriv_2(np.abs(a - d) / l)
    )
