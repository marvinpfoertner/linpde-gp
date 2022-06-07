from jax import numpy as jnp
import numpy as np
from probnum.typing import ScalarType

from linpde_gp import linfunctls
from linpde_gp.randprocs import kernels

from ... import _pv_crosscov


class Matern_Identity_LebesgueIntegral(_pv_crosscov.ProcessVectorCrossCovariance):
    def __init__(
        self,
        matern: kernels.Matern,
        integral: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        self._matern = matern
        self._integral = integral
        self._reverse = bool(reverse)

        assert self._matern.input_shape == self._integral.input_domain_shape

        super().__init__(
            randproc_input_shape=self._matern.input_shape,
            randproc_output_shape=(),
            randvar_shape=self._integral.output_shape,
            reverse=reverse,
        )

    @property
    def matern(self) -> kernels.Matern:
        return self._matern

    @property
    def integral(self) -> linfunctls.LebesgueIntegral:
        return self._integral

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        ell = self._matern.lengthscale
        a, b = self._integral.domain

        # adapted from `probnum.quad.kernel_embeddings._matern_lebesgue`
        return (
            1.0
            / (105.0 * ell**2)
            * (
                96.0 * np.sqrt(7.0) * ell**3
                - np.exp(np.sqrt(7.0) * (x - b) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    - 231.0 * ell**2 * (x - b)
                    + 63.0 * np.sqrt(7.0) * ell * (x - b) ** 2
                    - 49.0 * (x - b) ** 3
                )
                - np.exp(np.sqrt(7.0) * (a - x) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    + 231.0 * ell**2 * (x - a)
                    + 63.0 * np.sqrt(7.0) * ell * (x - a) ** 2
                    + 49.0 * (x - a) ** 3
                )
            )
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        ell = self._matern.lengthscale
        a, b = self._integral.domain

        # adapted from `probnum.quad.kernel_embeddings._matern_lebesgue`
        return (
            1.0
            / (105.0 * ell**2)
            * (
                96.0 * jnp.sqrt(7.0) * ell**3
                - jnp.exp(jnp.sqrt(7.0) * (x - b) / ell)
                * (
                    48.0 * jnp.sqrt(7.0) * ell**3
                    - 231.0 * ell**2 * (x - b)
                    + 63.0 * jnp.sqrt(7.0) * ell * (x - b) ** 2
                    - 49.0 * (x - b) ** 3
                )
                - jnp.exp(jnp.sqrt(7.0) * (a - x) / ell)
                * (
                    48.0 * jnp.sqrt(7.0) * ell**3
                    + 231.0 * ell**2 * (x - a)
                    + 63.0 * jnp.sqrt(7.0) * ell * (x - a) ** 2
                    + 49.0 * (x - a) ** 3
                )
            )
        )


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(
    self, pv_crosscov: Matern_Identity_LebesgueIntegral, /, argnum: int = 0
) -> ScalarType:
    if not (
        (pv_crosscov.reverse and argnum == 1)
        or (not pv_crosscov.reverse and argnum == 2)
    ):
        raise ValueError("TODO")

    if self.domain != pv_crosscov.integral.domain:
        raise NotImplementedError()

    # adapted from `probnum.quad.kernel_embeddings._matern_lebesgue`
    ell = pv_crosscov.matern.lengthscale
    a, b = self.domain
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
