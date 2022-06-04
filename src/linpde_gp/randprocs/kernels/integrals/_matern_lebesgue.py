from multiprocessing.sharedctypes import Value
from typing import Optional

from jax import numpy as jnp
import numpy as np

from linpde_gp.linfuncops import UndefinedLebesgueIntegral

from .._jax import JaxKernel
from .._matern import Matern


class Matern_UndefinedLebesgueIntegral_Identity(JaxKernel):
    def __init__(self, matern: Matern, lower_bound: float):
        self._matern = matern
        self._lower_bound = lower_bound

        super().__init__(self._matern.input_shape, output_shape=())

    @property
    def matern(self) -> Matern:
        return self._matern

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        ell = self._matern.lengthscale
        a = self._lower_bound
        b = x0

        return (
            1.0
            / (105.0 * ell**2)
            * (
                96.0 * np.sqrt(7.0) * ell**3
                - np.exp(np.sqrt(7.0) * (x1 - b) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    - 231.0 * ell**2 * (x1 - b)
                    + 63.0 * np.sqrt(7.0) * ell * (x1 - b) ** 2
                    - 49.0 * (x1 - b) ** 3
                )
                - np.exp(np.sqrt(7.0) * (a - x1) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    + 231.0 * ell**2 * (x1 - a)
                    + 63.0 * np.sqrt(7.0) * ell * (x1 - a) ** 2
                    + 49.0 * (x1 - a) ** 3
                )
            )
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return super()._evaluate_jax(x0, x1)


class Matern_Identity_UndefinedLebesgueIntegral(JaxKernel):
    def __init__(self, matern: Matern, lower_bound: float):
        self._matern = matern
        self._lower_bound = lower_bound

        super().__init__(self._matern.input_shape, output_shape=())

    @property
    def matern(self) -> Matern:
        return self._matern

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        ell = self._matern.lengthscale
        a = self._lower_bound
        b = x1

        return (
            1.0
            / (105.0 * ell**2)
            * (
                96.0 * np.sqrt(7.0) * ell**3
                - np.exp(np.sqrt(7.0) * (x0 - b) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    - 231.0 * ell**2 * (x0 - b)
                    + 63.0 * np.sqrt(7.0) * ell * (x0 - b) ** 2
                    - 49.0 * (x0 - b) ** 3
                )
                - np.exp(np.sqrt(7.0) * (a - x0) / ell)
                * (
                    48.0 * np.sqrt(7.0) * ell**3
                    + 231.0 * ell**2 * (x0 - a)
                    + 63.0 * np.sqrt(7.0) * ell * (x0 - a) ** 2
                    + 49.0 * (x0 - a) ** 3
                )
            )
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return super()._evaluate_jax(x0, x1)


@UndefinedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, k: Matern, /, *, argnum: int = 0):  # pylint: disable=unused-argument
    if argnum == 0:
        return Matern_UndefinedLebesgueIntegral_Identity(
            matern=k,
            lower_bound=self.lower_bound,
        )
    elif argnum == 1:
        return Matern_Identity_UndefinedLebesgueIntegral(
            matern=k,
            lower_bound=self.lower_bound,
        )

    raise ValueError("TODO")


class Matern_UndefinedLebesgueIntegral_UndefinedLebesgueIntegral(JaxKernel):
    def __init__(self, matern: Matern, lower_bound: float):
        self._matern = matern
        self._lower_bound = lower_bound

        super().__init__(self._matern.input_shape, output_shape=())

    @property
    def matern(self) -> Matern:
        return self._matern

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        assert x0 == x1
        assert x0.size == 1

        outshape = ()

        if x0.ndim == 1 and x1.ndim == 1:
            x0 = x0[0]
            x1 = x1[0]

            outshape = (1,)
        elif x0.ndim == 2 and x1.ndim == 2:
            x0 = x0[0, 0]
            x1 = x1[0, 0]

            outshape = (1, 1)

        assert x0.shape == ()
        assert x1.shape == ()

        ell = self._matern.lengthscale
        a = self._lower_bound
        b = x0
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
        ).reshape(outshape)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        return super()._evaluate_jax(x0, x1)


@UndefinedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(
    self, k: Matern_Identity_UndefinedLebesgueIntegral, /, *, argnum: int = 0
):  # pylint: disable=unused-argument
    if argnum == 0 and k.lower_bound == self.lower_bound:
        return Matern_UndefinedLebesgueIntegral_UndefinedLebesgueIntegral(
            matern=k.matern,
            lower_bound=self.lower_bound,
        )

    return super(UndefinedLebesgueIntegral, self).__call__(k, argnum=argnum)


@UndefinedLebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(
    self, k: Matern_UndefinedLebesgueIntegral_Identity, /, *, argnum: int = 0
):  # pylint: disable=unused-argument
    if argnum == 1 and k.lower_bound == self.lower_bound:
        return Matern_UndefinedLebesgueIntegral_UndefinedLebesgueIntegral(
            matern=k.matern,
            lower_bound=self.lower_bound,
        )

    return super(UndefinedLebesgueIntegral, self).__call__(k, argnum=argnum)
