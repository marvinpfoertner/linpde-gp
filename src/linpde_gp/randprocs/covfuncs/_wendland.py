from collections.abc import Iterable
import fractions
import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from probnum.randprocs.covfuncs import IsotropicMixin
from probnum.typing import ArrayLike, ShapeLike

from linpde_gp import functions

from ._jax import JaxCovarianceFunction, JaxIsotropicMixin

_USE_KEOPS = True
try:
    from pykeops.numpy import LazyTensor
except ImportError:  # pragma: no cover
    _USE_KEOPS = False


class WendlandCovarianceFunction(
    JaxCovarianceFunction, JaxIsotropicMixin, IsotropicMixin
):
    r"""A radial covariance function with compact support obtained via the Wendland
    functions from [1]_.

    Yields sparse covariance matrices, which can speed up computations.

    Parameters
    ----------
    k :
        Smoothness parameter. The associated Wendland function is
        :math:`2k`-times continuously differentiable.

    References
    ---------
    .. [1] Holger Wendland, Scattered Data Approximation. Cambridge University Press,
           2004.
    """

    def __init__(
        self,
        input_shape: ShapeLike,
        k: int,
        lengthscales: Optional[ArrayLike] = None,
    ):
        super().__init__(input_shape=input_shape)
        self._d = int(np.prod(input_shape))
        self._k = k
        self._func = WendlandFunction(self._d, self._k)

        # Input lengthscales
        self._lengthscales = np.asarray(
            lengthscales if lengthscales is not None else 1.0,
            dtype=np.double,
        )

    @property
    def d(self):
        return self._d

    @property
    def k(self):
        return self._k

    @functools.cached_property
    def _scale_factors(self) -> np.ndarray:
        return 1 / self._lengthscales

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        scaled_dists = self._euclidean_distances(
            x0, x1, scale_factors=self._scale_factors
        )

        return self._func(scaled_dists)

    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        scaled_dists = self._euclidean_distances_jax(
            x0, x1, scale_factors=self._scale_factors
        )

        return self._func.jax(scaled_dists)

    def _keops_lazy_tensor(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> "LazyTensor":
        if not _USE_KEOPS:  # pragma: no cover
            raise ImportError()

        scaled_dists = self._euclidean_distances_keops(
            x0, x1, scale_factors=self._scale_factors
        )

        return self._func._evaluate_keops(
            scaled_dists
        )  # pylint: disable=protected-access


class WendlandPolynomial(functions.RationalPolynomial):
    r"""Polynomial part :math:`p_{d, k}` of the Wendland functions as defined in Theorem
    9.12 in [1]_.

    Parameters
    ----------
    d :
        The maximal dimension :math:`d` of the input space :math:`\mathbb{R}^d`, on
        which the associated Wendland kernel is positive definite.
    k :
        Smoothness parameter. The associated Wendland function :math:`\phi_{d, k}` is
        :math:`2k`-times continuously differentiable.

    References
    ---------
    .. [1] Holger Wendland, Scattered Data Approximation. Cambridge University Press,
           2004.
    """

    def __init__(self, d: int, k: int) -> None:
        self._d = int(d)
        self._k = int(k)

        # Initialize
        d_l_js_s = WendlandPolynomial.phi_l_poly(self.l).rational_coefficients

        # Integrate
        for _ in range(0, k):
            d_l_js_s = (
                sum(d_l_j_s / (j + 2) for j, d_l_j_s in enumerate(d_l_js_s)),
                fractions.Fraction(0),
                *(-d_l_jm2_s / j for j, d_l_jm2_s in enumerate(d_l_js_s, 2)),
            )

        # Normalize
        d_l_js_s = tuple(d_l_j_s / d_l_js_s[0] for d_l_j_s in d_l_js_s)

        super().__init__(d_l_js_s)

        assert self.degree == self.l + 2 * self._k

    @property
    def d(self) -> int:
        r"""The maximal dimension :math:`d` of the input space :math:`\mathbb{R}^d`, on
        which the associated Wendland kernel is positive definite."""
        return self._d

    @property
    def k(self) -> int:
        r"""Smoothness parameter. The associated Wendland function :math:`\phi_{d, k}`
        is :math:`2k`-times continuously differentiable."""
        return self._k

    @property
    def l(self) -> int:
        r"""Degree :math:`\lfloor \frac{d}{2} \rfloor + k + 1` of the Wendland
        polynomial."""
        return self._d // 2 + self._k + 1

    @staticmethod
    def phi_l_poly(l: int) -> functions.RationalPolynomial:
        r"""Polynomial obtained by applying the binomial theorem to
        :math:`\phi_l(r) := (1 - r)^l`."""
        return functions.RationalPolynomial(
            (
                l_choose_j if j % 2 == 0 else -l_choose_j
                for j, l_choose_j in enumerate(pascal_row(l))
            )
        )


class WendlandFunction(functions.JaxFunction):
    r"""Wendland function :math:`\phi_{d, k}` from Definition 9.11 in [1]_.

    By Theorem 9.13 in [1]_, the Wendland functions are given by

    .. math::
        \phi_{d, k}(r) =
        \begin{cases}
            p_{d,k}(r), & 0 \le r \le 1, \\
            0,          & r > 0,
        \end{cases}

    where :math:`p_{d, k}` is a Wendland polynomial. They define the radial part of the
    Wendland kernels.

    Parameters
    ----------
    d :
        The maximal dimension :math:`d` of the input space :math:`\mathbb{R}^d`, on
        which the associated Wendland kernel is still positive definite.
    k :
        Smoothness parameter. The Wendland function :math:`\phi_{d, k}` is :math:`2k`
        times continuously differentiable.

    References
    ---------
    .. [1] Holger Wendland, Scattered Data Approximation. Cambridge University Press,
           2004.
    """

    def __init__(self, d: int, k: int) -> None:
        self._p_dk = WendlandPolynomial(d, k)

        super().__init__(input_shape=(), output_shape=())

    @property
    def p_dk(self) -> WendlandPolynomial:
        """Polynomial part :math:`p_{d, k}` of the Wendland function."""
        return self._p_dk

    def _evaluate(  # pylint: disable=arguments-renamed
        self, r: np.ndarray
    ) -> np.ndarray:
        return np.where(r <= 1, self._p_dk(r), np.zeros_like(r))

    def _evaluate_jax(  # pylint: disable=arguments-renamed
        self, r: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.where(r <= 1, self._p_dk.jax(r), jnp.zeros_like(r))

    def _evaluate_keops(self, r: LazyTensor) -> LazyTensor:
        return (1 - r).ifelse(
            self._p_dk._evaluate_keops(r), 0.0
        )  # pylint: disable=protected-access


###########
# Helpers #
###########


def pascal_row(n: int) -> Iterable[int]:
    """Computes an entire row in Pascal's triangle efficiently."""
    n_choose_k = fractions.Fraction(1)

    yield int(n_choose_k)

    for k in range(1, n + 1):
        n_choose_k *= fractions.Fraction(n + 1 - k, k)

        assert n_choose_k.denominator == 1

        yield int(n_choose_k)
