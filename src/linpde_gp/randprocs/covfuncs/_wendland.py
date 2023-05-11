import fractions
from collections.abc import Iterable

import numpy as np
from jax import numpy as jnp
from linpde_gp import functions


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
        """The maximal dimension :math:`d` of the input space :math:`\mathbb{R}^d`, on
        which the associated Wendland kernel is positive definite."""
        return self._d

    @property
    def k(self) -> int:
        """Smoothness parameter. The associated Wendland function :math:`\phi_{d, k}` is
        :math:`2k`-times continuously differentiable."""
        return self._k

    @property
    def l(self) -> int:
        """Degree :math:`\lfloor \frac{d}{2} \rfloor + k + 1` of the Wendland
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

    def _evaluate(self, r: np.ndarray) -> np.ndarray:
        return np.where(r <= 1, self._p_dk(r), np.zeros_like(r))

    def _evaluate_jax(self, r: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(r <= 1, self._p_dk(r), jnp.zeros_like(r))


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
