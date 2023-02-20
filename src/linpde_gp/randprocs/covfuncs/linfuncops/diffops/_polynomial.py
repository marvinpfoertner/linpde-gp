from __future__ import annotations

from fractions import Fraction
import itertools
from collections.abc import Iterable
import numpy as np
from jax import numpy as jnp


class RationalPolynomial:
    def __init__(self, coeffs: Iterable[Fraction]) -> None:
        coeffs = tuple(coeffs)

        if len(coeffs) < 1:
            raise ValueError("The polynomial must have at least one coefficient")

        self._coeffs = coeffs
        self._coeffs_floating = np.asarray(self._coeffs, dtype=np.double)

    @property
    def coefficients(self) -> tuple[Fraction]:
        return self._coeffs

    @property
    def degree(self) -> int:
        return len(self._coeffs) - 1

    def __repr__(self) -> str:
        return " + ".join(
            f"{coeff.numerator}/{coeff.denominator} * x^{k}"
            for k, coeff in enumerate(self._coeffs)
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        res = np.full_like(x, self._coeffs_floating[self.degree])

        for k in range(self.degree - 1, -1, -1):
            res *= x
            res += self._coeffs_floating[k]

        return res

    def jax(self, x: jnp.ndarray) -> jnp.ndarray:
        res = jnp.full_like(x, self._coeffs_floating[self.degree])

        for k in range(self.degree - 1, -1, -1):
            res *= x
            res += self._coeffs_floating[k]

        return res

    def differentiate(self) -> RationalPolynomial:
        return RationalPolynomial(
            coeff * k for k, coeff in enumerate(self._coeffs[1:], start=1)
        )

    def __neg__(self) -> RationalPolynomial:
        return RationalPolynomial(-coeff for coeff in self._coeffs)

    def __add__(self, other: RationalPolynomial) -> RationalPolynomial:
        return RationalPolynomial(
            (
                coeff0 + coeff1
                for coeff0, coeff1 in itertools.zip_longest(
                    self._coeffs, other._coeffs, fillvalue=0
                )
            )
        )

    def __sub__(self, other: RationalPolynomial) -> RationalPolynomial:
        return RationalPolynomial(
            (
                coeff0 - coeff1
                for coeff0, coeff1 in itertools.zip_longest(
                    self._coeffs, other._coeffs, fillvalue=0
                )
            )
        )

    def __lshift__(self, degrees: int) -> RationalPolynomial:
        assert 0 <= degrees <= self.degree

        if any(coeff != 0 for coeff in self._coeffs[:degrees]):
            raise ValueError(f"The first {degrees} of the polynomial are not all zeros")

        return RationalPolynomial(self._coeffs[degrees:])
