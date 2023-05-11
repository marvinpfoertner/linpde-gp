from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
import functools
import itertools

from jax import numpy as jnp
import numpy as np
from probnum.typing import FloatLike
from pykeops.numpy import LazyTensor, Pm

from . import _jax
from ._constant import Constant


class Monomial(_jax.JaxFunction):
    def __init__(self, degree: int) -> None:
        super().__init__(input_shape=(), output_shape=())

        degree = int(degree)

        if degree < 0:
            raise ValueError("The degree of the monomial must not be negative.")

        self._degree = degree

    @property
    def degree(self) -> int:
        return self._degree

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return x**self._degree

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return x**self._degree


class Polynomial(_jax.JaxFunction):
    def __init__(self, coeffs: Iterable[FloatLike]) -> None:
        super().__init__(input_shape=(), output_shape=())

        coeffs = tuple(float(coeff) for coeff in coeffs)

        if len(coeffs) < 1:
            coeffs = (0.0,)

        self._coeffs = coeffs

    @property
    def coefficients(self) -> tuple[float]:
        return self._coeffs

    @property
    def degree(self) -> int:
        return len(self._coeffs) - 1

    def __repr__(self) -> str:
        return " + ".join(f"{coeff} * x^{k}" for k, coeff in enumerate(self._coeffs))

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        res = np.full_like(x, self._coeffs[self.degree])

        for k in range(self.degree - 1, -1, -1):
            res *= x
            res += self._coeffs[k]

        return res

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        res = jnp.full_like(x, self._coeffs[self.degree])

        for k in range(self.degree - 1, -1, -1):
            res *= x
            res += self._coeffs[k]

        return res

    def _evaluate_keops(self, x: LazyTensor) -> LazyTensor:
        res = Pm(self._coeffs[self.degree])

        for k in range(self.degree - 1, -1, -1):
            res *= x
            res += Pm(self._coeffs[k])

        return res

    def differentiate(self) -> Polynomial:
        return Polynomial(
            coeff * k for k, coeff in enumerate(self._coeffs[1:], start=1)
        )

    def integrate(self) -> Polynomial:
        return Polynomial(
            (0.0,) + tuple(coeff / (i + 1) for i, coeff in enumerate(self._coeffs))
        )

    def __neg__(self) -> Polynomial:
        return Polynomial(-coeff for coeff in self._coeffs)

    @functools.singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)

    @__add__.register
    def _(self, other: Constant):
        return Polynomial((self._coeffs[0] + other.value,) + self._coeffs[1:])

    @functools.singledispatchmethod
    def __sub__(self, other):
        return super().__sub__(other)

    @functools.singledispatchmethod
    def __rmul__(self, other):
        try:
            other = float(other)
        except TypeError:
            return super().__rmul__(other)

        return Polynomial(other * coeff for coeff in self._coeffs)

    @functools.singledispatchmethod
    def __floordiv__(self, other):
        return NotImplemented


@Polynomial.__add__.register  # pylint: disable=no-member
def _(self, other: Polynomial) -> Polynomial:
    return Polynomial(
        (
            coeff0 + coeff1
            for coeff0, coeff1 in itertools.zip_longest(
                self.coefficients, other.coefficients, fillvalue=0
            )
        )
    )


@Polynomial.__sub__.register  # pylint: disable=no-member
def _(self, other: Polynomial) -> Polynomial:
    return Polynomial(
        (
            coeff0 - coeff1
            for coeff0, coeff1 in itertools.zip_longest(
                self.coefficients, other.coefficients, fillvalue=0
            )
        )
    )


@Polynomial.__floordiv__.register  # pylint: disable=no-member
def _(self, monomial: Monomial) -> Polynomial:
    if not 0 <= monomial.degree <= self.degree:
        raise ValueError(
            "The degree of the monomial is larger than the degree of the polynomial"
        )

    if any(coeff != 0 for coeff in self.coefficients[: monomial.degree]):
        raise ValueError(
            f"The first {monomial.degree} of the polynomial are not all zeros"
        )

    return Polynomial(self.coefficients[monomial.degree :])


class RationalPolynomial(Polynomial):
    def __init__(self, coeffs: Iterable[Fraction]) -> None:
        coeffs = tuple(Fraction(coeff) for coeff in coeffs)

        if len(coeffs) < 1:
            coeffs = (Fraction(0),)

        self._rational_coeffs = coeffs

        super().__init__(coeffs)

    @property
    def rational_coefficients(self) -> tuple[Fraction]:
        return self._rational_coeffs

    def __repr__(self) -> str:
        if all(coeff == 0 for coeff in self._rational_coeffs):
            return "0"

        return " ".join(
            str(coeff)
            if k == 0
            else (
                "".join(
                    [
                        "+" if coeff > 0 else "-",
                        f" {str(abs(coeff))}" if abs(coeff) != 1 else "",
                        f" x^{k}",
                    ]
                )
            )
            for k, coeff in enumerate(self._rational_coeffs)
            if coeff != 0
        )

    def differentiate(self) -> RationalPolynomial:
        return RationalPolynomial(
            coeff * k for k, coeff in enumerate(self._rational_coeffs[1:], start=1)
        )

    def integrate(self) -> RationalPolynomial:
        return RationalPolynomial(
            (Fraction(0, 1),)
            + tuple(coeff / (i + 1) for i, coeff in enumerate(self._rational_coeffs))
        )

    def __neg__(self) -> RationalPolynomial:
        return RationalPolynomial(-coeff for coeff in self._rational_coeffs)

    @functools.singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)

    @functools.singledispatchmethod
    def __sub__(self, other):
        return super().__sub__(other)

    @functools.singledispatchmethod
    def __mul__(self, other):
        return super().__mul__(other)

    @functools.singledispatchmethod
    def __rmul__(self, other):
        return super().__rmul__(other)

    @functools.singledispatchmethod
    def __divmod__(self, other):
        return NotImplemented

    @functools.singledispatchmethod
    def __floordiv__(self, other):
        return NotImplemented


@RationalPolynomial.__add__.register  # pylint: disable=no-member
def _(self, other: RationalPolynomial) -> RationalPolynomial:
    return RationalPolynomial(
        (
            coeff0 + coeff1
            for coeff0, coeff1 in itertools.zip_longest(
                self.rational_coefficients, other.rational_coefficients, fillvalue=0
            )
        )
    )


@RationalPolynomial.__sub__.register  # pylint: disable=no-member
def _(self, other: RationalPolynomial) -> RationalPolynomial:
    return RationalPolynomial(
        (
            coeff0 - coeff1
            for coeff0, coeff1 in itertools.zip_longest(
                self.rational_coefficients, other.rational_coefficients, fillvalue=0
            )
        )
    )


@RationalPolynomial.__mul__.register  # pylint: disable=no-member
def _mul_rational_polynomial(self, other: RationalPolynomial) -> RationalPolynomial:
    return RationalPolynomial(
        (
            sum(
                self.rational_coefficients[i] * other.rational_coefficients[k - i]
                for i in range(max(0, k - other.degree), min(k, self.degree) + 1)
            )
            for k in range(self.degree + other.degree + 1)
        )
    )


@RationalPolynomial.__mul__.register(Fraction)  # pylint: disable=no-member
@RationalPolynomial.__mul__.register(int)  # pylint: disable=no-member
@RationalPolynomial.__rmul__.register(Fraction)  # pylint: disable=no-member
@RationalPolynomial.__rmul__.register(int)  # pylint: disable=no-member
def _mul_rational_polynomial_fraction(self, other: Fraction | int):
    return RationalPolynomial((other * coeff for coeff in self.rational_coefficients))


@RationalPolynomial.__mul__.register  # pylint: disable=no-member
def _mul_rational_polynomial_monomial(self, other: Monomial) -> RationalPolynomial:
    return RationalPolynomial(
        (
            *(0 for _ in range(other.degree)),
            *self.rational_coefficients,
        )
    )


@RationalPolynomial.__divmod__.register  # pylint: disable=no-member
def _divmod_rational_polynomial(self, other: RationalPolynomial) -> RationalPolynomial:
    rem = self
    quot_coeffs = []

    for i in reversed(range(other.degree, self.degree + 1)):
        c = rem.rational_coefficients[i] / other.rational_coefficients[-1]

        quot_coeffs.append(c)

        rem = rem - c * other * Monomial(i - other.degree)

    return RationalPolynomial(reversed(quot_coeffs)), rem


@RationalPolynomial.__floordiv__.register  # pylint: disable=no-member
def _(self, monomial: Monomial) -> RationalPolynomial:
    if not 0 <= monomial.degree <= self.degree:
        raise ValueError(
            "The degree of the monomial is larger than the degree of the polynomial"
        )

    if any(coeff != 0 for coeff in self.rational_coefficients[: monomial.degree]):
        raise ValueError(
            f"The first {monomial.degree} of the polynomial are not all zeros"
        )

    return RationalPolynomial(self.rational_coefficients[monomial.degree :])
