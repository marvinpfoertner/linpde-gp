from __future__ import annotations

from collections.abc import Iterable
import functools

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike

from . import _jax
from ._constant import Constant
from ._polynomial import Polynomial


class Piecewise(_jax.JaxFunction):
    def __init__(
        self,
        xs: ArrayLike,
        fns: Iterable[pn.functions.Function],
    ):
        xs = np.atleast_1d(xs)

        if xs.ndim != 1:
            raise ValueError()

        self._xs = xs

        fns = tuple(fns)

        if len(fns) != self._xs.size - 1:
            raise ValueError()

        if not all(fn.input_shape == () and fn.output_shape == () for fn in fns):
            raise ValueError()

        self._fns = fns

        super().__init__(input_shape=(), output_shape=())

    @property
    def xs(self) -> np.ndarray:
        return self._xs

    @property
    def pieces(self) -> pn.functions.Function:
        return self._fns

    @property
    def num_pieces(self) -> int:
        return len(self._fns)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.piecewise(
            x,
            condlist=(
                [(self._xs[0] <= x) & (x <= self._xs[1])]
                + [
                    (self._xs[i] < x) & (x <= self._xs[i + 1])
                    for i in range(1, len(self._fns))
                ]
            ),
            funclist=self._fns,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.piecewise(
            x,
            condlist=(
                [(self._xs[0] <= x) & (x <= self._xs[1])]
                + [
                    (self._xs[i] < x) & (x <= self._xs[i + 1])
                    for i in range(1, len(self._fns))
                ]
            ),
            funclist=[fn.jax for fn in self._fns],
        )

    @functools.singledispatchmethod
    def __rmul__(self, other):
        try:
            other = float(other)
        except TypeError:
            return super().__rmul__(other)

        return Piecewise(self.xs, fns=[other * piece for piece in self.pieces])


class PiecewiseLinear(Piecewise):
    @staticmethod
    def from_points(xs: ArrayLike, ys: ArrayLike):
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        pieces = []

        for l, r, y_l, y_r in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            slope = (y_r - y_l) / (r - l)
            offset = y_l - slope * l

            pieces.append(Polynomial((offset, slope)))

        return PiecewiseLinear(xs=xs, fns=pieces)

    @functools.singledispatchmethod
    def __add__(self, other):
        return super().__add__(other)

    @functools.singledispatchmethod
    def __radd__(self, other):
        return NotImplemented

    @__add__.register
    @__radd__.register
    def _(self, other: Polynomial) -> Piecewise:
        if other.degree == 1:
            return PiecewiseLinear(
                self.xs,
                fns=[piece + other for piece in self.pieces],
            )

        return Piecewise(
            self.xs,
            fns=[piece + other for piece in self.pieces],
        )

    @__add__.register  # (Constant)
    @__radd__.register  # (Constant)
    def _(self, other: Constant):
        return PiecewiseLinear(
            self.xs,
            fns=[piece + other for piece in self.pieces],
        )

    @functools.singledispatchmethod
    def __rmul__(self, other):
        try:
            other = float(other)
        except TypeError:
            return super().__rmul__(other)

        return PiecewiseLinear(self.xs, fns=[other * piece for piece in self.pieces])


class PiecewiseConstant(Piecewise):
    def __init__(
        self,
        xs: ArrayLike,
        ys: ArrayLike,
    ):
        ys = np.atleast_1d(ys)

        if ys.ndim != 1:
            raise ValueError()

        self._ys = ys

        super().__init__(xs=xs, fns=[Constant((), y) for y in ys])

    @property
    def ys(self) -> np.ndarray:
        return self._ys

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.piecewise(
            x,
            condlist=[
                (self._xs[i] < x) & (x <= self._xs[i + 1]) for i in range(self._ys.size)
            ],
            funclist=self._ys,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.piecewise(
            x,
            condlist=[
                (self._xs[i] < x) & (x <= self._xs[i + 1]) for i in range(self._ys.size)
            ],
            funclist=self._ys,
        )
