from __future__ import annotations

import abc
import functools
import operator
from typing import Type

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ScalarLike, ShapeLike, ShapeType


class Covariance(abc.ABC):
    r"""Covariance between two random variables.

    The covariance between two :class:`pn.randvars.RandomVariable`s :code:`x0` and
    :code:`x1` with shapes :code:`shape0` and :code:`shape1`, respectively, can be
    represented as an array :code:`cov_array` with shape :code:`shape0 + shape1` and
    entries

    ..math::
        \texttt{cov_array}[i_0, \dotsc, i_{d_0 - 1}, j_0, \dotsc, j_{d_1 - 1}]
        = \operatorname{Cov}(
            \texttt{x0}[i_0, \dotsc, i_{d_0 - 1}],
            \texttt{x1}[j_0, \dotsc, j_{d_1 - 1}]
        ).

    However, especially for Gaussian (process) inference, the covariance is best
    represented as the matrix

    ..math::
        \texttt{cov_matrix}[i, j]
        = \operatorname{Cov}(\texttt{x0_vec}[i], \texttt{x1_vec}[j])

    with shape :code:`(size0, size1)`, where :code:`x0_vec` and :code:`x1_vec` are
    random vectors with shapes :code:`(size0,)` and :code:`(size1,)` obtained by
    rearranging the entries of :code:`x0` and :code:`x1`.

    :class:`Covariance` objects provide simultaneous access to both representations of
    the covariance between two :class:`pn.randvars.RandomVariable`s.

    Parameters
    ----------
    shape0
        The shape of the random variable :code:`x0`.

    shape1
        The shape of the random variable :code:`x1`.
    """

    def __init__(self, shape0: ShapeLike, shape1: ShapeLike) -> None:
        self._shape0 = pn.utils.as_shape(shape0)
        self._shape1 = pn.utils.as_shape(shape1)

    @property
    def shape0(self) -> ShapeType:
        """The shape of the 'left' random variable :code:`x0`."""
        return self._shape0

    @property
    def ndim0(self) -> int:
        return len(self.shape0)

    @functools.cached_property
    def size0(self) -> int:
        return functools.reduce(operator.mul, self.shape0, 1)

    @property
    def shape1(self) -> ShapeType:
        """The shape of the 'right' random variable :code:`x1`."""
        return self._shape1

    @property
    def ndim1(self) -> int:
        return len(self.shape1)

    @functools.cached_property
    def size1(self) -> int:
        return functools.reduce(operator.mul, self.shape1, 1)

    @property
    @abc.abstractmethod
    def array(self) -> np.ndarray:
        r"""The array representation of the covariance, i.e. an array :code:`cov_array`
        with shape :attr:`shape0` :code:` + ` :attr:`shape1` and entries

        ..math::
            \texttt{cov_array}[i_0, \dotsc, i_{d_0 - 1}, j_0, \dotsc, j_{d_1 - 1}]
            = \operatorname{Cov}(
                \texttt{x0}[i_0, \dotsc, i_{d_0 - 1}],
                \texttt{x1}[j_0, \dotsc, j_{d_1 - 1}]
            ).
        """

    @property
    @abc.abstractmethod
    def linop(self) -> pn.linops.LinearOperator:
        r"""Matrix-free representation of the covariance matrix, i.e.

            ..math::
                \texttt{cov_matrix}[i, j]
                = \operatorname{Cov}(\texttt{x0_vec}[i], \texttt{x1_vec}[j]),

            where :code:`x0_vec` and :code:`x1_vec` are random vectors with shapes
            :code:`(size0,)` and :code:`(size1,)` obtained by flattening :code:`x0` and
            :code:`x1`.
        def"""

    @property
    @abc.abstractmethod
    def matrix(self) -> np.ndarray:
        """Shorthand for :code:`cov.linop.todense()`."""

    def flatten0(self, event0: ArrayLike, /) -> np.ndarray:
        event0 = np.asarray(event0)

        if event0.shape != self.shape0:
            raise ValueError(
                "The shape of the event must be the same as `shape0`, but"
                f"{event0.shape} != {self.shape0}."
            )

        return np.reshape(event0, (-1,), order="C")

    def flatten1(self, event1: ArrayLike, /) -> np.ndarray:
        event1 = np.asarray(event1)

        if event1.shape != self.shape1:
            raise ValueError(
                "The shape of the event must be the same as `shape1`, but"
                f"{event1.shape} != {self.shape1}."
            )

        return np.reshape(event1, (-1,), order="C")


class ArrayCovariance(Covariance):
    @staticmethod
    def from_scalar(var: ScalarLike) -> ArrayCovariance:
        return ArrayCovariance(
            np.asarray(var),
            shape0=(),
            shape1=(),
        )

    def __init__(
        self,
        cov_array: ArrayLike,
        shape0: ShapeLike,
        shape1: ShapeLike,
    ) -> None:
        super().__init__(shape0, shape1)

        self._cov_array = np.asarray(cov_array)

        if self._cov_array.shape != self.shape0 + self.shape1:
            raise ValueError(
                "The shape of `cov_array` must be `shape0 + shape1`, but"
                f"`{self._cov_array.shape} != {self.shape0} + {self.shape1}`."
            )

    @property
    def array(self) -> np.ndarray:
        return self._cov_array

    @property
    def linop(self) -> pn.linops.LinearOperator:
        return pn.linops.aslinop(self.matrix)

    @functools.cached_property
    def matrix(self) -> np.ndarray:
        return np.reshape(self.array, (self.size0, self.size1), order="C")

    def __neg__(self) -> ArrayCovariance:
        return -1.0 * self

    def __add__(self, other) -> Covariance | Type[NotImplemented]:
        if (
            isinstance(other, ArrayCovariance)
            and self.shape0 == other.shape0
            and self.shape1 == other.shape1
        ):
            return ArrayCovariance(self.array + other.array, self.shape0, self.shape1)
        if isinstance(other, LinearOperatorCovariance):
            return other + self
        return NotImplemented

    def __sub__(self, other) -> Covariance | Type[NotImplemented]:
        return self + (-other)

    def __rmul__(self, other) -> ArrayCovariance | Type[NotImplemented]:
        if np.ndim(other) == 0:
            return ArrayCovariance(other * self.array, self.shape0, self.shape1)
        return NotImplemented


class LinearOperatorCovariance(Covariance):
    def __init__(
        self,
        cov_linop: pn.linops.LinearOperatorLike,
        shape0: ShapeLike,
        shape1: ShapeLike,
    ) -> None:
        super().__init__(shape0, shape1)

        self._cov_linop = pn.linops.aslinop(cov_linop)

        if self._cov_linop.shape != (self.size0, self.size1):
            raise ValueError(
                "The shape of `cov_linop` must be `(size0, size1)`, but"
                f"`{self._cov_linop.shape} != ({self.size0}, {self.size1})`."
            )

    @functools.cached_property
    def array(self) -> np.ndarray:
        return np.reshape(self.matrix, self.shape0 + self.shape1, order="C")

    @property
    def linop(self) -> pn.linops.LinearOperator:
        return self._cov_linop

    @property
    def matrix(self) -> np.ndarray:
        return self._cov_linop.todense(cache=True)

    def __neg__(self) -> LinearOperatorCovariance:
        return -1.0 * self

    def __add__(self, other) -> LinearOperatorCovariance | Type[NotImplemented]:
        if (
            isinstance(other, (ArrayCovariance, LinearOperatorCovariance))
            and self.shape0 == other.shape0
            and self.shape1 == other.shape1
        ):
            return LinearOperatorCovariance(
                self.linop + other.linop, self.shape0, self.shape1
            )
        return NotImplemented

    def __sub__(self, other) -> LinearOperatorCovariance | Type[NotImplemented]:
        return self + (-other)

    def __rmul__(self, other) -> LinearOperatorCovariance | Type[NotImplemented]:
        if np.ndim(other) == 0:
            return LinearOperatorCovariance(
                other * self.linop, self.shape0, self.shape1
            )
        return NotImplemented
