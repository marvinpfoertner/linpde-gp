import functools
import operator

import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from linpde_gp import linfuncops

from . import _linfunctl


class ScaledLinearFunctional(_linfunctl.LinearFunctional):
    def __init__(
        self, linfunctl: _linfunctl.LinearFunctional, scalar: ScalarLike
    ) -> None:
        self._linfunctl = linfunctl

        super().__init__(
            input_shapes=self._linfunctl.input_shapes,
            output_shape=self._linfunctl.output_shape,
        )

        if not np.ndim(scalar) == 0:
            raise ValueError()

        self._scalar = np.asarray(scalar, dtype=np.double)

    @property
    def linfunctl(self) -> _linfunctl.LinearFunctional:
        return self._linfunctl

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._scalar * self._linfunctl(f, **kwargs)

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    def __rmul__(self, other) -> _linfunctl.LinearFunctional:
        if np.ndim(other) == 0:
            return ScaledLinearFunctional(
                linfunctl=self._linfunctl,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)


class SumLinearFunctional(_linfunctl.LinearFunctional):
    def __init__(self, *summands: _linfunctl.LinearFunctional) -> None:
        self._summands = tuple(summands)

        input_domain_shape = self._summands[0].input_domain_shape
        input_codomain_shape = self._summands[0].input_codomain_shape
        output_shape = self._summands[0].output_shape

        assert all(
            summand.input_domain_shape == input_domain_shape
            for summand in self._summands
        )
        assert all(
            summand.input_codomain_shape == input_codomain_shape
            for summand in self._summands
        )
        assert all(summand.output_shape == output_shape for summand in self._summands)

        super().__init__(
            input_shapes=(input_domain_shape, input_codomain_shape),
            output_shape=output_shape,
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return functools.reduce(
            operator.add, (summand(f, **kwargs) for summand in self._summands)
        )

    @__call__.register
    def _(self, randproc: pn.randprocs.RandomProcess, /) -> pn.randprocs.RandomProcess:
        return super().__call__(randproc)


class CompositeLinearFunctional(_linfunctl.LinearFunctional):
    def __init__(
        self,
        *,
        linop: pn.linops.LinearOperatorLike | None,
        linfunctl: _linfunctl.LinearFunctional,
        linfuncop: linfuncops.LinearFunctionOperator | None,
    ) -> None:
        if linfuncop:
            assert linfuncop.output_shapes == linfunctl.input_shapes

        if linop is not None:
            assert linfunctl.output_ndim == 1
            assert linfunctl.output_shape == linop.shape[1:]

        self._linop = pn.linops.aslinop(linop) if linop is not None else None
        self._linfunctl = linfunctl
        self._linfuncop = linfuncop

        super().__init__(
            input_shapes=(
                self._linfunctl.input_shapes
                if self._linfuncop is None
                else self._linfuncop.input_shapes
            ),
            output_shape=(
                self._linfunctl.output_shape
                if self._linop is None
                else self._linop.shape[0:1]
            ),
        )

    @property
    def linop(self) -> pn.linops.LinearOperator | None:
        return self._linop

    @property
    def linfunctl(self) -> _linfunctl.LinearFunctional:
        return self._linfunctl

    @property
    def linfuncop(self) -> linfuncops.LinearFunctionOperator | None:
        return self._linfuncop

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return super().__call__(f, **kwargs)

    @__call__.register
    def _(self, f: pn.functions.Function, /, **kwargs):
        Lf = f

        if self._linfuncop is not None:
            Lf = self._linfuncop(Lf, **kwargs)

        Lf = self._linfunctl(Lf, **kwargs)

        if self._linop is not None:
            Lf = self._linop(Lf, axis=-1)

        return Lf

    def __matmul__(self, other):
        if isinstance(other, linfuncops.LinearFunctionOperator):
            return CompositeLinearFunctional(
                linop=self._linop,
                linfunctl=self._linfunctl,
                linfuncop=(
                    other if self._linfuncop is None else self._linfuncop @ other
                ),
            )

        return super().__matmul__(other)

    def __rmatmul__(self, other):
        if isinstance(other, (np.ndarray, pn.linops.LinearOperator)):
            return CompositeLinearFunctional(
                linop=other if self._linop is None else other @ self._linop,
                linfunctl=self._linfunctl,
                linfuncop=self._linfuncop,
            )

        return super().__rmatmul__(other)
