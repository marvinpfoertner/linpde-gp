import functools
import operator

import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from ._linfuncop import LinearFunctional, LinearFunctionOperator


class ScaledLinearFunctionOperator(LinearFunctionOperator):
    def __init__(self, linfuncop: LinearFunctionOperator, scalar: ScalarLike) -> None:
        self._linfuncop = linfuncop

        super().__init__(
            input_shapes=self._linfuncop.input_shapes,
            output_shapes=self._linfuncop.output_shapes,
        )

        if not np.ndim(scalar) == 0:
            raise ValueError()

        self._scalar = np.asarray(scalar, dtype=np.double)

    @property
    def linfuncop(self) -> LinearFunctionOperator:
        return self._linfuncop

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._scalar * self._linfuncop(f, **kwargs)

    # TODO: Only need until GPs can be scaled
    @__call__.register
    def _(
        self, gp: pn.randprocs.GaussianProcess, /, **kwargs
    ) -> pn.randprocs.GaussianProcess:
        return super().__call__(gp, **kwargs)

    def __rmul__(self, other) -> LinearFunctionOperator:
        if np.ndim(other) == 0:
            return ScaledLinearFunctionOperator(
                linfuncop=self._linfuncop,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)


class SumLinearFunctionOperator(LinearFunctionOperator):
    def __init__(self, *summands: LinearFunctionOperator) -> None:
        self._summands = tuple(summands)

        input_domain_shape = self._summands[0].input_domain_shape
        input_codomain_shape = self._summands[0].input_codomain_shape
        output_domain_shape = self._summands[0].output_domain_shape
        output_codomain_shape = self._summands[0].output_codomain_shape

        assert all(
            summand.input_domain_shape == input_domain_shape
            for summand in self._summands
        )
        assert all(
            summand.input_codomain_shape == input_codomain_shape
            for summand in self._summands
        )
        assert all(
            summand.output_domain_shape == output_domain_shape
            for summand in self._summands
        )
        assert all(
            summand.output_codomain_shape == output_codomain_shape
            for summand in self._summands
        )

        super().__init__(
            input_shapes=(input_domain_shape, input_codomain_shape),
            output_shapes=(output_domain_shape, output_codomain_shape),
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return functools.reduce(
            operator.add, (summand(f, **kwargs) for summand in self._summands)
        )

    @__call__.register
    def _(self, randproc: pn.randprocs.RandomProcess, /) -> pn.randprocs.RandomProcess:
        return super().__call__(randproc)


class CompositeLinearFunctionOperator(LinearFunctionOperator):
    def __init__(self, *linfuncops: LinearFunctionOperator) -> None:
        assert all(
            L0.input_shape == L1.output_shape
            for L0, L1 in zip(linfuncops[:-1], linfuncops[1:])
        )

        self._linfuncops = tuple(linfuncops)

        super().__init__(
            input_shapes=self._linfuncops[-1].input_shapes,
            output_shapes=self._linfuncops[0].output_shapes,
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return functools.reduce(
            lambda h, linfuncop: linfuncop(h, **kwargs),
            reversed(self._linfuncops),
            f,
        )


class ScaledLinearFunctional(LinearFunctional):
    def __init__(self, linfunctl: LinearFunctional, scalar: ScalarLike) -> None:
        self._linfunctl = linfunctl

        super().__init__(
            input_shapes=self._linfunctl.input_shapes,
            output_shape=self._linfunctl.output_shape,
        )

        if not np.ndim(scalar) == 0:
            raise ValueError()

        self._scalar = np.asarray(scalar, dtype=np.double)

    @property
    def linfunctl(self) -> LinearFunctional:
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

    def __rmul__(self, other) -> LinearFunctional:
        if np.ndim(other) == 0:
            return ScaledLinearFunctional(
                linfunctl=self._linfunctl,
                scalar=np.asarray(other) * self._scalar,
            )

        return super().__rmul__(other)


class SumLinearFunctional(LinearFunctional):
    def __init__(self, *summands: LinearFunctional) -> None:
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


class CompositeLinearFunctional(LinearFunctional):
    def __init__(
        self,
        linfunctl: LinearFunctional,
        *linfuncops: LinearFunctionOperator,
    ) -> None:
        assert linfunctl.input_shapes == linfuncops[0].output_shapes
        assert all(
            L0.input_shape == L1.output_shape
            for L0, L1 in zip(linfuncops[:-1], linfuncops[1:])
        )

        self._linfunctl = linfunctl
        self._linfuncops = tuple(linfuncops)

        super().__init__(
            input_shapes=self._linfuncops[-1].input_shapes,
            output_shape=self._linfunctl.output_shape,
        )

    @functools.singledispatchmethod
    def __call__(self, f, /, **kwargs):
        return self._linfunctl(
            functools.reduce(
                lambda h, linfuncop: linfuncop(h, **kwargs),
                reversed(self._linfuncops),
                f,
            ),
            **kwargs,
        )
