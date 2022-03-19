import functools
import operator

import probnum as pn

from ._linfuncop import LinearFunctionOperator


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
