import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, ShapeLike

from linpde_gp import linfuncops, linfunctls


class DeterministicProcess(pn.randprocs.RandomProcess[ArrayLike, np.ndarray]):
    def __init__(self, fn: pn.functions.Function):
        self._fn = fn

        super().__init__(
            input_shape=self._fn.input_shape,
            output_shape=self._fn.output_shape,
            dtype=np.double,
        )

    def __call__(self, args: ArrayLike) -> pn.randvars.Constant:
        return pn.randvars.Constant(support=self._fn(np.asarray(args)))

    @property
    def mean(self) -> pn.functions.Function:
        return self._fn

    def as_fn(self) -> pn.functions.Function:
        return self._fn

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: np.ndarray,
        size: ShapeLike = (),
    ) -> np.ndarray:
        return self(args).sample(rng, size=size)


@linfunctls.LinearFunctional.__call__.register(  # pylint: disable=no-member
    DeterministicProcess
)
def _(self, randproc: DeterministicProcess, /) -> pn.randvars.Constant:
    return pn.randvars.Constant(self(randproc.as_fn()))


@linfuncops.LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    DeterministicProcess
)
def _(self, randproc: DeterministicProcess, /) -> DeterministicProcess:
    return DeterministicProcess(self(randproc.as_fn()))
