import numpy as np
import probnum as pn
from probnum.typing import ShapeLike, ShapeType

from linpde_gp import functions, linfuncops


class ProcessVectorCrossCovariance(functions.JaxFunction):
    def __init__(
        self,
        randproc_input_shape: ShapeLike,
        randproc_output_shape: ShapeLike,
        randvar_shape: ShapeLike,
        transpose: bool = True,
    ):
        self._randproc_input_shape = pn.utils.as_shape(randproc_input_shape)
        self._randproc_output_shape = pn.utils.as_shape(randproc_output_shape)
        self._randvar_shape = pn.utils.as_shape(randvar_shape)

        self._transposed = bool(transpose)

        super().__init__(
            input_shape=randproc_input_shape,
            output_shape=(
                self._randvar_shape + self._randproc_output_shape
                if self._transposed
                else self._randproc_output_shape + self._randvar_shape
            ),
        )

    @property
    def randproc_input_shape(self) -> ShapeType:
        return self._randproc_input_shape

    @property
    def randproc_output_shape(self) -> ShapeType:
        return self._randproc_output_shape

    @property
    def randvar_shape(self) -> ShapeType:
        return self._randvar_shape

    @property
    def transposed(self) -> bool:
        return self._transposed


@linfuncops.LinearFunctional.__call__.register  # pylint: disable=no-member
@linfuncops.LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, cross_cov: ProcessVectorCrossCovariance, /) -> np.ndarray:
    raise NotImplementedError()
