from collections.abc import Sequence
from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp.linops import BlockMatrix

from ._jax import JaxCovarianceFunctionMixin


class StackCovarianceFunction(
    JaxCovarianceFunctionMixin, pn_covfuncs.CovarianceFunction
):
    def __init__(
        self, *covfuncs: pn_covfuncs.CovarianceFunction, output_idx: int = 1
    ) -> None:
        if any(
            covfunc.input_shape_0 != covfuncs[0].input_shape_0 for covfunc in covfuncs
        ):
            raise ValueError()

        if any(
            covfunc.input_shape_1 != covfuncs[0].input_shape_1 for covfunc in covfuncs
        ):
            raise ValueError()

        if any(
            covfunc.output_shape_0 != () or covfunc.output_shape_1 != ()
            for covfunc in covfuncs
        ):
            raise ValueError()

        self._covfuncs = tuple(covfuncs)

        output_idx = int(output_idx)

        if output_idx not in (0, 1):
            raise ValueError()

        self._output_idx = output_idx

        super().__init__(
            input_shape_0=covfuncs[0].input_shape_0,
            input_shape_1=covfuncs[0].input_shape_1,
            output_shape_0=(len(self._covfuncs),) if self._output_idx == 0 else (),
            output_shape_1=(len(self._covfuncs),) if self._output_idx == 1 else (),
        )

    @property
    def covfuncs(self) -> Sequence[pn_covfuncs.CovarianceFunction]:
        return self._covfuncs

    @property
    def output_idx(self) -> int:
        return self._output_idx

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        return np.stack([covfunc(x0, x1) for covfunc in self._covfuncs], axis=-1)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        return jnp.stack([covfunc.jax(x0, x1) for covfunc in self._covfuncs], axis=-1)

    def linop(
        self, x0: pn.utils.ArrayLike, x1: Optional[pn.utils.ArrayLike] = None
    ) -> pn.linops.LinearOperator:
        if self._output_idx == 0:
            return BlockMatrix([[covfunc.linop(x0, x1)] for covfunc in self._covfuncs])
        else:
            return BlockMatrix([[covfunc.linop(x0, x1) for covfunc in self._covfuncs]])
