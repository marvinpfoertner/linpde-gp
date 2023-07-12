from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.randprocs import covfuncs as pn_covfuncs
from probnum.typing import ArrayLike

from linpde_gp.linops import BlockMatrix

from ._jax import JaxCovarianceFunctionMixin


class StackCovarianceFunction(
    JaxCovarianceFunctionMixin, pn_covfuncs.CovarianceFunction
):
    def __init__(self, covfuncs: ArrayLike, output_idx: int = 1) -> None:
        self._covfuncs = np.asarray(covfuncs)
        covfuncs_flat = self._covfuncs.reshape(-1, order="C")

        if any(
            covfunc.input_shape_0 != covfuncs_flat[0].input_shape_0
            for covfunc in covfuncs_flat
        ):
            raise ValueError()

        if any(
            covfunc.input_shape_1 != covfuncs_flat[0].input_shape_1
            for covfunc in covfuncs_flat
        ):
            raise ValueError()

        if any(
            covfunc.output_shape_0 != () or covfunc.output_shape_1 != ()
            for covfunc in covfuncs_flat
        ):
            raise ValueError()

        output_idx = int(output_idx)

        if output_idx not in (0, 1):
            raise ValueError()

        self._output_idx = output_idx

        super().__init__(
            input_shape_0=covfuncs_flat[0].input_shape_0,
            input_shape_1=covfuncs_flat[0].input_shape_1,
            output_shape_0=self._covfuncs.shape if self._output_idx == 0 else (),
            output_shape_1=self._covfuncs.shape if self._output_idx == 1 else (),
        )

    @property
    def covfuncs(self) -> np.ndarray:
        return self._covfuncs

    @property
    def output_idx(self) -> int:
        return self._output_idx

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        evals = np.empty_like(self._covfuncs, dtype=np.object_)
        batch_shape = None
        for idx, covfunc in np.ndenumerate(self._covfuncs):
            evals[idx] = covfunc(x0, x1)
            if batch_shape is None:
                batch_shape = evals[idx].shape

        res = np.zeros(batch_shape + self._covfuncs.shape)
        for idx, eval_at_idx in np.ndenumerate(evals):
            res[(..., *idx)] = eval_at_idx
        return res

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        evals = np.empty_like(self._covfuncs, dtype=np.object_)
        batch_shape = None
        for idx, covfunc in np.ndenumerate(self._covfuncs):
            evals[idx] = covfunc.jax(x0, x1)
            if batch_shape is None:
                batch_shape = evals[idx].shape

        res = jnp.zeros(batch_shape + self._covfuncs.shape)
        for idx, eval_at_idx in np.ndenumerate(evals):
            res.at[(..., *idx)].set(eval_at_idx)
        return res

    def linop(
        self, x0: pn.utils.ArrayLike, x1: Optional[pn.utils.ArrayLike] = None
    ) -> pn.linops.LinearOperator:
        if self._output_idx == 0:
            return BlockMatrix(
                [
                    [covfunc.linop(x0, x1)]
                    for covfunc in self.covfuncs.reshape(-1, order="C")
                ]
            )
        return BlockMatrix(
            [
                [
                    covfunc.linop(x0, x1)
                    for covfunc in self.covfuncs.reshape(-1, order="C")
                ]
            ]
        )
