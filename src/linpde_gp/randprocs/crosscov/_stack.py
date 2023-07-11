from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike

from linpde_gp.linops import BlockMatrix

from ._pv_crosscov import ProcessVectorCrossCovariance


class StackedProcessVectorCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(self, pv_crosscovs: ArrayLike):
        self._pv_crosscovs = np.asarray(pv_crosscovs)
        pv_crosscovs_flat = self._pv_crosscovs.reshape(-1, order="C")

        if any(
            pv_crosscov.randproc_input_shape
            != pv_crosscovs_flat[0].randproc_input_shape
            for pv_crosscov in pv_crosscovs_flat
        ):
            raise ValueError()

        if any(
            pv_crosscov.randproc_output_shape != () for pv_crosscov in pv_crosscovs_flat
        ):
            raise ValueError()

        if any(
            pv_crosscov.randvar_shape != pv_crosscovs_flat[0].randvar_shape
            for pv_crosscov in pv_crosscovs_flat
        ):
            raise ValueError()

        if any(
            pv_crosscov.reverse != pv_crosscovs_flat[0].reverse
            for pv_crosscov in pv_crosscovs_flat
        ):
            raise ValueError()

        super().__init__(
            randproc_input_shape=pv_crosscovs_flat[0].randproc_input_shape,
            randproc_output_shape=self._pv_crosscovs.shape,
            randvar_shape=pv_crosscovs_flat[0].randvar_shape,
            reverse=pv_crosscovs_flat[0].reverse,
        )

    @property
    def pv_crosscovs(self) -> np.ndarray:
        return self._pv_crosscovs

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        evals = np.empty_like(self._pv_crosscovs, dtype=np.object_)
        batch_shape = None
        for idx, pv_crosscov in np.ndenumerate(self._pv_crosscovs):
            evals[idx] = pv_crosscov(x)
            if batch_shape is None and self.reverse:
                batch_shape = evals[idx].shape[self.randvar_ndim :]
            elif batch_shape is None:
                batch_shape = evals[idx].shape[: evals[idx].ndim - self.randvar_ndim]

        if self.reverse:
            res = np.zeros(self.randvar_shape + batch_shape + self._pv_crosscovs.shape)
            for idx, eval_at_idx in np.ndenumerate(evals):
                res[(..., *idx)] = eval_at_idx
        else:
            res = np.zeros(self._pv_crosscovs.shape + batch_shape + self.randvar_shape)
            for idx, eval_at_idx in np.ndenumerate(evals):
                res[(*idx, ...)] = eval_at_idx
            # Move entire batch shape to the front
            res = np.moveaxis(
                res,
                range(self._pv_crosscovs.ndim, res.ndim - self.randvar_ndim),
                range(len(batch_shape)),
            )

        return res

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        evals = np.empty_like(self._pv_crosscovs, dtype=np.object_)
        batch_shape = None
        for idx, pv_crosscov in np.ndenumerate(self._pv_crosscovs):
            evals[idx] = pv_crosscov.jax(x)
            if batch_shape is None and self.reverse:
                batch_shape = evals[idx].shape[self.randvar_ndim :]
            elif batch_shape is None:
                batch_shape = evals[idx].shape[: evals[idx].ndim - self.randvar_ndim]

        if self.reverse:
            res = jnp.zeros(self.randvar_shape + batch_shape + self._pv_crosscovs.shape)
            for idx, eval_at_idx in np.ndenumerate(evals):
                res.at[(..., *idx)].set(eval_at_idx)
        else:
            res = jnp.zeros(self._pv_crosscovs.shape + batch_shape + self.randvar_shape)
            for idx, eval_at_idx in np.ndenumerate(evals):
                res.at[(*idx, ...)].set(eval_at_idx)
            # Move entire batch shape to the front
            res = jnp.moveaxis(
                res,
                range(self._pv_crosscovs.ndim, res.ndim - self.randvar_ndim),
                range(len(batch_shape)),
            )

        return res

    def _evaluate_linop(self, x: np.ndarray) -> BlockMatrix:
        if self.reverse:
            return BlockMatrix(
                [
                    [
                        pv_crosscov.evaluate_linop(x)
                        for pv_crosscov in self.pv_crosscovs.reshape(-1, order="C")
                    ]
                ]
            )
        return BlockMatrix(
            [
                [pv_crosscov.evaluate_linop(x)]
                for pv_crosscov in self.pv_crosscovs.reshape(-1, order="C")
            ]
        )

    def __repr__(self) -> str:
        res = "StackedPVCrossCov[\n\t"
        res += ",\n\t".join(repr(pv_crosscov) for pv_crosscov in self.pv_crosscovs)
        res += "\n]"
        return res
