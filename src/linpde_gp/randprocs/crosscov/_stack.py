from collections.abc import Sequence

from jax import numpy as jnp
import numpy as np

from ._pv_crosscov import ProcessVectorCrossCovariance


class StackedProcessVectorCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(self, *pv_crosscovs: ProcessVectorCrossCovariance):
        if any(
            pv_crosscov.randproc_input_shape != pv_crosscovs[0].randproc_input_shape
            for pv_crosscov in pv_crosscovs
        ):
            raise ValueError()

        if any(pv_crosscov.randproc_output_shape != () for pv_crosscov in pv_crosscovs):
            raise ValueError()

        if any(
            pv_crosscov.randvar_shape != pv_crosscovs[0].randvar_shape
            for pv_crosscov in pv_crosscovs
        ):
            raise ValueError()

        if any(
            pv_crosscov.reverse != pv_crosscovs[0].reverse
            for pv_crosscov in pv_crosscovs
        ):
            raise ValueError()

        self._pv_crosscovs = tuple(pv_crosscovs)

        super().__init__(
            randproc_input_shape=self._pv_crosscovs[0].randproc_input_shape,
            randproc_output_shape=(len(self._pv_crosscovs),),
            randvar_shape=self._pv_crosscovs[0].randvar_shape,
            reverse=self._pv_crosscovs[0].reverse,
        )

    @property
    def pv_crosscovs(self) -> Sequence[ProcessVectorCrossCovariance]:
        return self._pv_crosscovs

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.stack(
            [pv_crosscov(x) for pv_crosscov in self._pv_crosscovs],
            axis=-1 if self.reverse else -self.randvar_ndim - 1,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [pv_crosscov.jax(x) for pv_crosscov in self._pv_crosscovs],
            axis=-1 if self.reverse else -self.randvar_ndim - 1,
        )
