from __future__ import annotations

from jax import numpy as jnp
import numpy as np

from ._pv_crosscov import ProcessVectorCrossCovariance


class ReshapedProcessVectorCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(self, pv_crosscov: ProcessVectorCrossCovariance, order: str = "C"):
        self._pv_crosscov = pv_crosscov
        self._order = order

        super().__init__(
            randproc_input_shape=pv_crosscov.randproc_input_shape,
            randproc_output_shape=pv_crosscov.randproc_output_shape,
            randvar_shape=(pv_crosscov.randvar_size,),
            reverse=pv_crosscov.reverse,
        )

    @property
    def pv_crosscov(self):
        return self._pv_crosscov

    @property
    def order(self):
        return self._order

    def _get_output_shape(self, x: np.ndarray | jnp.ndarray) -> tuple[int]:
        x_batch_shape = x.shape[: x.ndim - self.pv_crosscov.randproc_input_ndim]
        x_output_shape = self.randproc_output_shape

        if self.reverse:
            return self.randvar_shape + x_batch_shape + x_output_shape
        return x_batch_shape + x_output_shape + self.randvar_shape

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        res = self.pv_crosscov(x)
        output_shape = self._get_output_shape(x)
        return res.reshape(output_shape, order=self.order)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        res = self.pv_crosscov.jax(x)
        output_shape = self._get_output_shape(x)
        return res.reshape(output_shape, order=self.order)
