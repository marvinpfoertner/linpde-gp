from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import LinearOperatorLike

from ._pv_crosscov import ProcessVectorCrossCovariance


class ParametricProcessVectorCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(
        self,
        crosscov: LinearOperatorLike,
        basis: pn.functions.Function,
        reverse: bool = False,
    ):
        self._crosscov = pn.linops.aslinop(crosscov)
        self._basis = basis

        super().__init__(
            randproc_input_shape=basis.input_shape,
            randproc_output_shape=(),
            randvar_shape=crosscov.shape[:1] if reverse else crosscov.shape[1:],
            reverse=reverse,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pv_crosscov = self._crosscov(self._basis(x), axis=-1)

        if self.reverse:
            return np.moveaxis(pv_crosscov, -1, 0)

        return pv_crosscov

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return super()._evaluate_jax(x)
