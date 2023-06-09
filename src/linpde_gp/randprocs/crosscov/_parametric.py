from jax import numpy as jnp
import numpy as np
import probnum as pn

from linpde_gp.randvars import Covariance

from ._pv_crosscov import ProcessVectorCrossCovariance


class ParametricProcessVectorCrossCovariance(ProcessVectorCrossCovariance):
    def __init__(
        self,
        crosscov: Covariance,
        basis: pn.functions.Function,
        reverse: bool = False,
    ):
        self._crosscov = crosscov
        self._basis = basis

        super().__init__(
            randproc_input_shape=basis.input_shape,
            randproc_output_shape=(),
            randvar_shape=crosscov.shape0 if reverse else crosscov.shape1,
            reverse=reverse,
        )

    @property
    def crosscov(self) -> Covariance:
        return self._crosscov

    @property
    def basis(self) -> pn.functions.Function:
        return self._basis

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        pv_crosscov = self._crosscov.linop(self._basis(x), axis=-1)

        if self.reverse:
            return np.moveaxis(pv_crosscov, -1, 0)

        return pv_crosscov

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()
