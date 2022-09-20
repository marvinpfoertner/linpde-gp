from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ScalarLike, ScalarType

from . import _pv_crosscov


class ScaledProcessVectorCrossCovariance(_pv_crosscov.ProcessVectorCrossCovariance):
    def __init__(
        self,
        pv_crosscov: _pv_crosscov.ProcessVectorCrossCovariance,
        scalar: ScalarLike,
    ):
        self._pv_crosscov = pv_crosscov
        self._scalar = pn.utils.as_numpy_scalar(scalar)

        super().__init__(
            randproc_input_shape=pv_crosscov.randproc_input_shape,
            randproc_output_shape=pv_crosscov.randproc_output_shape,
            randvar_shape=pv_crosscov.randvar_shape,
            reverse=pv_crosscov.reverse,
        )

    @property
    def pv_crosscov(self) -> _pv_crosscov.ProcessVectorCrossCovariance:
        return self._pv_crosscov

    @property
    def scalar(self) -> ScalarType:
        return self._scalar

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._scalar * self._pv_crosscov(x)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._scalar * self._pv_crosscov.jax(x)


class SumProcessVectorCrossCovariance(_pv_crosscov.ProcessVectorCrossCovariance):
    def __init__(self, *pv_crosscovs: _pv_crosscov.ProcessVectorCrossCovariance):
        self._pv_crosscovs = tuple(pv_crosscovs)

        assert all(
            pv_crosscov.randproc_input_shape == pv_crosscovs[0].randproc_input_shape
            and (
                pv_crosscov.randproc_output_shape
                == pv_crosscovs[0].randproc_output_shape
            )
            and pv_crosscov.randvar_shape == pv_crosscovs[0].randvar_shape
            and pv_crosscov.reverse == pv_crosscovs[0].reverse
            for pv_crosscov in self._pv_crosscovs
        )

        super().__init__(
            randproc_input_shape=self._pv_crosscovs[0].randproc_input_shape,
            randproc_output_shape=self._pv_crosscovs[0].randproc_output_shape,
            randvar_shape=self._pv_crosscovs[0].randvar_shape,
            reverse=self._pv_crosscovs[0].reverse,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return sum(pv_crosscov(x) for pv_crosscov in self._pv_crosscovs)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return sum(pv_crosscov.jax(x) for pv_crosscov in self._pv_crosscovs)


class LinOpProcessVectorCrossCovariance(_pv_crosscov.ProcessVectorCrossCovariance):
    def __init__(
        self,
        linop: pn.linops.LinearOperator,
        pv_crosscov: _pv_crosscov.ProcessVectorCrossCovariance,
    ):
        assert pv_crosscov.randvar_ndim == 1
        assert linop.shape[1:] == pv_crosscov.randvar_shape

        self._linop = linop
        self._pv_crosscov = pv_crosscov

        super().__init__(
            randproc_input_shape=self._pv_crosscov.randproc_input_shape,
            randproc_output_shape=self._pv_crosscov.randproc_output_shape,
            randvar_shape=linop.shape[0:1],
            reverse=self._pv_crosscov.reverse,
        )

    @property
    def linop(self) -> pn.linops.LinearOperator:
        return self._linop

    @property
    def pv_crosscov(self) -> _pv_crosscov.ProcessVectorCrossCovariance:
        return self._pv_crosscov

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._linop(
            self._pv_crosscov(x),
            axis=0 if self.reverse else -1,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()
