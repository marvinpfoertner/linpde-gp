from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ScalarLike

from linpde_gp.linfuncops import LinearFunctionOperator
from linpde_gp.linfunctls import LinearFunctional

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

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._scalar * self._pv_crosscov(x)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._scalar * self._pv_crosscov.jax(x)


@LinearFunctionOperator.__call__.register
def _(
    self, crosscov: ScaledProcessVectorCrossCovariance, /, **kwargs
) -> ScaledProcessVectorCrossCovariance:
    return ScaledProcessVectorCrossCovariance(
        pv_crosscov=self(crosscov._pv_crosscov, **kwargs),
        scalar=crosscov._scalar,
    )


@LinearFunctional.__call__.register
def _(self, crosscov: ScaledProcessVectorCrossCovariance, /, **kwargs) -> np.ndarray:
    return crosscov._scalar * self(crosscov._pv_crosscov, **kwargs)


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
