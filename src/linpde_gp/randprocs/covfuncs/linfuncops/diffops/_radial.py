from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.randprocs import covfuncs as pn_covfuncs

from linpde_gp.linfuncops import diffops

from ..._jax import JaxCovarianceFunctionMixin
from .._abstract import CovarianceFunction_LinearFunctionOperator


class UnivariateRadialCovarianceFunction_Derivative_Derivative(
    JaxCovarianceFunctionMixin, CovarianceFunction_LinearFunctionOperator
):
    def __init__(
        self,
        covfunc: pn_covfuncs.CovarianceFunction,
        L0: diffops.Derivative,
        L1: diffops.Derivative,
        *,
        radial_derivative: pn.functions.Function,
    ) -> None:
        assert radial_derivative.input_shape == ()
        assert radial_derivative.output_shape == ()

        self._radial_derivative = radial_derivative

        self._output_scale_factor = (1 / covfunc.lengthscales) ** (L0.order + L1.order)
        self._pos_factor = self._output_scale_factor * (
            1.0 if L1.order % 2 == 0 else -1.0
        )
        self._neg_factor = self._output_scale_factor * (
            1.0 if L0.order % 2 == 0 else -1.0
        )

        super().__init__(covfunc, L0, L1)

    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        diffs = x0 - x1

        return self._radial_derivative(
            np.abs(diffs) / self._covfunc.lengthscales
        ) * np.where(diffs >= 0, self._pos_factor, self._neg_factor)

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        diffs = x0 - x1

        return self._radial_derivative.jax(
            jnp.abs(diffs) / self._covfunc.lengthscales
        ) * jnp.where(diffs >= 0, self._pos_factor, self._neg_factor)
