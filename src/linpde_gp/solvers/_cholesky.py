from typing import Optional

from jax import numpy as jnp
import numpy as np

from linpde_gp.linops import BlockMatrix2x2
from linpde_gp.randprocs.covfuncs import JaxCovarianceFunction

from ._gp_solver import ConcreteGPSolver, GPInferenceParams, GPSolver
from .covfuncs import DowndateCovarianceFunction


class CholeskyCovarianceFunction(DowndateCovarianceFunction):
    def __init__(self, gp_params: GPInferenceParams):
        self._gp_params = gp_params
        super().__init__(gp_params.prior.cov)

    def _downdate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        kLas_x0 = self._gp_params.kLas(x0)
        kLas_x1 = self._gp_params.kLas(x1) if x1 is not None else kLas_x0
        return (
            kLas_x0[..., None, :]
            @ (self._gp_params.prior_gram.solve(kLas_x1[..., None]))
        )[..., 0, 0]

    def _downdate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        kLas_x0 = self._gp_params.kLas.jax(x0)
        kLas_x1 = self._gp_params.kLas.jax(x1) if x1 is not None else kLas_x0
        return (
            kLas_x0[..., None, :]
            @ (self._gp_params.prior_gram.solve(kLas_x1[..., None]))
        )[..., 0, 0]


class ConcreteCholeskySolver(ConcreteGPSolver):
    """
    Concrete solver that uses the Cholesky decomposition.

    Uses a block Cholesky decomposition if possible.
    """

    def __init__(
        self,
        gp_params: GPInferenceParams,
        load_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        super().__init__(gp_params, load_path, save_path)

    def _compute_representer_weights(self):
        if self._gp_params.prior_representer_weights is not None:
            # Update existing representer weights
            assert isinstance(self._gp_params.prior_gram, BlockMatrix2x2)
            new_residual = self._get_residual(
                self._gp_params.Ys[-1], self._gp_params.Ls[-1], self._gp_params.bs[-1]
            )
            return self._gp_params.prior_gram.schur_update(
                self._gp_params.prior_representer_weights, new_residual
            )
        full_residual = self._get_full_residual()
        return self._gp_params.prior_gram.solve(full_residual)

    @property
    def posterior_cov(self) -> JaxCovarianceFunction:
        return CholeskyCovarianceFunction(self._gp_params)

    def _save_state(self) -> dict:
        # TODO: Actually save the Cholesky decomposition of the linear operator
        state = {"representer_weights": self._representer_weights}
        return state

    def _load_state(self, dict):
        self._representer_weights = dict["representer_weights"]


class CholeskySolver(GPSolver):
    """Solver that uses the Cholesky decomposition."""

    def __init__(
        self, load_path: Optional[str] = None, save_path: Optional[str] = None
    ):
        super().__init__(load_path, save_path)

    def get_concrete_solver(
        self, gp_params: GPInferenceParams
    ) -> ConcreteCholeskySolver:
        return ConcreteCholeskySolver(gp_params, self._load_path, self._save_path)
