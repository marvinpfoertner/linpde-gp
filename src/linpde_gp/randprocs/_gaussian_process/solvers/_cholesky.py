from ._gp_solver import GPSolver, ConcreteGPSolver, GPInferenceParams
import probnum as pn
import numpy as np
from linpde_gp.linops import BlockMatrix2x2


class ConcreteCholeskySolver(ConcreteGPSolver):
    def __init__(self, gp_params: GPInferenceParams):
        super().__init__(gp_params)

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

    def compute_posterior_cov(
        self, k_xx: np.ndarray, k_x0_X: np.ndarray, k_x1_X: np.ndarray
    ):
        return (
            k_xx
            - (
                k_x0_X[..., None, :]
                @ (self._gp_params.prior_gram.solve(k_x1_X[..., None]))
            )[..., 0, 0]
        )


class CholeskySolver(GPSolver):
    def __init__(self):
        super().__init__()

    def get_concrete_solver(
        self, gp_params: GPInferenceParams
    ) -> ConcreteCholeskySolver:
        return ConcreteCholeskySolver(gp_params)
