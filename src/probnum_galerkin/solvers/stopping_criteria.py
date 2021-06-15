import numpy as np
import probnum as pn
from probnum.type import FloatArgType


class MaxIterations:
    def __init__(self, maxiter: int) -> None:
        self._maxiter = maxiter

    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: "probnum_galerkin.solvers.beliefs.BayesCGBelief",
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State" = None,
    ) -> bool:
        return solver_state.iteration >= self._maxiter


class ResidualNorm:
    def __init__(self, atol: FloatArgType = 1e-5, rtol: FloatArgType = 1e-5) -> None:
        self.atol = pn.utils.as_numpy_scalar(atol)
        self.rtol = pn.utils.as_numpy_scalar(rtol)

    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: "probnum_galerkin.solvers.beliefs.BayesCGBelief",
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State" = None,
    ) -> bool:
        # Compare residual to tolerances
        b_norm = np.linalg.norm(problem.b, ord=2)

        return (
            solver_state.residual_norm <= self.atol
            or solver_state.residual_norm <= self.rtol * b_norm
        )
