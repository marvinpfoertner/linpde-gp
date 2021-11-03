import abc

import numpy as np
import probnum as pn


class ObservationOp(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        action: np.ndarray,
        solver_state: "linpde_gp.solvers.ProbabilisticLinearSolver.State",
    ) -> np.ndarray:
        pass


class ResidualMatVec(ObservationOp):
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        action: np.ndarray,
        solver_state: "linpde_gp.solvers.ProbabilisticLinearSolver.State",
    ) -> np.ndarray:
        return np.inner(action, solver_state.residual)


class ResidualNormSquared(ObservationOp):
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        action: np.ndarray,
        solver_state: "linpde_gp.solvers.ProbabilisticLinearSolver.State",
    ) -> np.ndarray:
        return solver_state.residual_norm_squared
