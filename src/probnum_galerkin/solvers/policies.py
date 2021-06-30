import abc
from typing import Callable, Iterable, Optional

import numpy as np
import probnum as pn


class Policy(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: pn.randvars.Normal,
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> np.ndarray:
        pass


class CGPolicy(Policy):
    def __init__(
        self,
        reorthogonalization_fn: Optional[
            Callable[
                [np.ndarray, Iterable[np.ndarray], pn.linops.LinearOperator], np.ndarray
            ]
        ] = None,
    ) -> None:
        self._reorthogonalization_fn = reorthogonalization_fn

    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: pn.randvars.Normal,
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> np.ndarray:
        action = solver_state.residual.copy()

        if solver_state.iteration > 0:
            # Orthogonalization
            beta = (
                solver_state.residual_norm_squared
                / solver_state.prev_residual_norm_squared
            )

            action += beta * solver_state.prev_action

            # (Optional) Reorthogonalization
            if self._reorthogonalization_fn is not None:
                if isinstance(solver_state.prior.x, pn.randvars.Normal):
                    inprod_matrix = problem.A @ solver_state.prior.x.cov @ problem.A.T
                elif isinstance(solver_state.prior.x, pn.randvars.Constant):
                    inprod_matrix = problem.A

                action = self._reorthogonalization_fn(
                    action,
                    solver_state.prev_actions,
                    inprod_matrix,
                )

        return action


class CovariancePolicy(Policy):
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: pn.randvars.Normal,
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> np.ndarray:
        if solver_state.iteration == 0:
            return solver_state.residual

        A_linop = pn.linops.aslinop(problem.A)
        prior_cov = pn.linops.aslinop(solver_state.prior.cov)

        return (
            A_linop.inv()
            @ (prior_cov.inv() @ belief.cov)
            @ (A_linop @ solver_state.residual)
        )
