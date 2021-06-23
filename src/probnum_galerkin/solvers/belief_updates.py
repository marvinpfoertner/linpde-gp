import abc

import numpy as np
import probnum as pn

from . import beliefs


class LinearSolverBeliefUpdate(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: beliefs.LinearSystemBelief,
        action: np.ndarray,
        observation: np.floating,
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> beliefs.LinearSystemBelief:
        pass


class BayesCGBeliefUpdate(LinearSolverBeliefUpdate):
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: beliefs.BayesCGBelief,
        action: np.ndarray,
        observation: np.floating,
        solver_state: "probnum_galerkin.solvers.ProbabilisticLinearSolver.State",
    ) -> beliefs.BayesCGBelief:
        matvec = problem.A @ action
        stepdir = solver_state.prior.cov_unscaled @ matvec

        E_sq = np.inner(matvec, stepdir)
        alpha = observation / E_sq

        return beliefs.BayesCGBelief(
            mean=belief.mean + alpha * stepdir,
            cov_unscaled=belief.cov_unscaled - np.outer(stepdir, stepdir) / E_sq,
            cov_scale=belief.cov_scale + alpha,
            num_steps=belief.num_steps + 1,
        )
