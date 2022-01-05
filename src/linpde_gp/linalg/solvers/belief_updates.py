import abc

import numpy as np
import probnum as pn
from probnum.typing import FloatLike

from ... import linops
from . import beliefs


class LinearSolverBeliefUpdate(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: beliefs.LinearSystemBelief,
        action: np.ndarray,
        observation: np.floating,
        solver_state: "linpde_gp.solvers.ProbabilisticLinearSolver.State",
    ) -> beliefs.LinearSystemBelief:
        pass


class GaussianInferenceBeliefUpdate(LinearSolverBeliefUpdate):
    def __init__(self, noise_var: FloatLike = 0.0) -> None:
        self._noise_var = noise_var

    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: beliefs.GaussianSolutionBelief,
        action: np.ndarray,
        observation: np.floating,
        solver_state: "linpde_gp.solvers.ProbabilisticLinearSolver.State",
    ) -> beliefs.GaussianSolutionBelief:
        adj_obs_operator = problem.A @ action

        cov_xy = belief.cov @ adj_obs_operator

        gram = adj_obs_operator.T @ cov_xy
        gram_pinv = 1.0 / (gram + self._noise_var)

        return beliefs.GaussianSolutionBelief(
            mean=belief.mean + cov_xy * (gram_pinv * observation),
            cov=belief.cov - linops.outer(cov_xy, cov_xy) * gram_pinv,
        )


class BayesCGBeliefUpdate(LinearSolverBeliefUpdate):
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: beliefs.BayesCGBelief,
        action: np.ndarray,
        observation: np.floating,
        solver_state: "linpde_gp.solvers.ProbabilisticLinearSolver.State",
    ) -> beliefs.BayesCGBelief:
        matvec = problem.A @ action
        stepdir = solver_state.prior.cov_unscaled @ matvec

        E_sq = np.inner(matvec, stepdir)
        alpha = observation / E_sq

        return beliefs.BayesCGBelief(
            mean=belief.mean + alpha * stepdir,
            cov_unscaled=(
                belief.cov_unscaled - linops.outer(stepdir, stepdir) * (1 / E_sq)
            ),
            cov_scale=belief.cov_scale + alpha,
            num_steps=belief.num_steps + 1,
        )
