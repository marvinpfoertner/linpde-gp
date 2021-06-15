from typing import Optional

import numpy as np
import probnum as pn

from . import beliefs
from ._base import ProbabilisticLinearSolver as _ProbabilisticLinearSolver


class BayesCGBeliefUpdate:
    def __call__(
        self,
        problem: pn.problems.LinearSystem,
        belief: beliefs.BayesCGBelief,
        action: np.ndarray,
        observation: Optional[np.floating],
        solver_state: "_ProbabilisticLinearSolver.State",
    ) -> beliefs.BayesCGBelief:
        matvec = problem.A @ action
        stepdir = solver_state.prior.cov_unscaled @ matvec

        E_sq = np.inner(matvec, stepdir)
        alpha = solver_state.residual_norm_sq / E_sq

        return beliefs.BayesCGBelief(
            mean=belief.mean + alpha * stepdir,
            cov_unscaled=belief.cov_unscaled - np.outer(stepdir, stepdir) / E_sq,
            cov_scale=belief.cov_scale + alpha,
            num_steps=belief.num_steps + 1,
        )
