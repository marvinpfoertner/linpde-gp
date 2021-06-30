from typing import Callable, Iterable, Optional

import numpy as np
import probnum as pn

from .. import linalg
from . import (
    _probabilistic_linear_solver,
    belief_updates,
    beliefs,
    observation_ops,
    policies,
    stopping_criteria,
)


class BayesCG(_probabilistic_linear_solver.ProbabilisticLinearSolver):
    def __init__(
        self,
        prior: beliefs.BayesCGBelief,
        stopping_criteria: Iterable[stopping_criteria.StoppingCriterion],
        reorthogonalization_fn: Optional[Callable[..., None]] = None,
    ) -> None:
        super().__init__(
            prior,
            policy=policies.CGPolicy(reorthogonalization_fn=reorthogonalization_fn),
            observation_op=observation_ops.ResidualNormSquared(),
            belief_update=belief_updates.BayesCGBeliefUpdate(),
            stopping_criteria=tuple(stopping_criteria),
        )


def bayescg(
    A,
    b,
    x0=None,
    maxiter=None,
    atol=1e-5,
    rtol=1e-5,
    reorthogonalize: bool = False,
    callback: Optional[Callable[..., None]] = None,
) -> pn.randvars.Normal:
    # Construct the problem to be solved
    problem = pn.problems.LinearSystem(A, b)

    # Construct the prior
    prior_dtype = np.result_type(problem.A.dtype, problem.b.dtype)

    if isinstance(x0, pn.randvars.Normal):
        prior = beliefs.BayesCGBelief(
            mean=x0.mean.astype(prior_dtype, copy=True),
            cov_unscaled=x0.cov.astype(prior_dtype, copy=True),
        )
    elif x0 is None or isinstance(x0, (np.ndarray, pn.randvars.Constant)):
        prior = beliefs.BayesCGBelief.from_linear_system(problem, mean=x0)
    else:
        raise TypeError()

    # Construct the stopping criteria
    if maxiter is None:
        maxiter = 10 * b.size

    stopping_criteria_ = (
        stopping_criteria.MaxIterations(maxiter),
        stopping_criteria.ResidualNorm(atol, rtol),
    )

    # Configure reorthogonalization
    reorthogonalization_fn = None

    if reorthogonalize:
        reorthogonalization_fn = linalg.modified_gram_schmidt

    # Sentinel Callback
    if callback is None:
        callback = lambda **kwargs: None

    # Construct the solver
    solver = BayesCG(
        prior, stopping_criteria_, reorthogonalization_fn=reorthogonalization_fn
    )

    # Run the algorithm
    for belief, solver_state, stop in solver.solve_iter(problem):
        callback(
            iteration=solver_state.iteration,
            x=belief.x,
            residual=solver_state.residual.copy(),
            stop=stop,
            action=(
                solver_state.prev_action.copy() if solver_state.iteration > 0 else None
            ),
        )

    return belief.x
