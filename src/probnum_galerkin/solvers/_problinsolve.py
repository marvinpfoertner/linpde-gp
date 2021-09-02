from typing import Callable, Optional

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


def problinsolve(
    A,
    b,
    x0=None,
    maxiter=None,
    atol=1e-5,
    rtol=1e-5,
    reorthogonalize: bool = False,
    noise_var=0.0,
    rng=None,
    callback: Optional[Callable[..., None]] = None,
) -> pn.randvars.Normal:
    # Construct the problem to be solved
    problem = pn.problems.LinearSystem(A, b)

    # Construct the prior
    prior_dtype = np.result_type(problem.A.dtype, problem.b.dtype)

    if isinstance(x0, pn.randvars.Normal):
        prior = beliefs.GaussianSolutionBelief(
            mean=x0.mean.astype(prior_dtype, copy=True),
            cov=x0.cov.astype(prior_dtype, copy=True),
        )
    elif x0 is None or isinstance(x0, (np.ndarray, pn.randvars.Constant)):
        prior = beliefs.GaussianSolutionBelief.from_linear_system(problem, mean=x0)
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
    solver = _probabilistic_linear_solver.ProbabilisticLinearSolver(
        prior,
        policy=policies.KrylovPolicy(
            reorthogonalization_fn=reorthogonalization_fn,
        ),
        # policy=policies.RandomPolicy(
        #     rng, reorthogonalization_fn=reorthogonalization_fn
        # ),
        observation_op=observation_ops.ResidualMatVec(),
        belief_update=belief_updates.GaussianInferenceBeliefUpdate(noise_var=noise_var),
        stopping_criteria=stopping_criteria_,
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
