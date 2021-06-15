from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import probnum as pn

from ._base import ProbabilisticLinearSolver as _ProbabilisticLinearSolver


class CGPolicy(_ProbabilisticLinearSolver.Policy):
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
        solver_state: "_ProbabilisticLinearSolver.State",
    ) -> np.ndarray:
        action = solver_state.residual

        if solver_state.iteration > 0:
            # Orthogonalization
            beta = solver_state.residual_norm_sq / solver_state.prev_residual_norm_sq

            action += beta * solver_state.action

            # (Optional) Reorthogonalization
            if self._reorthogonalization_fn is not None:
                action = self._reorthogonalization_fn(
                    action,
                    solver_state.prev_actions,
                    inprod_matrix=(
                        problem.A @ solver_state.prior.cov_unscaled @ problem.A.T
                    ),
                )

        return action
