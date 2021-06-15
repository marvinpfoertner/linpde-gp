import abc
from typing import Optional, Tuple

import numpy as np
import probnum as pn


class ProbabilisticLinearSolver(
    pn.ProbabilisticNumericalMethod[pn.problems.LinearSystem, pn.randvars.Normal]
):
    def __init__(
        self,
        prior: pn.randvars.Normal,
        policy,
        belief_update,
        stopping_criteria,
    ) -> None:
        super().__init__(prior)

        self._policy = policy
        self._belief_update = belief_update
        self._stopping_criteria = stopping_criteria

    def solve(
        self, problem: pn.problems.LinearSystem
    ) -> Tuple[pn.randvars.Normal, "ProbabilisticLinearSolver.State"]:
        for belief, solver_state, _ in self.solve_iter(problem):
            pass

        return belief, solver_state

    def solve_iter(
        self, problem: pn.problems.LinearSystem
    ) -> Tuple[pn.randvars.Normal, "ProbabilisticLinearSolver.State"]:
        solver_state = ProbabilisticLinearSolver.State(problem, self.prior)

        belief = self.prior

        while True:
            stop = any(
                stopping_criterion(problem, belief, solver_state)
                for stopping_criterion in self._stopping_criteria
            )

            yield belief, solver_state, stop

            if stop:
                break

            action = self._policy(problem, belief, solver_state)

            solver_state.action = action

            # TODO: observation
            observation = None

            belief = self._belief_update(
                problem, belief, action, observation, solver_state
            )

            solver_state.iteration += 1
            solver_state.update_residual(problem, belief)

    class State:
        def __init__(
            self,
            problem: pn.problems.LinearSystem,
            prior,
        ) -> None:
            self.iteration: int = 0

            self.prior = prior

            # Residual
            self._residual = problem.b - problem.A @ self.prior.x.mean
            self._residual_norm_sq = None
            self._residual_norm = None

            self._prev_residuals = []
            self._prev_residual_norms_sq = []

            # Actions
            self._action = None

            self._prev_actions = []

        @property
        def residual(self) -> np.ndarray:
            return self._residual

        def update_residual(self, problem: pn.problems.LinearSystem, belief) -> None:
            self._prev_residuals.append(self._residual)
            self._prev_residual_norms_sq.append(self._residual_norm_sq)

            self._residual = problem.b - problem.A @ belief.x.mean
            self._residual_norm_sq = None
            self._residual_norm = None

        @property
        def residual_norm_sq(self) -> np.floating:
            if self._residual_norm_sq is None:
                self._residual_norm_sq = np.inner(self._residual, self._residual)

            return self._residual_norm_sq

        @property
        def residual_norm(self) -> np.floating:
            if self._residual_norm is None:
                self._residual_norm = np.sqrt(self.residual_norm_sq)

            return self._residual_norm

        @property
        def prev_residual(self) -> Optional[np.ndarray]:
            if len(self._prev_residuals) == 0:
                return None

            return self._prev_residuals[-1]

        @property
        def prev_residual_norm_sq(self) -> Optional[np.floating]:
            if len(self._prev_residual_norms_sq) == 0:
                return None

            if self._prev_residual_norms_sq[-1] is None:
                prev_residual = self._prev_residuals[-1]

                self._prev_residual_norms_sq[-1] = np.inner(
                    prev_residual, prev_residual
                )

            return self._prev_residual_norms_sq[-1]

        @property
        def action(self) -> np.ndarray:
            return self._action

        @action.setter
        def action(self, value: np.ndarray) -> None:
            if self._action is not None:
                self._prev_actions.append(self._action)

            self._action = value

        @property
        def prev_action(self) -> np.ndarray:
            return self._prev_actions[-1]

        @property
        def prev_actions(self) -> Tuple[np.ndarray, ...]:
            return tuple(self._prev_actions)

    class Policy(abc.ABC):
        @abc.abstractmethod
        def __call__(
            self,
            problem: pn.problems.LinearSystem,
            belief: pn.randvars.Normal,
            solver_state: "ProbabilisticLinearSolver.State",
        ) -> np.ndarray:
            pass
