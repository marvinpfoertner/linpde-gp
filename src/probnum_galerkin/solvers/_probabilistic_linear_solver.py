from typing import Generator, Iterable, Iterator, Optional, Tuple

import numpy as np
import probnum as pn

from . import belief_updates, beliefs, observation_ops, policies, stopping_criteria


class ProbabilisticLinearSolver(
    pn.ProbabilisticNumericalMethod[
        pn.problems.LinearSystem, beliefs.LinearSystemBelief
    ]
):
    def __init__(
        self,
        prior: beliefs.LinearSystemBelief,
        policy: policies.Policy,
        observation_op: observation_ops.ObservationOp,
        belief_update: belief_updates.LinearSolverBeliefUpdate,
        stopping_criteria: Optional[Iterable[stopping_criteria.StoppingCriterion]],
    ) -> None:
        super().__init__(prior)

        self._policy = policy
        self._observation_op = observation_op
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
    ) -> Iterator[Tuple[pn.randvars.Normal, "ProbabilisticLinearSolver.State", bool]]:
        solver_state = ProbabilisticLinearSolver.State(problem, self.prior)

        while True:
            stop = any(
                stopping_criterion(problem, solver_state.belief, solver_state)
                for stopping_criterion in self._stopping_criteria
            )

            yield solver_state.belief, solver_state, stop

            if stop:
                break

            solver_state.action = self._policy(
                problem, solver_state.belief, solver_state
            )

            solver_state.observation = self._observation_op(
                problem, solver_state.action, solver_state
            )

            solver_state.belief = self._belief_update(
                problem,
                solver_state.belief,
                solver_state.action,
                solver_state.observation,
                solver_state,
            )

            solver_state.iteration += 1

    class State:
        def __init__(
            self,
            problem: pn.problems.LinearSystem,
            prior: beliefs.LinearSystemBelief,
        ) -> None:
            self.iteration: int = 0

            self.problem = problem

            # Belief
            self.prior = prior

            self._belief = prior

            # Actions
            self._action = None

            self._prev_actions = []

            # Observations
            self._observation = None

            self._prev_observations = []

            # Caches
            self._residual = None
            self._residual_norm_squared = None
            self._residual_norm = None

            self._prev_residuals = []
            self._prev_residual_norms_squared = []

        @property
        def belief(self) -> beliefs.LinearSystemBelief:
            return self._belief

        @belief.setter
        def belief(self, value: beliefs.LinearSystemBelief) -> None:
            self._belief = value

            # Invalidate caches
            self._prev_residuals.append(self.residual)
            self._prev_residual_norms_squared.append(self._residual_norm_squared)

            self._residual = None
            self._residual_norm_squared = None
            self._residual_norm = None

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

        @property
        def observation(self) -> np.ndarray:
            return self._observation

        @observation.setter
        def observation(self, value: np.ndarray) -> None:
            if self._observation is not None:
                self._prev_observations.append(self._observation)

            self._observation = value

        @property
        def residual(self) -> np.ndarray:
            if self._residual is None:
                self._residual = self.problem.b - self.problem.A @ self._belief.x.mean

            return self._residual

        @property
        def residual_norm_squared(self) -> np.floating:
            if self._residual_norm_squared is None:
                self._residual_norm_squared = np.inner(self.residual, self.residual)

            return self._residual_norm_squared

        @property
        def residual_norm(self) -> np.floating:
            if self._residual_norm is None:
                self._residual_norm = np.sqrt(self.residual_norm_squared)

            return self._residual_norm

        @property
        def prev_residual(self) -> Optional[np.ndarray]:
            if len(self._prev_residuals) == 0:
                return None

            return self._prev_residuals[-1]

        @property
        def prev_residual_norm_squared(self) -> Optional[np.floating]:
            if len(self._prev_residual_norms_squared) == 0:
                return None

            if self._prev_residual_norms_squared[-1] is None:
                prev_residual = self._prev_residuals[-1]

                self._prev_residual_norms_squared[-1] = np.inner(
                    prev_residual, prev_residual
                )

            return self._prev_residual_norms_squared[-1]
