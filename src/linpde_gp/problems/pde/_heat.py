import functools

from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import FloatLike

from linpde_gp import domains, functions
from linpde_gp.linfuncops import diffops
from linpde_gp.typing import DomainLike

from ._bvp import DirichletBoundaryCondition, InitialBoundaryValueProblem
from ._linear_pde import LinearPDE


class HeatEquation(LinearPDE):
    def __init__(
        self,
        domain: DomainLike,
        rhs: pn.functions.Function | None = None,
        alpha: FloatLike = 1.0,
    ):
        self._alpha = float(alpha)

        super().__init__(
            domain=domain,
            diffop=diffops.HeatOperator(domain_shape=domain.shape, alpha=self._alpha),
            rhs=rhs,
        )


class HeatEquationDirichletProblem(InitialBoundaryValueProblem):
    def __init__(
        self,
        t0: FloatLike,
        spatial_domain: DomainLike,
        T: FloatLike = float("inf"),
        rhs: pn.functions.Function | None = None,
        alpha: FloatLike = 1.0,
        initial_values: pn.functions.Function | None = None,
        solution: pn.functions.Function | None = None,
    ):
        domain = domains.CartesianProduct(
            domains.Interval(t0, T),
            spatial_domain,
        )

        pde = HeatEquation(domain, rhs=rhs, alpha=alpha)

        # Initial condition
        if initial_values is None:
            initial_values = functions.Zero(
                input_shape=spatial_domain.shape, output_shape=()
            )

        if (
            initial_values.input_shape != spatial_domain.shape
            and initial_values.output_shape != ()
        ):
            raise ValueError()

        initial_condition = DirichletBoundaryCondition(domain[1], initial_values)

        # Spatial boundary conditions
        boundary_conditions = tuple(
            DirichletBoundaryCondition(
                domains.CartesianProduct(domain[0], boundary_part),
                np.zeros(()),
            )
            for boundary_part in domain[1].boundary
        )

        if solution is None:
            if isinstance(initial_values, functions.Zero):
                solution = functions.Zero(domain.shape, output_shape=())
            elif (
                isinstance(initial_values, functions.TruncatedSineSeries)
                and initial_values.domain == domain[1]
            ):
                solution = Solution_HeatEquation_DirichletProblem_1D_InitialTruncatedSineSeries_BoundaryZero(
                    t0=t0,
                    spatial_domain=spatial_domain,
                    initial_values=initial_values,
                    alpha=alpha,
                )

        super().__init__(
            pde=pde,
            initial_condition=initial_condition,
            boundary_conditions=boundary_conditions,
            solution=solution,
        )


class Solution_HeatEquation_DirichletProblem_1D_InitialTruncatedSineSeries_BoundaryZero(
    pn.functions.Function
):
    def __init__(
        self,
        t0: FloatLike,
        spatial_domain: domains.Interval,
        initial_values: functions.TruncatedSineSeries,
        alpha: FloatLike,
    ):
        self._t0 = float(t0)
        self._spatial_domain = spatial_domain
        self._initial_values = initial_values
        self._alpha = float(alpha)

        assert isinstance(self._spatial_domain, domains.Interval)
        assert self._spatial_domain == self._initial_values.domain

        super().__init__(input_shape=(2,), output_shape=())

    @functools.cached_property
    def _decay_rates(self) -> np.ndarray:
        return self._alpha * self._initial_values.half_angular_frequencies**2

    def _evaluate(self, txs: np.ndarray) -> np.ndarray:
        l, _ = self._spatial_domain

        ts, xs = np.split(txs, 2, axis=-1)

        return np.sum(
            self._initial_values.coefficients
            * np.sin(self._initial_values.half_angular_frequencies * (xs - l))
            * np.exp(self._decay_rates * (self._t0 - ts)),
            axis=-1,
        )

    def _jax_evaluate(self, txs: jnp.ndarray) -> jnp.ndarray:
        l, _ = self._spatial_domain

        ts, xs = jnp.split(txs, 2, axis=-1)

        return jnp.sum(
            self._initial_values.coefficients
            * jnp.sin(self._initial_values.half_angular_frequencies * (xs - l))
            * jnp.exp(self._decay_rates * (self._t0 - ts)),
            axis=-1,
        )
