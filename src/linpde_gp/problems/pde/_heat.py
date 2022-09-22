import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, FloatLike

from linpde_gp import domains, functions, randprocs
from linpde_gp.linfuncops import diffops
from linpde_gp.typing import DomainLike, RandomProcessLike, RandomVariableLike

from ._bvp import BoundaryValueProblem, DirichletBoundaryCondition
from ._linear_pde import LinearPDE


class PoissonEquation(LinearPDE):
    def __init__(
        self,
        domain: DomainLike,
        rhs: RandomProcessLike,
        alpha: float = 1.0,
    ):
        domain = domains.asdomain(domain)

        super().__init__(
            domain=domain,
            diffop=-alpha * diffops.Laplacian(domain_shape=domain.shape),
            rhs=rhs,
        )

        self._alpha = alpha

    @property
    def alpha(self) -> float:
        return self._alpha


class HeatEquation(LinearPDE):
    def __init__(
        self,
        domain: DomainLike,
        rhs: RandomProcessLike,
        alpha: float = 1.0,
    ):
        domain = domains.asdomain(domain)

        super().__init__(
            domain=domain,
            diffop=diffops.HeatOperator(domain_shape=domain.shape, alpha=alpha),
            rhs=rhs,
        )


class PoissonEquationDirichletProblem(BoundaryValueProblem):
    def __init__(
        self,
        pde: PoissonEquation,
        boundary_values: RandomProcessLike | RandomVariableLike,
        solution: RandomProcessLike = None,
    ):
        if not isinstance(pde, PoissonEquation):
            raise TypeError("The given PDE must be a Poisson equation.")

        if pde.domain.shape == ():
            if not isinstance(pde.domain, domains.Interval):
                raise TypeError("In the scalar case, we only support Interval domains.")

            assert isinstance(pde.domain.boundary, domains.PointSet)

            if isinstance(
                boundary_values, (pn.randprocs.RandomProcess, pn.functions.Function)
            ):
                raise TypeError(
                    "In the scalar case, the boundary conditions must be "
                    "`RandomVariableLike`."
                )

            if not isinstance(boundary_values, pn.randvars.RandomVariable):
                boundary_values = np.asarray(boundary_values)

            boundary_values = pn.randvars.asrandvar(boundary_values)

            boundary_conditions = (
                DirichletBoundaryCondition(pde.domain.boundary, boundary_values),
            )

            if solution is None:
                rhs_is_deterministic_constant = isinstance(
                    pde.rhs, randprocs.DeterministicProcess
                ) and isinstance(pde.rhs.as_fn(), functions.Constant)

                bc_is_deterministic = isinstance(boundary_values, pn.randvars.Constant)

                if rhs_is_deterministic_constant and bc_is_deterministic:
                    solution = PoissonEquation1DConstRHSDirichletProblemSolution(
                        pde.domain,
                        rhs=pde.rhs.as_fn().value,
                        boundary_values=boundary_values.support,
                        alpha=pde.alpha,
                    )
        else:
            boundary_conditions = tuple(
                DirichletBoundaryCondition(boundary_part, boundary_values)
                for boundary_part in pde.domain.boundary
            )

        if solution is not None:
            solution = randprocs.asrandproc(solution)

        super().__init__(
            pde=pde,
            boundary_conditions=boundary_conditions,
            solution=solution,
        )


class PoissonEquation1DConstRHSDirichletProblemSolution(pn.functions.Function):
    def __init__(
        self,
        domain: DomainLike,
        rhs: FloatLike,
        boundary_values: ArrayLike,
        alpha: FloatLike = 1.0,
    ):
        super().__init__(input_shape=(), output_shape=())

        domain = domains.asdomain(domain)

        if not isinstance(domain, domains.Interval):
            raise TypeError("We only support Interval domains.")

        self._l, self._r = domain
        self._rhs = float(rhs)
        self._u_l, self._u_r = np.asarray(boundary_values)
        self._alpha = float(alpha)

        self._coeffs = [
            self._u_l,
            (self._u_r - self._u_l) / (self._r - self._l),
            0.5 * self._rhs / -self._alpha,
        ]

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        l, r = self._l, self._r
        a = self._coeffs

        return (a[2] * (x - r) + a[1]) * (x - l) + a[0]


def heat_1d_bvp(
    domain: DomainLike,
    initial_values,
    alpha: float = 1.0,
):
    domain = domains.asdomain(domain)

    assert isinstance(domain, domains.Box) and domain.shape[0] == 2

    return BoundaryValueProblem(
        pde=HeatEquation(
            domain=domain,
            rhs=functions.Zero(input_shape=domain.shape, output_shape=()),
            alpha=alpha,
        ),
        boundary_conditions=(
            DirichletBoundaryCondition(
                domain.boundary[0],
                values=initial_values,
            ),
        ),
    )
