from typing import Union

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike

from linpde_gp import domains, functions
from linpde_gp.linfuncops import diffops
from linpde_gp.typing import DomainLike

from ._bvp import BoundaryValueProblem, DirichletBoundaryCondition


class PoissonEquationDirichletProblem(BoundaryValueProblem):
    def __init__(
        self,
        domain: DomainLike,
        rhs: pn.Function,
        boundary_values: Union[
            ArrayLike,
            pn.randvars.RandomVariable,
            pn.Function,
            pn.randprocs.RandomProcess,
        ],
        solution=None,
    ):
        domain = domains.asdomain(domain)

        if isinstance(domain, domains.Interval):
            if not isinstance(boundary_values, pn.randvars.RandomVariable):
                boundary_values = np.asarray(boundary_values)

            boundary_conditions = (
                DirichletBoundaryCondition(domain.boundary[0], boundary_values[0]),
                DirichletBoundaryCondition(domain.boundary[1], boundary_values[1]),
            )

            if solution is None:
                if isinstance(rhs, functions.Constant) and isinstance(
                    boundary_values, np.ndarray
                ):
                    solution = PoissonEquation1DConstRHSDirichletProblemSolution(
                        domain,
                        rhs=rhs.value,
                        boundary_values=boundary_values,
                    )
        else:
            boundary_conditions = tuple(
                DirichletBoundaryCondition(boundary_part, boundary_values)
                for boundary_part in domain.boundary
            )

        super().__init__(
            domain=domain,
            diffop=-diffops.Laplacian(domain.shape),
            rhs=rhs,
            boundary_conditions=boundary_conditions,
            solution=solution,
        )


class PoissonEquation1DConstRHSDirichletProblemSolution(pn.Function):
    def __init__(
        self,
        domain: domains.Interval,
        rhs: np.ndarray,
        boundary_values: np.ndarray,
    ):
        super().__init__((), output_shape=())

        self._l, self._r = domain
        self._rhs = rhs
        self._u_l, self._u_r = boundary_values

        self._aff_slope = (self._u_r - self._u_l) / (self._r - self._l)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._u_l + (self._aff_slope - (self._rhs / 2.0) * (x - self._r)) * (
            x - self._l
        )


def heat_1d_bvp(
    domain: DomainLike,
    initial_values,
):
    domain = domains.asdomain(domain)

    assert isinstance(domain, domains.Box) and domain.shape[0] == 2

    return BoundaryValueProblem(
        domain=domain,
        diffop=diffops.HeatOperator(domain_shape=domain.shape),
        rhs=lambda tx: 0,
        boundary_conditions=(
            DirichletBoundaryCondition(
                domain.boundary[0],
                values=initial_values,
            ),
        ),
    )
