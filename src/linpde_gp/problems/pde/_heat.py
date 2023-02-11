import probnum as pn

from linpde_gp import domains, functions
from linpde_gp.linfuncops import diffops
from linpde_gp.typing import DomainLike

from ._bvp import BoundaryValueProblem, DirichletBoundaryCondition
from ._linear_pde import LinearPDE


class HeatEquation(LinearPDE):
    def __init__(
        self,
        domain: DomainLike,
        rhs: pn.functions.Function | None = None,
        alpha: float = 1.0,
    ):
        domain = domains.asdomain(domain)

        super().__init__(
            domain=domain,
            diffop=diffops.HeatOperator(domain_shape=domain.shape, alpha=alpha),
            rhs=rhs,
        )


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
