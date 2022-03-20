import numbers
from typing import Sequence

import numpy as np
import probnum as pn
from probnum.typing import FloatLike

from linpde_gp.linfuncops import diffops

from . import domains
from ._bvp import BoundaryValueProblem, DirichletBoundaryCondition


def poisson_1d_bvp(
    domain: domains.DomainLike,
    rhs,
    boundary_values: Sequence[float] = (0.0, 0.0),
    solution=None,
):
    domain = domains.asdomain(domain)

    assert isinstance(domain, domains.Interval)

    if solution is None and isinstance(rhs, numbers.Real):
        solution = poisson_1d_const_solution(*domain, rhs, *boundary_values)

    return BoundaryValueProblem(
        domain=domain,
        diffop=diffops.Laplacian(domain_shape=domain.shape, alpha=-1.0),
        rhs=rhs,
        boundary_conditions=(
            DirichletBoundaryCondition(
                domain.boundary[0], pn.randvars.asrandvar(boundary_values[0])
            ),
            DirichletBoundaryCondition(
                domain.boundary[1], pn.randvars.asrandvar(boundary_values[1])
            ),
        ),
        solution=solution,
    )


def poisson_bvp(
    domain: domains.DomainLike,
    rhs: pn.randprocs.RandomProcess,
    boundary_values: pn.randprocs.RandomProcess,
):
    domain = domains.asdomain(domain)

    return BoundaryValueProblem(
        domain=domain,
        diffop=diffops.Laplacian(domain_shape=domain.shape, alpha=-1.0),
        rhs=rhs,
        boundary_conditions=tuple(
            DirichletBoundaryCondition(boundary_part, boundary_values)
            for boundary_part in domain.boundary
        ),
    )


def poisson_1d_const_solution(l, r, rhs, u_l, u_r):
    aff_slope = (u_r - u_l) / (r - l)

    def u(x: FloatLike) -> np.floating:
        return u_l + (aff_slope - (rhs / 2.0) * (x - r)) * (x - l)

    return u


def heat_1d_bvp(
    domain: domains.DomainLike,
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
