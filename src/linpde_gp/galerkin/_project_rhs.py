import numpy as np
from plum import Dispatcher

from linpde_gp import functions, problems, randprocs

from . import bases

dispatch = Dispatcher()


@dispatch
def project(f: randprocs.DeterministicProcess, basis) -> np.ndarray:
    return project(f.as_fn(), basis)


@dispatch
def project(
    f: functions.Constant, basis: bases.ZeroBoundaryFiniteElementBasis
) -> np.ndarray:
    return (f.value / 2.0) * (basis.grid[2:] - basis.grid[:-2])


@dispatch
def project(f: functions.Constant, basis: bases.FiniteElementBasis) -> np.ndarray:
    assert len(basis._boundary_conditions) == 1

    boundary_condition = basis._boundary_conditions[0]

    assert isinstance(boundary_condition, problems.pde.DirichletBoundaryCondition)
    assert isinstance(boundary_condition.values, np.ndarray)

    u_l, u_r = basis._boundary_conditions[0].values

    rhs = np.empty_like(basis.grid)

    rhs[1:-1] = (f.value / 2) * (basis.grid[2:] - basis.grid[:-2])

    # Left Boundary Condition
    rhs[0] = u_l
    rhs[1] += u_l / (basis.grid[2] - basis.grid[1])

    # Right Boundary Condition
    rhs[-1] = u_r
    rhs[-2] += u_r / (basis.grid[-2] - basis.grid[-3])

    return rhs


@dispatch
def project(f: functions.Constant, basis: bases.FourierBasis) -> np.ndarray:
    l, r = basis._domain

    idcs = np.arange(1, len(basis) + 1)

    return (f.value * (r - l) / np.pi) * (1 - np.cos(np.pi * idcs)) / idcs
