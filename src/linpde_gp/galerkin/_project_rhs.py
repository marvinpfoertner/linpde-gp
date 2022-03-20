import numpy as np
from plum import Dispatcher

from linpde_gp import functions

from . import bases

dispatch = Dispatcher()


@dispatch
def project(
    f: functions.Constant, basis: bases.ZeroBoundaryFiniteElementBasis
) -> np.ndarray:
    return (f.value / 2.0) * (basis.grid[2:] - basis.grid[:-2])


@dispatch
def project(f: functions.Constant, basis: bases.FiniteElementBasis) -> np.ndarray:
    if not all(
        isinstance(boundary_condition.values, (np.ndarray, np.floating))
        for boundary_condition in basis._boundary_conditions
    ):
        raise NotImplementedError()

    u_l = basis._boundary_conditions[0].values
    u_r = basis._boundary_conditions[1].values

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
