import numbers

import numpy as np
import plum
import probnum as pn

from . import bases


@plum.dispatch
def project(f: numbers.Real, basis: bases.ZeroBoundaryFiniteElementBasis) -> np.ndarray:
    return (f / 2.0) * (basis.grid[2:] - basis.grid[:-2])


@plum.dispatch
def project(f: numbers.Real, basis: bases.FiniteElementBasis) -> np.ndarray:
    l, r = basis._domain

    if not all(
        isinstance(boundary_condition.values, pn.randvars.Constant)
        for boundary_condition in basis._boundary_conditions
    ):
        raise NotImplementedError()

    u_l = basis._boundary_conditions[0].values.mean
    u_r = basis._boundary_conditions[1].values.mean

    rhs = np.empty_like(basis.grid)

    rhs[1:-1] = (f / 2) * (basis.grid[2:] - basis.grid[:-2])

    # Left Boundary Condition
    rhs[0] = u_l
    rhs[1] += u_l / (basis.grid[2] - basis.grid[1])

    # Right Boundary Condition
    rhs[-1] = u_r
    rhs[-2] += u_r / (basis.grid[-2] - basis.grid[-3])

    return rhs


@plum.dispatch
def project(f: numbers.Real, basis: bases.FourierBasis) -> np.ndarray:
    l, r = basis._domain

    idcs = np.arange(1, len(basis) + 1)

    return (f * (r - l) / np.pi) * (1 - np.cos(np.pi * idcs)) / idcs
