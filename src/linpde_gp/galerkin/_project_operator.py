import numpy as np
from plum import Dispatcher
import probnum as pn
import scipy.sparse

from linpde_gp import linfuncops

from . import bases

dispatch = Dispatcher()


@dispatch
def project(
    linfuncop: linfuncops.ScaledLinearFunctionOperator, basis: bases.Basis
) -> pn.linops.LinearOperator:
    return linfuncop.scalar * project(linfuncop.linfuncop, basis)


@dispatch
def project(
    linfuncop: linfuncops.diffops.ScaledLinearDifferentialOperator, basis: bases.Basis
) -> pn.linops.LinearOperator:
    return linfuncop.scalar * project(linfuncop.lindiffop, basis)


@dispatch
def project(
    linfuncop: linfuncops.diffops.Laplacian,
    basis: bases.ZeroBoundaryFiniteElementBasis,
) -> pn.linops.Matrix:
    diag = -1.0 / (basis.grid[1:-1] - basis.grid[:-2])
    diag -= 1.0 / (basis.grid[2:] - basis.grid[1:-1])

    offdiag = 1.0 / (basis.grid[2:-1] - basis.grid[1:-2])

    return pn.linops.Matrix(
        scipy.sparse.diags(
            (offdiag, diag, offdiag),
            offsets=(-1, 0, 1),
            format="csr",
        )
    )


@dispatch
def project(
    linfuncop: linfuncops.diffops.Laplacian, basis: bases.FiniteElementBasis
) -> pn.linops.Matrix:
    diag = np.empty_like(basis.grid)
    offdiag = np.empty_like(diag, shape=(len(basis) - 1,))

    # Left boundary condition
    diag[0] = -1.0
    offdiag[0] = 0.0

    # Laplacian on the interior
    diag[1:-1] = -1 / (basis.grid[1:-1] - basis.grid[:-2])
    diag[1:-1] -= 1 / (basis.grid[2:] - basis.grid[1:-1])

    offdiag[1:-1] = 1.0 / (basis.grid[2:-1] - basis.grid[1:-2])

    # Right boundary condition
    diag[-1] = -1.0
    offdiag[-1] = 0.0

    return pn.linops.Matrix(
        scipy.sparse.diags(
            (offdiag, diag, offdiag),
            offsets=(-1, 0, 1),
            format="csr",
        )
    )


@dispatch
def project(
    linfuncop: linfuncops.diffops.Laplacian, basis: bases.FourierBasis
) -> pn.linops.Matrix:
    l, r = basis._domain

    idcs = np.arange(1, len(basis) + 1)

    return -pn.linops.Matrix(
        scipy.sparse.diags(
            (
                (idcs * (np.pi / (4 * (r - l))))
                * ((2 * np.pi) * idcs + np.sin((2 * np.pi) * idcs))
            ),
            offsets=0,
            format="csr",
            dtype=np.double,
        )
    )
