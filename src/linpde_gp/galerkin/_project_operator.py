import numpy as np
import probnum as pn
import scipy.sparse

from linpde_gp.linfuncops.diffops import ScaledLaplaceOperator

from . import bases


@ScaledLaplaceOperator.project.register
def _(self, basis: bases.ZeroBoundaryFiniteElementBasis) -> pn.linops.Matrix:
    assert self._alpha == -1.0

    diag = 1 / (basis.grid[1:-1] - basis.grid[:-2])
    diag += 1 / (basis.grid[2:] - basis.grid[1:-1])

    offdiag = -1.0 / (basis.grid[2:-1] - basis.grid[1:-2])

    return pn.linops.Matrix(
        scipy.sparse.diags(
            (offdiag, diag, offdiag),
            offsets=(-1, 0, 1),
            format="csr",
        )
    )


@ScaledLaplaceOperator.project.register
def _(self, basis: bases.FiniteElementBasis) -> pn.linops.Matrix:
    assert self._alpha == -1.0

    diag = np.empty_like(basis.grid)
    offdiag = np.empty_like(diag, shape=(len(basis) - 1,))

    # Left boundary condition
    diag[0] = 1.0
    offdiag[0] = 0.0

    # Negative Laplace operator on the interior
    diag[1:-1] = 1 / (basis.grid[1:-1] - basis.grid[:-2])
    diag[1:-1] += 1 / (basis.grid[2:] - basis.grid[1:-1])

    offdiag[1:-1] = -1.0 / (basis.grid[2:-1] - basis.grid[1:-2])

    # Right boundary condition
    diag[-1] = 1.0
    offdiag[-1] = 0.0

    return pn.linops.Matrix(
        scipy.sparse.diags(
            (offdiag, diag, offdiag),
            offsets=(-1, 0, 1),
            format="csr",
        )
    )


@ScaledLaplaceOperator.project.register
def _(self, basis: bases.FourierBasis) -> pn.linops.Matrix:
    assert self._alpha == -1.0

    l, r = basis._domain

    idcs = np.arange(1, len(basis) + 1)

    return pn.linops.Matrix(
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
