import functools
from typing import Callable

import numpy as np
import probnum as pn
import scipy.sparse
from probnum.type import FloatArgType
from probnum_galerkin import bases

from . import _base


class PoissonEquation(_base.LinearPDE):
    @functools.cached_property
    def solution(self) -> Callable[[FloatArgType], np.floating]:
        (l, r) = self._domain
        u_l = self._boundary_condition(l)
        u_r = self._boundary_condition(r)

        aff_slope = (u_r - u_l) / (r - l)

        def u(x: FloatArgType) -> np.floating:
            return u_l + (aff_slope - (self._rhs / 2.0) * (x - r)) * (x - l)

        return u

    def discretize(self, basis: bases.Basis) -> pn.problems.LinearSystem:
        if isinstance(basis, bases.ZeroBoundaryFiniteElementBasis):
            A = self._operator_zero_boundary_finite_element_basis(basis)
            b = self._rhs_zero_boundary_finite_element_basis(basis)
        elif isinstance(basis, bases.FiniteElementBasis):
            A = self._operator_finite_element_basis(basis)
            b = self._rhs_finite_element_basis(basis)
        else:
            raise NotImplementedError(
                f"Discretization with basis of type {basis.__class__.__name__} is not "
                f"implemented."
            )

        return pn.problems.LinearSystem(
            pn.linops.aslinop(A),
            b,
        )

    def _operator_zero_boundary_finite_element_basis(
        self, basis: bases.FiniteElementBasis
    ) -> pn.linops.Matrix:
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

    def _rhs_zero_boundary_finite_element_basis(
        self, basis: bases.FiniteElementBasis
    ) -> np.ndarray:
        return (self._rhs / 2.0) * (basis.grid[2:] - basis.grid[:-2])

    def _operator_finite_element_basis(
        self, basis: bases.FiniteElementBasis
    ) -> pn.linops.Matrix:
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

    def _rhs_finite_element_basis(self, basis: bases.FiniteElementBasis) -> np.ndarray:
        (l, r) = self._domain
        u_l = self._boundary_condition(l)
        u_r = self._boundary_condition(r)

        rhs = np.empty_like(basis.grid)

        rhs[1:-1] = (self._rhs / 2) * (basis.grid[2:] - basis.grid[:-2])

        # Left Boundary Condition
        rhs[0] = u_l
        rhs[1] += u_l / (basis.grid[2] - basis.grid[1])

        # Right Boundary Condition
        rhs[-1] = u_r
        rhs[-2] += u_r / (basis.grid[-2] - basis.grid[-3])

        return rhs

    def _operator_fourier_basis(self, basis: bases.FourierBasis) -> pn.linops.Matrix:
        (l, r) = self._domain

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

    def _rhs_fourier_basis(self, basis: bases.FourierBasis) -> np.ndarray:
        (l, r) = self._domain

        idcs = np.arange(1, len(basis) + 1)

        return (self._rhs * (r - l) / np.pi) * (1 - np.cos(np.pi * idcs)) / idcs
