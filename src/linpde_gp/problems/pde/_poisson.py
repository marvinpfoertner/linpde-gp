from jax import numpy as jnp
import numpy as np
import probnum as pn
from probnum.typing import ArrayLike, FloatLike

from linpde_gp import domains, functions
from linpde_gp.linfuncops import diffops
from linpde_gp.typing import DomainLike

from ._bvp import BoundaryValueProblem, DirichletBoundaryCondition
from ._linear_pde import LinearPDE


class PoissonEquation(LinearPDE):
    def __init__(
        self,
        domain: DomainLike,
        rhs: pn.functions.Function | None = None,
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


class PoissonEquationDirichletProblem(BoundaryValueProblem):
    def __init__(
        self,
        domain: DomainLike,
        *,
        rhs: pn.functions.Function | None = None,
        alpha: float = 1.0,
        boundary_values: pn.functions.Function | ArrayLike | None = None,
        solution: pn.functions.Function = None,
    ):
        pde = PoissonEquation(
            domain,
            rhs=rhs,
            alpha=alpha,
        )

        if boundary_values is None:
            boundary_values = functions.Zero(
                pde.domain.shape, pde.diffop.input_codomain_shape
            )

        if pde.domain.shape == ():
            if not isinstance(pde.domain, domains.Interval):
                raise TypeError("In the scalar case, we only support Interval domains.")

            if isinstance(boundary_values, pn.functions.Function):
                a, b = pde.domain
                boundary_values = (boundary_values(a), boundary_values(b))

            boundary_values = np.asarray(boundary_values)

            if solution is None:
                if isinstance(pde.rhs, functions.Constant):
                    solution = Solution_PoissonEquation_DirichletProblem_1D_RHSConstant(
                        pde.domain,
                        rhs=pde.rhs.value,
                        boundary_values=boundary_values,
                        alpha=pde.alpha,
                    )

        if isinstance(boundary_values, pn.functions.Function):
            boundary_conditions = tuple(
                DirichletBoundaryCondition(boundary_part, boundary_values)
                for boundary_part in pde.domain.boundary
            )
        else:
            boundary_values = np.asarray(boundary_values)

            boundary_conditions = tuple(
                DirichletBoundaryCondition(boundary_part, boundary_value)
                for boundary_part, boundary_value in zip(
                    pde.domain.boundary, boundary_values
                )
            )

        super().__init__(
            pde=pde,
            boundary_conditions=boundary_conditions,
            solution=solution,
        )


class Solution_PoissonEquation_DirichletProblem_1D_RHSConstant(functions.JaxFunction):
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

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        l, r = self._l, self._r
        a = self._coeffs

        return (a[2] * (x - r) + a[1]) * (x - l) + a[0]


class Solution_PoissonEquation_IVP_1D_RHSPolynomial(functions.Polynomial):
    def __init__(
        self,
        domain: DomainLike,
        rhs: functions.Polynomial,
        initial_values: ArrayLike,
        alpha: FloatLike,
    ) -> None:
        domain = domains.asdomain(domain)

        if not isinstance(domain, domains.Interval):
            raise TypeError("We only support Interval domains.")

        self._l, self._r = domain

        if not isinstance(rhs, functions.Polynomial):
            raise TypeError("`rhs` needs to be a `Polynomial`.")

        self._rhs = rhs
        self._initial_values = np.asarray(initial_values)
        self._alpha = float(alpha)

        rhs_int = rhs.integrate()
        rhs_dblint = rhs_int.integrate()

        coeff_1 = self._initial_values[1] - rhs_int(self._l) / -self._alpha
        coeff_0 = (
            self._initial_values[0]
            - self._l * coeff_1
            - rhs_dblint(self._l) / -self._alpha
        )

        super().__init__(
            (coeff_0, coeff_1)
            + tuple(coeff / -self._alpha for coeff in rhs_dblint.coefficients[2:])
        )


class Solution_PoissonEquation_IVP_1D_RHSPiecewisePolynomial(functions.Piecewise):
    def __init__(
        self,
        domain: DomainLike,
        rhs: functions.Piecewise,
        initial_values: ArrayLike,
        alpha: FloatLike,
    ) -> None:
        domain = domains.asdomain(domain)

        if not isinstance(domain, domains.Interval):
            raise TypeError("We only support Interval domains.")

        self._l, self._r = domain

        if not (
            isinstance(rhs, functions.Piecewise)
            and all(isinstance(piece, functions.Polynomial) for piece in rhs.pieces)
        ):
            raise TypeError("`rhs` needs to be piecewise polynomial.")

        self._rhs = rhs
        self._initial_values = np.asarray(initial_values)
        self._alpha = float(alpha)

        sol_pieces = []
        piece_initial_values = self._initial_values

        for rhs_piece, piece_l, piece_r in zip(rhs.pieces, rhs.xs[:-1], rhs.xs[1:]):
            sol_piece = Solution_PoissonEquation_IVP_1D_RHSPolynomial(
                (piece_l, piece_r),
                rhs=rhs_piece,
                initial_values=piece_initial_values,
                alpha=self._alpha,
            )

            sol_pieces.append(sol_piece)
            piece_initial_values = (
                sol_piece(piece_r),
                sol_piece.differentiate()(piece_r),
            )

        super().__init__(xs=rhs.xs, fns=sol_pieces)
