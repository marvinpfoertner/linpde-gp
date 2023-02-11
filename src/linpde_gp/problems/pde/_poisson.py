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

            assert isinstance(pde.domain.boundary, domains.PointSet)

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


class Solution_PoissonEquation_DirichletProblem_1D_RHSConstant(pn.functions.Function):
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
