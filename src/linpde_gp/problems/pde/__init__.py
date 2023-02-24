from ._bvp import (
    BoundaryValueProblem,
    DirichletBoundaryCondition,
    get_1d_dirichlet_boundary_observations,
)
from ._heat import (
    HeatEquation,
    HeatEquationDirichletProblem,
    Solution_HeatEquation_DirichletProblem_1D_InitialTruncatedSineSeries_BoundaryZero,
)
from ._linear_pde import LinearPDE
from ._poisson import (
    PoissonEquation,
    PoissonEquationDirichletProblem,
    Solution_PoissonEquation_DirichletProblem_1D_RHSConstant,
    Solution_PoissonEquation_IVP_1D_RHSPiecewisePolynomial,
    Solution_PoissonEquation_IVP_1D_RHSPolynomial,
)
