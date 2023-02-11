from ._bvp import (
    BoundaryValueProblem,
    DirichletBoundaryCondition,
    get_1d_dirichlet_boundary_observations,
)
from ._heat import HeatEquation, heat_1d_bvp
from ._linear_pde import LinearPDE
from ._poisson import (
    PoissonEquation,
    PoissonEquationDirichletProblem,
    Solution_PoissonEquation_DirichletProblem_1D_RHSConstant,
)
