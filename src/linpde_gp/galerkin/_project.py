import probnum as pn

from linpde_gp.problems.pde import BoundaryValueProblem

from . import bases
from ._project_operator import project as project_linfuncop
from ._project_rhs import project as project_function


def project(bvp: BoundaryValueProblem, basis: bases.Basis):
    A = project_linfuncop(bvp.diffop, basis)
    b = project_function(bvp.rhs, basis)

    return pn.problems.LinearSystem(pn.linops.aslinop(A), b)
