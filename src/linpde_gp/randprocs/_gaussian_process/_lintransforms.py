import numpy as np
import probnum as pn

from linpde_gp.linfuncops import LinearFunctionOperator
from linpde_gp.linfunctls import LinearFunctional


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, gp: pn.randprocs.GaussianProcess, /) -> pn.randvars.Normal:
    mean = self(gp.mean)
    crosscov = self(gp.cov, argnum=1)
    cov = self(crosscov, argnum=0)

    assert isinstance(mean, (np.ndarray, np.number))
    assert isinstance(cov, (np.ndarray, np.number))

    return pn.randvars.Normal(mean, cov)


@LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(self, gp: pn.randprocs.GaussianProcess, /) -> pn.randprocs.GaussianProcess:
    mean = self(gp.mean)
    crosscov = self(gp.cov, argnum=1)
    cov = self(crosscov, argnum=0)

    return pn.randprocs.GaussianProcess(mean, cov)
