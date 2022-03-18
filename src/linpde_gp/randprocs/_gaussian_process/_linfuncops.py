import probnum as pn

from ... import linfuncops


@linfuncops.LinearFunctionOperator.__call__.register
def _(self, gp: pn.randprocs.GaussianProcess, /) -> pn.randprocs.GaussianProcess:
    mean = self(gp.mean)
    crosscov = self(gp.cov, argnum=1)
    cov = self(crosscov, argnum=0)

    return pn.randprocs.GaussianProcess(mean, cov)
