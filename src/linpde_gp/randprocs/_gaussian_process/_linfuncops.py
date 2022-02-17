import probnum as pn

from ... import linfuncops


@linfuncops.LinearFunctionOperator.__call__.register
def _(self, f: pn.randprocs.GaussianProcess, **kwargs):
    mean = self(f.mean, argnum=0)
    crosscov = self(f.cov, argnum=1)
    cov = self(crosscov, argnum=0)

    return pn.randprocs.GaussianProcess(mean, cov)
