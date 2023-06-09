import dataclasses
import functools

import numpy as np
import probnum as pn

import linpde_gp


@dataclasses.dataclass(frozen=True)
class CovarianceFunctionLinearFunctionalsTestCase:
    covfunc: pn.randprocs.covfuncs.CovarianceFunction

    L0: linpde_gp.linfunctls.LinearFunctional | None
    L1: linpde_gp.linfunctls.LinearFunctional | None

    L0kL1_fallback: linpde_gp.randvars.Covariance

    @functools.cached_property
    def L0kL1(self) -> linpde_gp.randvars.Covariance:
        return self.L0(self.L1(self.covfunc, argnum=1))
