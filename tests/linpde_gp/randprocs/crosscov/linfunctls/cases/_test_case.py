import dataclasses
import functools
from typing import Type

import numpy as np
import probnum as pn

import linpde_gp


@dataclasses.dataclass(frozen=True)
class CovarianceFunctionLinearFunctionalTestCase:
    k: pn.randprocs.covfuncs.CovarianceFunction
    L: linpde_gp.linfunctls.LinearFunctional | None

    Lk_fallback: linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance
    kL_fallback: linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance

    X_test: np.ndarray

    expected_type: Type[pn.randprocs.covfuncs.CovarianceFunction] = None

    @functools.cached_property
    def Lk(self) -> linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance:
        return self.L(self.k, argnum=0)

    @functools.cached_property
    def kL(self) -> linpde_gp.randprocs.crosscov.ProcessVectorCrossCovariance:
        return self.L(self.k, argnum=1)
