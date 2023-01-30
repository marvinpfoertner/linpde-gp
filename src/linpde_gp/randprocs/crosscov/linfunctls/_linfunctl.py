import numpy as np

from linpde_gp.linfunctls import LinearFunctional

from .._arithmetic import (
    LinOpProcessVectorCrossCovariance,
    ScaledProcessVectorCrossCovariance,
)
from ._dirac import Kernel_Dirac_Identity, Kernel_Identity_Dirac


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: ScaledProcessVectorCrossCovariance, /) -> np.ndarray:
    return pv_crosscov.scalar * self(pv_crosscov.pv_crosscov)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    LinOpProcessVectorCrossCovariance
)
def _(self, pv_crosscov: LinOpProcessVectorCrossCovariance, /) -> np.ndarray:
    return pv_crosscov.linop(
        self(pv_crosscov.pv_crosscov),
        axis=0 if pv_crosscov.reverse else -1,
    )


@LinearFunctional.__call__.register(Kernel_Dirac_Identity)  # pylint: disable=no-member
def _(self, pv_crosscov: Kernel_Dirac_Identity, /) -> np.ndarray:
    return self(pv_crosscov.kernel, argnum=1)(pv_crosscov.dirac.X)


@LinearFunctional.__call__.register(Kernel_Identity_Dirac)  # pylint: disable=no-member
def _(self, pv_crosscov: Kernel_Identity_Dirac, /) -> np.ndarray:
    return self(pv_crosscov.kernel, argnum=0)(pv_crosscov.dirac.X)
