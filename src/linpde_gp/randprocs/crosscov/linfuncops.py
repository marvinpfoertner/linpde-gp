import numpy as np

from linpde_gp.linfuncops import LinearFunctionOperator

from ._arithmetic import ScaledProcessVectorCrossCovariance
from .linfunctls._dirac import (
    CovarianceFunction_Dirac_Identity,
    CovarianceFunction_Identity_Dirac,
)
from .linfunctls._reshaped import (
    CovarianceFunction_Reshaped_Identity, 
    CovarianceFunction_Identity_Reshaped
)
from .linfunctls._stacked import (
    CovarianceFunction_Stacked_Identity, 
    CovarianceFunction_Identity_Stacked
)


@LinearFunctionOperator.__call__.register  # pylint: disable=no-member
def _(
    self, pv_crosscov: ScaledProcessVectorCrossCovariance, /
) -> ScaledProcessVectorCrossCovariance:
    return ScaledProcessVectorCrossCovariance(
        pv_crosscov=self(pv_crosscov._pv_crosscov),
        scalar=pv_crosscov._scalar,
    )


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Dirac
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Dirac, /) -> np.ndarray:
    return CovarianceFunction_Identity_Dirac(
        self(pv_crosscov.covfunc, argnum=0),
        pv_crosscov.dirac,
    )


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Dirac_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Dirac_Identity, /) -> np.ndarray:
    return CovarianceFunction_Dirac_Identity(
        self(pv_crosscov, argnum=1),
        pv_crosscov.dirac,
    )

@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Reshaped
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Reshaped, /) -> np.ndarray:
    return CovarianceFunction_Identity_Reshaped(
        self(pv_crosscov.covfunc, argnum=0),
        pv_crosscov.reshaped,
    )


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Reshaped_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Reshaped_Identity, /) -> np.ndarray:
    return CovarianceFunction_Reshaped_Identity(
        self(pv_crosscov, argnum=1),
        pv_crosscov.reshaped,
    )

@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Stacked
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Stacked, /) -> np.ndarray:
    return CovarianceFunction_Identity_Stacked(
        self(pv_crosscov.covfunc, argnum=0),
        pv_crosscov.stacked,
    )


@LinearFunctionOperator.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Stacked_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Stacked_Identity, /) -> np.ndarray:
    return CovarianceFunction_Stacked_Identity(
        self(pv_crosscov, argnum=1),
        pv_crosscov.stacked,
    )