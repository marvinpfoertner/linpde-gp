import numpy as np
import probnum as pn

from linpde_gp.linfunctls import LinearFunctional

from .._arithmetic import (
    LinOpProcessVectorCrossCovariance,
    ScaledProcessVectorCrossCovariance,
    SumProcessVectorCrossCovariance,
)
from ._dirac import CovarianceFunction_Dirac_Identity, CovarianceFunction_Identity_Dirac
from ._evaluation import (
    CovarianceFunction_Evaluation_Identity,
    CovarianceFunction_Identity_Evaluation,
)


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(
    self, pv_crosscov: ScaledProcessVectorCrossCovariance, /
) -> pn.linops.LinearOperator:
    return pv_crosscov.scalar * self(pv_crosscov.pv_crosscov)


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(
    self, sum_pv_crosscov: SumProcessVectorCrossCovariance, /
) -> pn.linops.LinearOperator:
    return sum(self(summand) for summand in sum_pv_crosscov._pv_crosscovs)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    LinOpProcessVectorCrossCovariance
)
def _(
    self, pv_crosscov: LinOpProcessVectorCrossCovariance, /
) -> pn.linops.LinearOperator:
    return pv_crosscov.linop(
        self(pv_crosscov.pv_crosscov),
        axis=0 if pv_crosscov.reverse else -1,
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Dirac_Identity
)
def _(
    self, pv_crosscov: CovarianceFunction_Dirac_Identity, /
) -> pn.linops.LinearOperator:
    return self(pv_crosscov.covfunc, argnum=1)(pv_crosscov.dirac.X)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Dirac
)
def _(
    self, pv_crosscov: CovarianceFunction_Identity_Dirac, /
) -> pn.linops.LinearOperator:
    return self(pv_crosscov.covfunc, argnum=0)(pv_crosscov.dirac.X)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Evaluation_Identity
)
def _(
    self, pv_crosscov: CovarianceFunction_Evaluation_Identity, /
) -> pn.linops.LinearOperator:
    return self(pv_crosscov.covfunc, argnum=1)(pv_crosscov.evaluation_fctl.X)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Evaluation
)
def _(
    self, pv_crosscov: CovarianceFunction_Identity_Evaluation, /
) -> pn.linops.LinearOperator:
    return self(pv_crosscov.covfunc, argnum=0)(pv_crosscov.evaluation_fctl.X)
