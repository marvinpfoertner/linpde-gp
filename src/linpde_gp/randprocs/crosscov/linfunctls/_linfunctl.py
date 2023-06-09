import probnum as pn

from linpde_gp.linfunctls import LinearFunctional
from linpde_gp.randvars import ArrayCovariance, Covariance, LinearOperatorCovariance

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
def _(self, pv_crosscov: ScaledProcessVectorCrossCovariance, /) -> Covariance:
    inner_cov = self(pv_crosscov.pv_crosscov)
    linop_res = pv_crosscov.scalar * inner_cov.linop
    return LinearOperatorCovariance(linop_res, inner_cov.shape0, inner_cov.shape1)


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, sum_pv_crosscov: SumProcessVectorCrossCovariance, /) -> Covariance:
    first_cov = self(sum_pv_crosscov.pv_crosscovs[0])
    shape0, shape1 = first_cov.shape0, first_cov.shape1
    linop_res = sum(self(summand).linop for summand in sum_pv_crosscov.pv_crosscovs)
    return LinearOperatorCovariance(linop_res, shape0, shape1)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    LinOpProcessVectorCrossCovariance
)
def _(
    self, pv_crosscov: LinOpProcessVectorCrossCovariance, /
) -> pn.linops.LinearOperator:
    cov = self(pv_crosscov.pv_crosscov)
    axis = cov.ndim0 - 1 if pv_crosscov.reverse else cov.ndim0 + cov.ndim1 - 1

    array_res = pv_crosscov.linop(cov.array, axis=axis)
    shape0 = array_res.shape[: cov.ndim0]
    shape1 = array_res.shape[cov.ndim0 :]
    return ArrayCovariance(array_res, shape0, shape1)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Dirac_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Dirac_Identity, /) -> Covariance:
    res = self(pv_crosscov.covfunc, argnum=1)(pv_crosscov.dirac.X)
    return ArrayCovariance(res, pv_crosscov.dirac.output_shape, self.output_shape)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Dirac
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Dirac, /) -> Covariance:
    res = self(pv_crosscov.covfunc, argnum=0)(pv_crosscov.dirac.X)
    return ArrayCovariance(res, self.output_shape, pv_crosscov.dirac.output_shape)


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Evaluation_Identity
)
def _(self, pv_crosscov: CovarianceFunction_Evaluation_Identity, /) -> Covariance:
    res = self(pv_crosscov.covfunc, argnum=1)(pv_crosscov.evaluation_fctl.X)
    return ArrayCovariance(
        res, pv_crosscov.evaluation_fctl.output_shape, self.output_shape
    )


@LinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Evaluation
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Evaluation, /) -> Covariance:
    res = self(pv_crosscov.covfunc, argnum=0)(pv_crosscov.evaluation_fctl.X)
    return ArrayCovariance(
        res, self.output_shape, pv_crosscov.evaluation_fctl.output_shape
    )
