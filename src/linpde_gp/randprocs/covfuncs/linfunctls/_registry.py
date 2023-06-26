import functools
import operator

from probnum.randprocs import covfuncs as pn_covfuncs
from probnum.randprocs.covfuncs._arithmetic_fallbacks import (
    ScaledCovarianceFunction,
    SumCovarianceFunction,
)

from linpde_gp import linfunctls
from linpde_gp.linfunctls._evaluation import _EvaluationFunctional
from linpde_gp.linfunctls.projections.l2 import (
    L2Projection_UnivariateLinearInterpolationBasis,
)
from linpde_gp.randprocs import covfuncs

from .._utils import validate_covfunc_transformation

########################################################################################
# General `LinearFunctional`s ##########################################################
########################################################################################


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k_scaled: ScaledCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k_scaled, argnum)

    # pylint: disable=protected-access
    return k_scaled._scalar * self(k_scaled._covfunc, argnum=argnum)


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: SumCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    # pylint: disable=protected-access
    return functools.reduce(
        operator.add, (self(summand, argnum=argnum) for summand in k._summands)
    )


@linfunctls.CompositeLinearFunctional.__call__.register(  # pylint: disable=no-member
    pn_covfuncs.CovarianceFunction
)
def _(self, k: pn_covfuncs.CovarianceFunction, /, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    res = k

    if self.linfuncop is not None:
        res = self.linfuncop(res, argnum=argnum)

    res = self.linfunctl(res, argnum=argnum)

    if self.linop is not None:
        from ...crosscov import (  # pylint: disable=import-outside-toplevel
            LinOpProcessVectorCrossCovariance,
        )

        res = LinOpProcessVectorCrossCovariance(self.linop, res)

    return res


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.StackCovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    from ...crosscov import (  # pylint: disable=import-outside-toplevel
        StackedProcessVectorCrossCovariance,
    )

    return StackedProcessVectorCrossCovariance(
        *(self(covfunc, argnum=argnum) for covfunc in k.covfuncs)
    )


@linfunctls.LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: covfuncs.Zero, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    from ...crosscov import Zero  # pylint: disable=import-outside-toplevel

    return Zero(
        randproc_input_shape=k.input_shape_1 if argnum == 0 else k.input_shape_0,
        randproc_output_shape=k.output_shape_1 if argnum == 0 else k.output_shape_0,
        randvar_shape=self.output_shape,
        reverse=(argnum == 0),
    )


########################################################################################
# Point Evaluation #####################################################################
########################################################################################


@linfunctls.DiracFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    match argnum:
        case 0:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Dirac_Identity,
            )

            return CovarianceFunction_Dirac_Identity(
                covfunc=k,
                dirac=self,
            )
        case 1:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Identity_Dirac,
            )

            return CovarianceFunction_Identity_Dirac(
                covfunc=k,
                dirac=self,
            )

    raise ValueError("`argnum` must either be 0 or 1.")


@_EvaluationFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    match argnum:
        case 0:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Evaluation_Identity,
            )

            return CovarianceFunction_Evaluation_Identity(
                covfunc=k,
                evaluation_fctl=self,
            )
        case 1:
            from ...crosscov.linfunctls import (  # pylint: disable=import-outside-toplevel
                CovarianceFunction_Identity_Evaluation,
            )

            return CovarianceFunction_Identity_Evaluation(
                covfunc=k,
                evaluation_fctl=self,
            )

    raise ValueError("`argnum` must either be 0 or 1.")


########################################################################################
# Integrals ############################################################################
########################################################################################


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def covfunc_lebesgue_integral(
    self, k: pn_covfuncs.CovarianceFunction, /, *, argnum: int = 0
):
    validate_covfunc_transformation(self, k, argnum)

    try:
        return super(linfunctls.LebesgueIntegral, self).__call__(k, argnum=argnum)
    except NotImplementedError:
        from ...crosscov.linfunctls.integrals import (  # pylint: disable=import-outside-toplevel
            CovarianceFunction_Identity_LebesgueIntegral,
        )

        return CovarianceFunction_Identity_LebesgueIntegral(
            k, self, reverse=(argnum == 0)
        )


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, k: pn_covfuncs.Matern, /, *, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.input_shape == () and k.is_half_integer:
        from ...crosscov.linfunctls.integrals import (  # pylint: disable=import-outside-toplevel
            UnivariateHalfIntegerMaternLebesgueIntegral,
        )

        return UnivariateHalfIntegerMaternLebesgueIntegral(
            matern=k,
            integral=self,
            reverse=(argnum == 0),
        )

    return covfunc_lebesgue_integral(self, k, argnum=argnum)


########################################################################################
# Projections ##########################################################################
########################################################################################


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register(  # pylint: disable=no-member
    pn_covfuncs.CovarianceFunction
)
def _(self, k: pn_covfuncs.CovarianceFunction, /, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    try:
        return super(L2Projection_UnivariateLinearInterpolationBasis, self).__call__(
            k, argnum=argnum
        )
    except NotImplementedError:
        from ...crosscov.linfunctls.projections import (  # pylint: disable=import-outside-toplevel
            CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis,
        )

        return CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis(
            covfunc=k,
            proj=self,
            reverse=(argnum == 0),
        )


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register(  # pylint: disable=no-member
    pn_covfuncs.Matern,
)
def _(self, k: pn_covfuncs.Matern, /, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    from ...crosscov.linfunctls.projections import (  # pylint: disable=import-outside-toplevel
        CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis,
        Matern32_L2Projection_UnivariateLinearInterpolationBasis,
    )

    if k.nu == 1.5:
        return Matern32_L2Projection_UnivariateLinearInterpolationBasis(
            covfunc=k,
            proj=self,
            reverse=(argnum == 0),
        )

    return CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis(
        covfunc=k,
        proj=self,
        reverse=(argnum == 0),
    )


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register(  # pylint: disable=no-member
    covfuncs.GalerkinCovarianceFunction
)
def _(self, k: covfuncs.GalerkinCovarianceFunction, /, argnum: int = 0):
    validate_covfunc_transformation(self, k, argnum)

    if k.P is self:
        from ...crosscov import (  # pylint: disable=import-outside-toplevel
            ParametricProcessVectorCrossCovariance,
        )

        return ParametricProcessVectorCrossCovariance(
            crosscov=k.PkP,
            basis=k.P.basis,
            reverse=(argnum == 0),
        )

    return super(L2Projection_UnivariateLinearInterpolationBasis, self).__call__(
        k, argnum=argnum
    )
