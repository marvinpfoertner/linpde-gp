import probnum as pn
from probnum.randprocs.covfuncs._arithmetic_fallbacks import ScaledCovarianceFunction

from linpde_gp.linfunctls import (
    CompositeLinearFunctional,
    DiracFunctional,
    FlattenedLinearFunctional,
    LebesgueIntegral,
    LinearFunctional,
)
from linpde_gp.linfunctls.projections.l2 import (
    L2Projection_UnivariateLinearInterpolationBasis,
)

from ._galerkin import GalerkinCovarianceFunction
from ._matern import Matern


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k_scaled: ScaledCovarianceFunction, /, *, argnum: int = 0):
    return k_scaled._scalar * self(k_scaled._covfunc, argnum=argnum)


@CompositeLinearFunctional.__call__.register(  # pylint: disable=no-member
    pn.randprocs.covfuncs.CovarianceFunction
)
def _(self, k: pn.randprocs.covfuncs.CovarianceFunction, /, argnum: int = 0):
    res = k

    if self.linfuncop is not None:
        res = self.linfuncop(res, argnum=argnum)

    res = self.linfunctl(res, argnum=argnum)

    if self.linop is not None:
        from ..crosscov import LinOpProcessVectorCrossCovariance

        res = LinOpProcessVectorCrossCovariance(self.linop, res)

    return res

@DiracFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn.randprocs.covfuncs.CovarianceFunction, /, *, argnum: int = 0):
    match argnum:
        case 0:
            from ..crosscov.linfunctls import CovarianceFunction_Dirac_Identity

            return CovarianceFunction_Dirac_Identity(
                covfunc=k,
                dirac=self,
            )
        case 1:
            from ..crosscov.linfunctls import CovarianceFunction_Identity_Dirac

            return CovarianceFunction_Identity_Dirac(
                covfunc=k,
                dirac=self,
            )

    raise ValueError("`argnum` must either be 0 or 1.")

@FlattenedLinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn.randprocs.kernels.Kernel, /, *, argnum: int = 0):
    match argnum:
        case 0:
            from ..crosscov.linfunctls import CovarianceFunction_Flattened_Identity

            return CovarianceFunction_Flattened_Identity(
                covfunc=k,
                flatten=self,
            )
        case 1:
            from ..crosscov.linfunctls import CovarianceFunction_Identity_Flattened

            return CovarianceFunction_Identity_Flattened(
                covfunc=k,
                flatten=self,
            )

    raise ValueError("`argnum` must either be 0 or 1.")


@LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, k: Matern, /, *, argnum: int = 0):
    if argnum not in (0, 1):
        raise ValueError("`argnum` must either be 0 or 1.")

    from ..crosscov.linfunctls.integrals import (  # pylint: disable=import-outside-toplevel
        Matern_Identity_LebesgueIntegral,
    )

    return Matern_Identity_LebesgueIntegral(
        matern=k,
        integral=self,
        reverse=(argnum == 0),
    )


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register(  # pylint: disable=no-member
    pn.randprocs.covfuncs.CovarianceFunction
)
def _(self, k: pn.randprocs.covfuncs.CovarianceFunction, /, argnum: int = 0):
    from ..crosscov.linfunctls.projections import (  # pylint: disable=import-outside-toplevel
        CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis,
    )

    return CovarianceFunction_L2Projection_UnivariateLinearInterpolationBasis(
        covfunc=k,
        proj=self,
        reverse=(argnum == 0),
    )


@L2Projection_UnivariateLinearInterpolationBasis.__call__.register(  # pylint: disable=no-member
    pn.randprocs.covfuncs.Matern,
)
def _(self, k: pn.randprocs.covfuncs.Matern, /, argnum: int = 0):
    from ..crosscov.linfunctls.projections import (  # pylint: disable=import-outside-toplevel
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
    GalerkinCovarianceFunction
)
def _(self, k: GalerkinCovarianceFunction, /, argnum: int = 0):
    if k._projection is self:
        from ..crosscov import ParametricProcessVectorCrossCovariance

        return ParametricProcessVectorCrossCovariance(
            crosscov=k._PkPa,
            basis=k._projection._basis,
            reverse=(argnum == 0),
        )

    return super(L2Projection_UnivariateLinearInterpolationBasis, self).__call__(
        k, argnum=argnum
    )
