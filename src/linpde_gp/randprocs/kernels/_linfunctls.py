import probnum as pn
from probnum.randprocs.kernels._arithmetic_fallbacks import ScaledKernel

from linpde_gp.linfunctls import DiracFunctional, LebesgueIntegral, LinearFunctional

from ._matern import Matern


@LinearFunctional.__call__.register  # pylint: disable=no-member
def _(self, k_scaled: ScaledKernel, /, *, argnum: int = 0):
    return k_scaled._scalar * self(k_scaled._kernel, argnum=argnum)


@DiracFunctional.__call__.register  # pylint: disable=no-member
def _(self, k: pn.randprocs.kernels.Kernel, /, *, argnum: int = 0):
    match argnum:
        case 0:
            from ..crosscov.linfunctls import Kernel_Dirac_Identity

            return Kernel_Dirac_Identity(
                kernel=k,
                dirac=self,
            )
        case 1:
            from ..crosscov.linfunctls import Kernel_Identity_Dirac

            return Kernel_Identity_Dirac(
                kernel=k,
                dirac=self,
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
