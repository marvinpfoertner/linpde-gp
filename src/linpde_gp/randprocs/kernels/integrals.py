from linpde_gp.linfunctls import LebesgueIntegral

from ._matern import Matern


@LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, k: Matern, /, *, argnum: int = 0):
    if argnum not in (0, 1):
        raise ValueError("`argnum` must either be 0 or 1.")

    from ..crosscov.integrals import (  # pylint: disable=import-outside-toplevel
        Matern_Identity_LebesgueIntegral,
    )

    return Matern_Identity_LebesgueIntegral(
        matern=k,
        integral=self,
        reverse=(argnum == 0),
    )
