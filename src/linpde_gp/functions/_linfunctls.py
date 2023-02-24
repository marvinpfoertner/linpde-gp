import numpy as np

from linpde_gp.linfunctls import LebesgueIntegral

from ._polynomial import Polynomial


@LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Polynomial) -> np.ndarray:
    f_antideriv = f.integrate()

    return f_antideriv(self.domain[1]) - f_antideriv(self.domain[0])
