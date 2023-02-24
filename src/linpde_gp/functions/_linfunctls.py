import numpy as np

from linpde_gp.linfunctls import LebesgueIntegral

from ._piecewise import Piecewise
from ._polynomial import Polynomial


@LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Polynomial) -> np.ndarray:
    f_antideriv = f.integrate()

    return f_antideriv(self.domain[1]) - f_antideriv(self.domain[0])


@LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, f: Piecewise) -> np.ndarray:
    idx_start = np.searchsorted(f.xs, self.domain[0], side="right") - 1
    idx_stop = np.searchsorted(f.xs, self.domain[1], side="right")

    if idx_start == -1:
        raise ValueError("Integral domain is larger than function domain")

    xs = (
        (self.domain[0],)
        + tuple(f.xs[idx_start + 1 : idx_stop - 1])
        + (self.domain[1],)
    )
    fs = f.pieces[idx_start : idx_stop - 1]

    return sum(
        LebesgueIntegral((piece_l, piece_r))(piece)
        for piece, piece_l, piece_r in zip(fs, xs[:-1], xs[1:])
    )
