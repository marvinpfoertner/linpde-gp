import numpy as np
import probnum as pn


def outer(u: np.ndarray, v: np.ndarray) -> pn.linops.LinearOperator:
    return pn.linops.aslinop(u[:, None]) @ pn.linops.aslinop(v[None, :])
