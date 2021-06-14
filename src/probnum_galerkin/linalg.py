from typing import Iterable, Optional, Union

import numpy as np
import probnum as pn


def gram_schmidt(
    v: np.ndarray,
    orthogonal_basis: Iterable[np.ndarray],
    normalize: bool = False,
    inprod: Optional[Union[np.ndarray, pn.linops.LinearOperator]] = None,
) -> np.ndarray:
    if inprod is not None:
        inprod = lambda x, y: np.inner(x, inprod @ y)
    else:
        inprod = lambda x, y: np.inner(x, y)

    v_orth = v.copy()

    for u in orthogonal_basis:
        v_orth -= (inprod(u, v) / inprod(u, u)) * u

    if normalize:
        v_orth /= np.sqrt(inprod(v_orth, v_orth))

    return v_orth


def modified_gram_schmidt(
    v: np.ndarray,
    orthogonal_basis: Iterable[np.ndarray],
    normalize: bool = False,
    inprod_matrix: Optional[Union[np.ndarray, pn.linops.LinearOperator]] = None,
) -> np.ndarray:
    if inprod_matrix is not None:
        inprod_ = lambda x, y: np.inner(x, inprod_matrix @ y)
    else:
        inprod_ = lambda x, y: np.inner(x, y)

    v_orth = v.copy()

    for u in orthogonal_basis:
        v_orth -= (inprod_(u, v_orth) / inprod_(u, u)) * u

    if normalize:
        v_orth /= np.sqrt(inprod_(v_orth, v_orth))

    return v_orth
