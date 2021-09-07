from typing import Callable, Iterable, Optional, Union

import numpy as np
import probnum as pn


def euclidean_inprod(
    v: np.ndarray,
    w: np.ndarray,
    A: Optional[Union[np.ndarray, pn.linops.LinearOperator]] = None,
) -> np.ndarray:
    r"""(Modified) Euclidean inner product :math:`\langle v, w \rangle_A := v^T A w`."""

    v_T = v[..., None, :]
    w = w[..., :, None]

    if A is None:
        vw_inprod = v_T @ w
    else:
        vw_inprod = v_T @ (A @ w)

    return np.squeeze(vw_inprod, axis=(-2, -1))


def euclidean_norm(
    v: np.ndarray,
    A: Optional[Union[np.ndarray, pn.linops.LinearOperator]] = None,
) -> np.ndarray:
    r"""(Modified) Euclidean norm :math:`\lVert v \rVert_A := \sqrt{v^T A v}`."""

    if A is None:
        return np.linalg.norm(v, ord=2, axis=-1, keepdims=False)

    return np.sqrt(euclidean_inprod(v, v, A))


def gram_schmidt(
    v: np.ndarray,
    orthogonal_basis: Iterable[np.ndarray],
    inprod: Optional[
        Union[
            np.ndarray,
            pn.linops.LinearOperator,
            Callable[[np.ndarray, np.ndarray], np.ndarray],
        ]
    ] = None,
    normalize: bool = False,
) -> np.ndarray:
    if inprod is None:
        inprod_fn = euclidean_inprod
        norm_fn = euclidean_norm
    elif isinstance(inprod, (np.ndarray, pn.linops.LinearOperator)):
        inprod_fn = lambda v, w: euclidean_inprod(v, w, A=inprod)
        norm_fn = lambda v: euclidean_norm(v, A=inprod)
    else:
        inprod_fn = inprod
        norm_fn = lambda v: np.sqrt(inprod_fn(v, v))

    v_orth = v.copy()

    for u in orthogonal_basis:
        v_orth -= (inprod_fn(u, v) / inprod_fn(u, u)) * u

    if normalize:
        v_orth /= norm_fn(v_orth)

    return v_orth


def modified_gram_schmidt(
    v: np.ndarray,
    orthogonal_basis: Iterable[np.ndarray],
    inprod: Optional[
        Union[
            np.ndarray,
            pn.linops.LinearOperator,
            Callable[[np.ndarray, np.ndarray], np.ndarray],
        ]
    ] = None,
    normalize: bool = False,
) -> np.ndarray:
    if inprod is None:
        inprod_fn = euclidean_inprod
        norm_fn = euclidean_norm
    elif isinstance(inprod, (np.ndarray, pn.linops.LinearOperator)):
        inprod_fn = lambda v, w: euclidean_inprod(v, w, A=inprod)
        norm_fn = lambda v: euclidean_norm(v, A=inprod)
    else:
        inprod_fn = inprod
        norm_fn = lambda v: np.sqrt(inprod_fn(v, v))

    v_orth = v.copy()

    for u in orthogonal_basis:
        v_orth -= (inprod_fn(u, v_orth) / inprod_fn(u, u)) * u

    if normalize:
        v_orth /= norm_fn(v_orth)

    return v_orth


def pairwise_inprods(
    vs: Iterable[np.ndarray],
    ws: Optional[Iterable[np.ndarray]] = None,
    inprod: Optional[
        Union[
            np.ndarray,
            pn.linops.LinearOperator,
            Callable[[np.ndarray, np.ndarray], np.ndarray],
        ]
    ] = None,
    normalize: bool = False,
):
    if inprod is None:
        inprod_fn = euclidean_inprod
        norm_fn = euclidean_norm
    elif isinstance(inprod, (np.ndarray, pn.linops.LinearOperator)):
        inprod_fn = lambda v, w: euclidean_inprod(v, w, A=inprod)
        norm_fn = lambda v: euclidean_norm(v, A=inprod)
    else:
        inprod_fn = inprod
        norm_fn = lambda v: np.sqrt(inprod_fn(v, v))

    assert all(len(v.shape) == 1 for v in vs)
    vs = np.vstack(tuple(v[None, :] for v in vs))

    vw_equal = False

    if ws is not None:
        assert all(len(w.shape) == 1 for w in ws)
        ws = np.vstack(tuple(w[None, :] for w in ws))
    else:
        vw_equal = True

        ws = vs

    inprods = inprod_fn(vs[:, None, :], ws[None, :, :])

    if normalize:
        if vw_equal:
            v_norms = np.sqrt(np.diag(inprods))
            w_norms = v_norms
        else:
            v_norms = norm_fn(vs)
            w_norms = norm_fn(ws)

        inprods /= v_norms[:, None]
        inprods /= w_norms[None, :]

    return inprods


def pivoted_cholesky(A: np.ndarray, k: int) -> np.ndarray:
    """
    TODO:
    - Handle different memory types
    - Implement for linear operators
    """
    N, _ = A.shape

    assert 1 <= k <= N

    L = np.zeros((N, k), dtype=A.dtype, order="F")

    perm = np.arange(N)
    perm_diag = np.diag(A).copy()

    for m in range(k):
        # Pivotization
        i = np.argmax(perm_diag[m:]) + m

        perm[m], perm[i] = perm[i], perm[m]
        perm_diag[m], perm_diag[i] = perm_diag[i], perm_diag[m]

        # Cholesky algorithm
        buf = np.empty_like(L[m:, m])

        buf[0] = np.sqrt(perm_diag[m])  # Pivot

        buf[1:] = A[perm[(m + 1) :], perm[m]]
        buf[1:] -= L[perm[(m + 1) :], :m] @ L[perm[m], :m]
        buf[1:] /= buf[0]

        perm_diag[m:] -= buf ** 2

        L[perm[m:], m] = buf

    return L
