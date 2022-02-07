import functools
from collections.abc import Callable, Sequence
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import probnum as pn
import scipy.linalg
from numpy.typing import ArrayLike

from ... import linfuncops
from . import _jax


class PosteriorGaussianProcess(pn.randprocs.GaussianProcess):
    @classmethod
    def from_observations(
        cls,
        prior: pn.randprocs.GaussianProcess,
        X: ArrayLike,
        Y: ArrayLike,
        Y_err: Optional[pn.randvars.Normal] = None,
    ):
        return PosteriorGaussianProcess.from_linop_observations(
            prior=prior,
            L=None,  # measurement operator is evaluation at X
            X=X,
            Y=Y,
            Y_err=Y_err,
        )

    @classmethod
    def from_linop_observations(
        cls,
        prior: pn.randprocs.GaussianProcess,
        L: Optional[linfuncops.LinearFunctionOperator],
        X: ArrayLike,
        Y: ArrayLike,
        Y_err: Optional[pn.randvars.Normal] = None,
    ):
        X, Y, kLa, pred_mean_X, gram_XX = cls._preprocess_observations(
            prior=prior,
            L=L,
            X=X,
            Y=Y,
            Y_err=Y_err,
        )

        # Compute representer weights
        gram_XX_cho = scipy.linalg.cho_factor(gram_XX)

        representer_weights = scipy.linalg.cho_solve(gram_XX_cho, (Y - pred_mean_X))

        return cls(
            prior=prior,
            Xs=(X,),
            Ys=(Y,),
            Y_errs=(Y_err,),
            kLas=(kLa,),
            gram_Xs_Xs_blocks=((gram_XX,),),
            gram_Xs_Xs_cho=gram_XX_cho,
            representer_weights=representer_weights,
        )

    def __init__(
        self,
        prior: pn.randprocs.GaussianProcess,
        Xs: Sequence[np.ndarray],
        Ys: Sequence[np.ndarray],
        Y_errs: Sequence[pn.randvars.Normal],
        kLas: Sequence[pn.randprocs.kernels.Kernel],
        gram_Xs_Xs_blocks: Sequence[Sequence[np.ndarray]],
        gram_Xs_Xs_cho: tuple[np.ndarray, bool],
        representer_weights: np.ndarray,
    ):
        self._prior = prior

        self._Xs = tuple(Xs)
        self._Ys = tuple(Ys)
        self._Y_errs = tuple(Y_errs)

        self._kLas = tuple(kLas)
        self._kLas_Xs: Callable[[np.ndarray], np.ndarray] = lambda x: np.concatenate(
            [kLa(x[..., None, :], X) for kLa, X in zip(self._kLas, self._Xs)],
            axis=-1,
        )
        self._kLas_Xs_jax: Callable[[jnp.ndarray], jnp.ndarray] = jnp.vectorize(
            lambda x: jnp.hstack(kLa.jax(x, X) for kLa, X in zip(self._kLas, self._Xs)),
            signature="(d)->(n)",
        )

        self._gram_Xs_Xs_blocks = tuple(tuple(row) for row in gram_Xs_Xs_blocks)
        self._gram_Xs_Xs_cho = gram_Xs_Xs_cho

        self._representer_weights = representer_weights

        super().__init__(
            mean=PosteriorGaussianProcess.Mean(
                prior_mean=self._prior._meanfun,
                kLas_Xs=self._kLas_Xs,
                kLas_Xs_jax=self._kLas_Xs_jax,
                representer_weights=self._representer_weights,
            ),
            cov=PosteriorGaussianProcess.Kernel(
                prior_kernel=self._prior._covfun,
                kLas_Xs=self._kLas_Xs,
                kLas_Xs_jax=self._kLas_Xs_jax,
                gram_Xs_Xs_cho=self._gram_Xs_Xs_cho,
            ),
        )

    class Mean(_jax.JaxMean):
        def __init__(
            self,
            prior_mean: _jax.JaxMean,
            kLas_Xs: Callable[[np.ndarray], np.ndarray],
            kLas_Xs_jax: Callable[[jnp.ndarray], jnp.ndarray],
            representer_weights: np.ndarray,
        ):
            self._prior_mean = prior_mean

            self._kLas_Xs = kLas_Xs
            self._kLas_Xs_jax = kLas_Xs_jax

            self._representer_weights = representer_weights

            super().__init__(m=self._jax, vectorize=True)

        def __call__(self, x: np.ndarray) -> np.ndarray:
            m_x = self._prior_mean(x)
            kLas_x_Xs = self._kLas_Xs(x)

            return m_x + kLas_x_Xs @ self._representer_weights

        @functools.partial(jax.jit, static_argnums=0)
        def _jax(self, x: jnp.ndarray) -> jnp.ndarray:
            m_x = self._prior_mean.jax(x)
            kLas_x_Xs = self._kLas_Xs_jax(x)

            return m_x + kLas_x_Xs @ self._representer_weights

    class Kernel(_jax.JaxKernel):
        def __init__(
            self,
            prior_kernel: _jax.JaxKernel,
            kLas_Xs: Callable[[np.ndarray], np.ndarray],
            kLas_Xs_jax: Callable[[jnp.ndarray], jnp.ndarray],
            gram_Xs_Xs_cho: np.ndarray,
        ):
            self._prior_kernel = prior_kernel

            self._kLas_Xs = kLas_Xs
            self._kLas_Xs_jax = kLas_Xs_jax

            self._gram_Xs_Xs_cho = gram_Xs_Xs_cho

            super().__init__(
                self._jax,
                input_dim=self._prior_kernel.input_dim,
                vectorize=True,
            )

        def _evaluate(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
            k_xx = self._prior_kernel(x0, x1)
            kLas_x_Xs = self._kLas_Xs(x0)
            Lks_Xs_x = self._kLas_Xs(x1) if x1 is not None else kLas_x_Xs

            return (
                k_xx
                - (
                    kLas_x_Xs[..., None, :]
                    @ scipy.linalg.cho_solve(
                        self._gram_Xs_Xs_cho, Lks_Xs_x.transpose()
                    ).transpose()[..., :, None]
                )[..., 0, 0]
            )

        @functools.partial(jax.jit, static_argnums=0)
        def _jax(self, x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
            k_xx = self._prior_kernel.jax(x0, x1)
            kLas_x_Xs = self._kLas_Xs_jax(x0)
            Lks_Xs_x = self._kLas_Xs_jax(x1)

            return k_xx - kLas_x_Xs @ jax.scipy.linalg.cho_solve(
                self._gram_Xs_Xs_cho, Lks_Xs_x
            )

    @functools.cached_property
    def gram_Xs_Xs(self) -> np.ndarray:
        return np.block(
            [
                [
                    self._gram_Xs_Xs_blocks[i][j]
                    if i >= j
                    else self._gram_Xs_Xs_blocks[j][i].T
                    for j in range(len(self._Xs))
                ]
                for i in range(len(self._Xs))
            ]
        )

    def condition_on_observations(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        Y_err: Optional[pn.randvars.Normal] = None,
    ):
        return self.condition_on_linop_observations(
            L=None,
            X=X,
            Y=Y,
            Y_err=Y_err,
        )

    def condition_on_linop_observations(
        self,
        L: Optional[linfuncops.LinearFunctionOperator],
        X: ArrayLike,
        Y: ArrayLike,
        Y_err: Optional[pn.randvars.Normal] = None,
    ):
        X, Y, kLa, pred_mean_X, gram_XX = self._preprocess_observations(
            prior=self._prior,
            L=L,
            X=X,
            Y=Y,
            Y_err=Y_err,
        )

        # Compute lower-left block in the new kernel gram matrix
        gram_X_Xs_blocks = tuple(
            kLa_prev(X[:, None, :], X_prev[None, :, :])
            if L is None
            else L(kLa_prev, argnum=0)(X[:, None, :], X_prev[None, :, :])
            for kLa_prev, X_prev in zip(self._kLas, self._Xs)
        )
        gram_X_Xs = np.concatenate(gram_X_Xs_blocks, axis=-1)

        gram_X_row_blocks = gram_X_Xs_blocks + (gram_XX,)

        # Update the Cholesky decomposition of the previous kernel Gram matrix and the
        # representer weights
        gram_Xs_Xs_cho, representer_weights = _block_cholesky(
            A_cho=self._gram_Xs_Xs_cho,
            B=gram_X_Xs.T,
            D=gram_XX,
            sol_update=(
                self._representer_weights,
                Y - pred_mean_X,
            ),
        )

        return PosteriorGaussianProcess(
            self._prior,
            Xs=self._Xs + (X,),
            Ys=self._Ys + (Y,),
            Y_errs=self._Y_errs + (Y_err,),
            kLas=self._kLas + (kLa,),
            gram_Xs_Xs_blocks=self._gram_Xs_Xs_blocks + (gram_X_row_blocks,),
            gram_Xs_Xs_cho=gram_Xs_Xs_cho,
            representer_weights=representer_weights,
        )

    @classmethod
    def _preprocess_observations(
        cls,
        prior: pn.randprocs.GaussianProcess,
        L: Optional[linfuncops.LinearFunctionOperator],
        X: ArrayLike,
        Y: ArrayLike,
        Y_err: Optional[Union[pn.randvars.Normal, pn.randvars.Constant]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        pn.randprocs.GaussianProcess,
        pn.randprocs.kernels.Kernel,
        np.ndarray,
        np.ndarray,
    ]:
        # Reshape to (N, input_dim) and (N,)
        X = np.asarray(X)
        Y = np.asarray(Y)

        assert prior.output_dim is None  # TODO: Support vector-valued GPs
        assert X.ndim >= 1 and X.shape[-1] == prior.input_dim
        assert Y.shape == X.shape[:-1]  # TODO: Support vector-valued linfuncops

        X = X.reshape((-1, prior.input_dim), order="C")
        Y = Y.reshape((-1,), order="C")

        # Apply measurement operator to prior
        if L is None:
            Lf = prior
            kLa = prior._covfun
        else:
            Lf = L(prior)
            kLa = L(prior._covfun, argnum=1)

        # Compute predictive mean and kernel Gram matrix
        pred_mean_X = Lf._meanfun(X)
        gram_XX = Lf._covfun(X[:, None, :], X[None, :, :])

        if Y_err is not None:
            assert isinstance(Y_err, (pn.randvars.Constant, pn.randvars.Normal))
            assert Y_err.shape == Y.shape

            pred_mean_X += Y_err.mean.reshape((-1,), order="C")
            # This assumes that the covariance matrix is raveled in C-order
            gram_XX += Y_err.cov

        return X, Y, kLa, pred_mean_X, gram_XX


pn.randprocs.GaussianProcess.condition_on_observations = (
    lambda *args, **kwargs: PosteriorGaussianProcess.from_observations(*args, **kwargs)
)


pn.randprocs.GaussianProcess.condition_on_linop_observations = (
    lambda *args, **kwargs: PosteriorGaussianProcess.from_linop_observations(
        *args, **kwargs
    )
)


@linfuncops.LinearFunctionOperator.__call__.register
def _(self, f: PosteriorGaussianProcess, **kwargs) -> PosteriorGaussianProcess:
    linop_prior = self(f._prior)

    return PosteriorGaussianProcess(
        prior=linop_prior,
        Xs=f._Xs,
        Ys=f._Ys,
        Y_errs=f._Y_errs,
        kLas=[self(k_cross, argnum=0) for k_cross in f._kLas],
        gram_Xs_Xs_blocks=f._gram_Xs_Xs_blocks,
        gram_Xs_Xs_cho=f._gram_Xs_Xs_cho,
        representer_weights=f._representer_weights,
    )


def _schur_update(A_cho, B, C, D, A_inv_u, v):
    """
    This function solves the linear system

    [[A, B], @ [[x], = [[u],
     [C, D]]    [y]]    [v]]

    given the Cholesky factor of A, the matrices B, C, and D, and the vectors A^{-1} u, and v.
    """
    A_inv_B = jax.scipy.linalg.cho_solve(A_cho, B)

    S = D - C @ A_inv_B

    y = jax.scipy.linalg.solve(S, v - C @ A_inv_u)
    x = A_inv_u - A_inv_B @ y

    return jnp.concatenate((x, y))


def _block_cholesky(
    A_cho: tuple[np.ndarray, bool],
    B: np.ndarray,
    D: np.ndarray,
    sol_update: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, bool]:
    A_sqrt, lower = A_cho

    tri = np.tril if lower else np.triu

    A_sqrt = tri(A_sqrt)

    L_A_inv_B = scipy.linalg.solve_triangular(
        A_sqrt,
        B,
        lower=lower,
        trans="N" if lower else "T",
    )

    # Compute the Schur complement of A in the block matrix
    S = D - L_A_inv_B.T @ L_A_inv_B
    S_sqrt = scipy.linalg.cholesky(S, lower=lower)
    S_cho = (S_sqrt, lower)

    # Assemble the block Cholesky factor
    block_sqrt = np.empty_like(
        A_sqrt,
        shape=(
            A_sqrt.shape[0] + D.shape[0],
            A_sqrt.shape[1] + D.shape[1],
        ),
    )

    A_i_end, A_j_end = A_sqrt.shape
    D_i_start, D_j_start = A_sqrt.shape

    # A-block
    block_sqrt[:A_i_end, :A_j_end] = A_sqrt

    # B-block
    block_sqrt[:A_i_end, D_j_start:] = 0 if lower else L_A_inv_B

    # C-block
    block_sqrt[D_i_start:, :A_j_end] = L_A_inv_B.T if lower else 0

    # D-block
    block_sqrt[D_i_start:, D_j_start:] = S_sqrt

    block_cho = (block_sqrt, lower)

    if sol_update is not None:
        C = B.T
        A_inv_u, v = sol_update

        y = scipy.linalg.cho_solve(S_cho, v - C @ A_inv_u)
        x = A_inv_u - scipy.linalg.solve_triangular(  # A^{-1} @ B @ y
            A_sqrt,
            L_A_inv_B @ y,
            lower=lower,
            trans="T" if lower else "N",
        )

        return block_cho, np.concatenate((x, y))

    return block_cho
