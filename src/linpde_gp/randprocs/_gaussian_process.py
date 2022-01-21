from collections.abc import Sequence

import jax
import jax.numpy as np
import probnum as pn
import scipy.linalg

from . import _jax


class PosteriorGaussianProcess(pn.randprocs.GaussianProcess):
    @classmethod
    def from_measurements(
        cls, prior: pn.randprocs.GaussianProcess, X: np.ndarray, fX: np.ndarray
    ):
        mX = prior._meanfun(X)
        kXX = prior._covfun.jax(X[:, None, :], X[None, :, :])
        kXX_cho = jax.scipy.linalg.cho_factor(kXX)

        representer_weights = jax.scipy.linalg.cho_solve(kXX_cho, (fX - mX))

        return cls(
            prior=prior,
            locations=[X],
            measurements=[fX],
            cross_covariances=[prior._covfun],
            gram_matrices=[[kXX]],
            representer_weights=representer_weights,
        )

    def __init__(
        self,
        prior: pn.randprocs.RandomProcess,
        locations: Sequence[np.ndarray],
        measurements: Sequence[np.ndarray],
        cross_covariances: Sequence[pn.randprocs.kernels.Kernel],
        gram_matrices: Sequence[Sequence[np.ndarray]],
        representer_weights: np.ndarray,
    ):
        self._prior = prior

        self._locations = tuple(locations)
        self._measurements = tuple(measurements)

        self._cross_covariances = tuple(cross_covariances)
        self._cross_covariance = lambda x: np.hstack(
            k_cross.jax(x, X)
            for k_cross, X in zip(self._cross_covariances, self._locations)
        )

        self._gram_matrices = tuple(tuple(row) for row in gram_matrices)
        self._gram_matrix = np.block(
            [
                [
                    self._gram_matrices[i][j] if i >= j else self._gram_matrices[j][i].T
                    for j in range(len(self._locations))
                ]
                for i in range(len(self._locations))
            ]
        )
        self._gram_matrix_cholesky = scipy.linalg.cho_factor(self._gram_matrix)

        self._representer_weights = representer_weights

        super().__init__(
            mean=PosteriorGaussianProcess.Mean(
                prior_mean=self._prior._meanfun,
                cross_covariance=self._cross_covariance,
                representer_weights=self._representer_weights,
            ),
            cov=PosteriorGaussianProcess.Kernel(
                prior_kernel=self._prior._covfun,
                cross_covariance=self._cross_covariance,
                gram_matrix_cholesky=self._gram_matrix_cholesky,
            ),
        )

    class Mean(_jax.JaxMean):
        def __init__(
            self,
            prior_mean: _jax.JaxMean,
            cross_covariance,
            representer_weights: np.ndarray,
        ):
            self._prior_mean = prior_mean
            self._representer_weights = representer_weights
            self._cross_covariance = cross_covariance

            super().__init__(m=self._call, vectorize=True)

        def _call(self, x):
            mx = self._prior_mean(x)
            kLadj_xX = self._cross_covariance(x)

            return mx + kLadj_xX @ self._representer_weights

    class Kernel(_jax.JaxKernel):
        def __init__(
            self,
            prior_kernel: _jax.JaxKernel,
            cross_covariance,
            gram_matrix_cholesky,
        ):
            self._prior_kernel = prior_kernel
            self._gram_matrix_cholesky = gram_matrix_cholesky
            self._cross_covariance = cross_covariance

            super().__init__(
                self._call,
                input_dim=self._prior_kernel.input_dim,
                vectorize=True,
            )

        def _call(self, x0, x1):
            k_xx = self._prior_kernel.jax(x0, x1)
            kLadj_xX = self._cross_covariance(x0)
            LkLadj_XX_cho = self._gram_matrix_cholesky
            Lk_Xx = self._cross_covariance(x1)

            return k_xx + kLadj_xX @ jax.scipy.linalg.cho_solve(LkLadj_XX_cho, Lk_Xx)

    def condition_on_observations(self, X, fX):
        k_X_Xprevs = tuple(
            k_cross.jax(X[:, None, :], X_prev[None, :, :])
            for k_cross, X_prev in zip(self._cross_covariances, self._locations)
        )
        kXX = self._prior._covfun.jax(X[:, None, :], X[None, :, :])
        new_gram_row = k_X_Xprevs + (kXX,)

        C = np.hstack(k_X_Xprevs)
        new_representer_weights = _schur_update(
            A_cho=self._gram_matrix_cholesky,
            B=C.T,
            C=C,
            D=kXX,
            A_inv_u=self._representer_weights,
            v=(fX - self._prior._meanfun(X)),
        )

        return PosteriorGaussianProcess(
            self._prior,
            locations=self._locations + (X,),
            measurements=self._measurements + (fX,),
            cross_covariances=self._cross_covariances + (self._prior._covfun,),
            gram_matrices=self._gram_matrices + (new_gram_row,),
            representer_weights=new_representer_weights,
        )

    def apply_jax_linop(self, linop):
        return (
            PosteriorGaussianProcess(
                prior=self._prior.apply_jax_linop(linop)[0],
                locations=self._locations,
                measurements=self._measurements,
                cross_covariances=[
                    _jax.JaxKernel(
                        linop(k_cross.jax, argnum=0),
                        input_dim=self.input_dim,
                        vectorize=True,
                    )
                    for k_cross in self._cross_covariances
                ],
                gram_matrices=self._gram_matrices,
                representer_weights=self._representer_weights,
            ),
            None,
            # TODO
        )


def _schur_update(A_cho, B, C, D, A_inv_u, v):
    """
    This function solves the linear system

    [[A, B], @ [[x], = [[u],
     [C, D]]    [y]]    [v]]

    given the Cholesky factor of A, B, C, D, A^{-1} u, and v.
    """
    A_inv_B = jax.scipy.linalg.cho_solve(A_cho, B)

    S = D - C @ A_inv_B

    y = jax.scipy.linalg.solve(S, v - C @ A_inv_u)
    x = A_inv_u - A_inv_B @ y

    return np.concatenate((x, y))
