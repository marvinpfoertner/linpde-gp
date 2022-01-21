from audioop import cross
from collections.abc import Sequence

import jax
import jax.numpy as np
import probnum as pn
import scipy.linalg
from xxlimited import new

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

            return k_xx - kLadj_xX @ jax.scipy.linalg.cho_solve(LkLadj_XX_cho, Lk_Xx)

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
        linop_prior, crosscov_prior = self._prior.apply_jax_linop(linop)

        linop_gp = PosteriorGaussianProcess(
            prior=linop_prior,
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
        )

        @jax.jit
        def _crosscov(x0, x1):
            k_x0_X = self._cross_covariance(x0)
            k_L_adj_X_x1 = linop_gp._cross_covariance(x1).T
            crosscov = crosscov_prior.jax(x0, x1)
            crosscov -= k_x0_X @ jax.scipy.linalg.cho_solve(
                self._gram_matrix_cholesky, k_L_adj_X_x1
            )

            return crosscov

        crosscov = _jax.JaxKernel(_crosscov, input_dim=self._input_dim, vectorize=True)
        crosscov.prior_crosscov = crosscov_prior

        return (linop_gp, crosscov)

    def condition_on_jax_linop_observations(self, linop, X, LfX):
        predictive_gp, crosscov_predictive = self.apply_jax_linop(linop)
        kLa = crosscov_predictive.prior_crosscov

        # Generate the new row in the Gram matrix
        new_gram_row = [
            Lk_cross.jax(X[:, None, :], X_prev[None, :, :])
            for Lk_cross, X_prev in zip(
                predictive_gp._cross_covariances, predictive_gp._locations
            )
        ]
        new_gram_row.append(
            predictive_gp._prior._covfun.jax(X[:, None, :], X[None, :, :])
        )
        new_gram_row = tuple(new_gram_row)

        C = np.hstack(new_gram_row[:-1])
        new_representer_weights = _schur_update(
            A_cho=self._gram_matrix_cholesky,
            B=C.T,
            C=C,
            D=new_gram_row[-1],
            A_inv_u=self._representer_weights,
            v=(LfX - predictive_gp._prior._meanfun(X)),
        )

        return PosteriorGaussianProcess(
            self._prior,
            locations=self._locations + (X,),
            measurements=self._measurements + (LfX,),
            cross_covariances=self._cross_covariances + (kLa,),
            gram_matrices=self._gram_matrices + (new_gram_row,),
            representer_weights=new_representer_weights,
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

    return np.concatenate((x, y))
