from collections.abc import Sequence
from typing import Optional

import jax.numpy as np
import probnum as pn

from . import _jax


class PosteriorGaussianProcess(pn.randprocs.GaussianProcess):
    def __init__(
        self,
        prior: pn.randprocs.RandomProcess,
        representer_weights: Sequence[np.ndarray],
        cross_covariances: Sequence[Optional[pn.kernels.Kernel]],
        gram_matrices: Sequence[Sequence[np.ndarray]],
    ):
        self._prior = prior

        self._representer_weights = representer_weights
        self._cross_covariances = cross_covariances

        self._gram_matrices = gram_matrices

        self._alpha = np.vstack(self._representer_weights)
        self._gram_matrix = np.block(
            [
                [
                    self._gram_matrices[i][j] if i >= j else self._gram_matrices[j][i]
                    for j in range(len(self._representer_weights))
                ]
                for i in range(len(self._representer_weights))
            ]
        )
        self._cross_covariance = PosteriorGaussianProcess.CrossCovariance(
            self._prior.input_dim,
            self._cross_covariances,
        )

        super().__init__(
            mean=PosteriorGaussianProcess.Mean(
                prior_mean=self._prior._meanfun,
                representer_weights=self._alpha,
                cross_covariance=self._cross_covariance,
            ),
            cov=PosteriorGaussianProcess.Kernel(
                prior_kernel=self._prior._covfun,
                gram_matrix_cholesky=None,
                cross_covariance=self._cross_covariance,
            ),
        )

    class Mean(_jax.JaxMean):
        def __init__(
            self,
            prior_mean: _jax.JaxMean,
            representer_weights: np.ndarray,
            cross_covariance: "PosteriorGaussianProcess.CrossCovariance",
        ):
            pass

    class Kernel(_jax.JaxKernel):
        def __init__(
            self,
            prior_kernel: _jax.JaxKernel,
            gram_matrix_cholesky,
            cross_covariance: "PosteriorGaussianProcess.CrossCovariance",
        ):
            pass

    class CrossCovariance(_jax.JaxKernel):
        def __init__(self, input_dim: int, cross_covariances):
            pass
