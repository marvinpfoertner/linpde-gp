from typing import Optional

import numpy as np
import probnum as pn
import scipy.linalg


def condition_normal_on_observations(
    prior: pn.randvars.Normal,
    observations: np.ndarray,
    noise: Optional[pn.randvars.Normal],
    transform: Optional[pn.linops.LinearOperatorLike] = None,
) -> pn.randvars.Normal:
    r"""Conditions a Gaussian random variable on linearly transformed observations under
    additive Gaussian noise.

    To be precise, we observe linearly transformed realizations :math:`y := A x +
    \epsilon` of the prior :math:`x \sim \mathcal{N}(\mu_0, \Sigma_0)`, where the noise
    :math:`\epsilon` follows a Gaussian distribution, i.e. :math:`\epsilon \sim
    \mathcal{N}(b, \Lambda)`.
    """

    A = transform

    if A is not None:
        A = pn.linops.aslinop(A)

    # Compute predictive
    pred = prior

    if A is not None:
        # pred = A @ pred
        pred = pn.randvars.Normal(
            mean=A @ pred.mean,
            cov=(A @ pred.cov) @ A.T,
        )

    if noise is not None:
        pred = pred + noise

    # Compute cross-covariance
    crosscov_pred_prior = prior.cov if A is None else A @ prior.cov

    # Factorize Gram matrix
    gram_cho = pred.cov_cholesky

    if isinstance(gram_cho, pn.linops.LinearOperator):
        gram_cho = gram_cho.todense()

    # Compute gain
    gain = scipy.linalg.cho_solve(
        (gram_cho, True),
        crosscov_pred_prior.todense()
        if isinstance(crosscov_pred_prior, pn.linops.LinearOperator)
        else crosscov_pred_prior,
    ).T

    # Construct posterior distribution
    return pn.randvars.Normal(
        mean=prior.mean + gain @ (observations - pred.mean),
        cov=prior.cov - crosscov_pred_prior.T @ gain.T,
    )


pn.randvars.Normal.condition_on_observations = condition_normal_on_observations
