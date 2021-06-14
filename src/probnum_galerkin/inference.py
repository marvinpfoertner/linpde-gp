from typing import Union

import numpy as np
import probnum as pn
import scipy.linalg


def linear_gaussian_model(
    prior: pn.randvars.Normal,
    A: Union[np.ndarray, pn.linops.LinearOperatorLike],
    measurement_noise: pn.randvars.Normal,
    measurements: np.ndarray,
) -> pn.randvars.Normal:
    prior_x = prior

    if not isinstance(A, np.ndarray):
        A = pn.linops.aslinop(A)

    prior_pred_cov_yx = A @ prior_x.cov

    prior_pred_Ax = pn.randvars.Normal(
        mean=A @ prior.mean,
        cov=prior_pred_cov_yx @ A.T,
    )
    prior_pred_y = prior_pred_Ax + measurement_noise

    gain = scipy.linalg.cho_solve(
        scipy.linalg.cho_factor(prior_pred_y.dense_cov),
        prior_pred_cov_yx
        if isinstance(prior_pred_cov_yx, np.ndarray)
        else prior_pred_cov_yx.todense(),
    ).T

    return pn.randvars.Normal(
        mean=prior.mean + gain @ (measurements - prior_pred_y.mean),
        cov=prior.cov - gain @ prior_pred_cov_yx,
    )
