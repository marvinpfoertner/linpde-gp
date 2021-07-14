import functools
from typing import Union

import numpy as np
import probnum as pn
import scipy.linalg


class LinearGaussianModel:
    r"""Model for linear observations of a Gaussian random variable under Gaussian noise.

    To be precise, we posit a prior :math:`x \sim \mathcal{N}(\mu_0, \Sigma_0)` and we
    observe realizations of :math:`y := Ax + \epsilon`, where the noise :math:`\epsilon`
    is also assumed to follow a Gaussian distribution, i.e.
    :math:`\epsilon \sim \mathcal{N}(b, \Lambda)`.

    Instances of this class give access to quantities inferred under this model.
    """

    def __init__(
        self,
        prior_x: pn.randvars.Normal,
        A: pn.linops.LinearOperatorLike,
        measurement_noise: pn.randvars.Normal,
    ) -> None:
        self._prior_x = prior_x
        self._A = pn.linops.aslinop(A)
        self._measurement_noise = measurement_noise

    @functools.cached_property
    def prior_pred_Ax(self) -> pn.randvars.Normal:
        r"""Random variable :math:`Ax`."""
        return pn.randvars.Normal(
            mean=self._A @ self._prior_x.mean,
            cov=self._cross_cov_Ax_x @ self._A.T,
        )

    @functools.cached_property
    def prior_pred_y(self) -> pn.randvars.Normal:
        r"""Random variable :math:`y := Ax + \epsilon`."""
        return self.prior_pred_Ax + self._measurement_noise

    def posterior_x(self, y_measurements: np.ndarray) -> pn.randvars.Normal:
        r"""Random variable :math:`x \mid y = y_\text{meas}`."""
        return pn.randvars.Normal(
            mean=(
                self._prior_x.mean
                + self._gain @ (y_measurements - self.prior_pred_y.mean)
            ),
            cov=self._prior_x.cov - self._gain @ self._cross_cov_Ax_x,
        )

    @functools.cached_property
    def _cross_cov_Ax_x(self) -> Union[np.ndarray, pn.linops.LinearOperator]:
        return self._A @ self._prior_x.cov

    @functools.cached_property
    def _gain(self) -> np.ndarray:
        return scipy.linalg.cho_solve(
            scipy.linalg.cho_factor(self.prior_pred_y.dense_cov),
            self._cross_cov_Ax_x
            if isinstance(self._cross_cov_Ax_x, np.ndarray)
            else self._cross_cov_Ax_x.todense(),
        ).T
