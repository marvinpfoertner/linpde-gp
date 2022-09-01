import functools

import numpy as np
import probnum as pn
from probnum.typing import ArrayLike
import scipy.stats

from linpde_gp import domains


class TruncatedGaussianMixturePDF(pn.functions.Function):
    def __init__(
        self,
        domain: domains.Interval,
        means: ArrayLike,
        stds: ArrayLike,
    ) -> None:
        self._domain = domains.asdomain(domain)
        self._means = np.asarray(means)
        self._stds = np.asarray(stds)

        if not isinstance(self._domain, domains.Interval):
            raise TypeError()

        super().__init__(
            input_shape=self._domain.shape,
            output_shape=(),
        )

    def _gaussian_pdfs(self, x: np.ndarray) -> np.ndarray:
        return (
            1.0
            / (np.sqrt(2 * np.pi) * self._stds)
            * np.exp(-0.5 * ((x[..., None] - self._means) / self._stds) ** 2)
        )

    @functools.cached_property
    def _Zs(self) -> np.ndarray:
        Zs = scipy.stats.norm.cdf((self._domain[1] - self._means) / self._stds)
        Zs -= scipy.stats.norm.cdf((self._domain[0] - self._means) / self._stds)
        return Zs

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.mean(self._gaussian_pdfs(x) / self._Zs, axis=-1)
