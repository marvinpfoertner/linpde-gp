import functools

from jax import numpy as jnp
import numpy as np
from probnum.typing import ArrayLike

from .. import domains
from ._jax import JaxFunction


class TruncatedSineSeries(JaxFunction):
    def __init__(
        self,
        domain: domains.Interval,
        coefficients: ArrayLike,
    ) -> None:
        domain = domains.asdomain(domain)

        if not isinstance(domain, domains.Interval):
            raise TypeError("`domain` must be an `Interval`")

        self._domain = domain

        super().__init__(input_shape=self._domain.shape, output_shape=())

        coefficients = np.asarray(coefficients)

        if coefficients.ndim != 1:
            raise ValueError()

        self._coefficients = coefficients

    @property
    def domain(self) -> domains.Interval:
        return self._domain

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @functools.cached_property
    def half_angular_frequencies(self) -> np.ndarray:
        l, r = self._domain

        return np.pi * np.arange(1, self._coefficients.shape[-1] + 1) / (r - l)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        l, _ = self._domain

        return np.sum(
            self._coefficients
            * np.sin(self.half_angular_frequencies * (x[..., None] - l)),
            axis=-1,
        )

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        l, _ = self._domain

        return jnp.sum(
            self._coefficients
            * jnp.sin(self.half_angular_frequencies * (x[..., None] - l)),
            axis=-1,
        )
