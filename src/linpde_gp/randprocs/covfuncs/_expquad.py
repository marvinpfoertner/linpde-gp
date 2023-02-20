import functools
from typing import Optional

import jax
from jax import numpy as jnp
import probnum as pn

from ._jax import JaxCovarianceFunctionMixin, JaxIsotropicMixin


class ExpQuad(
    JaxCovarianceFunctionMixin, JaxIsotropicMixin, pn.randprocs.covfuncs.ExpQuad
):
    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if x1 is None:
            return jnp.ones_like(
                x0,
                shape=x0.shape[: x0.ndim - self.input_ndim],
            )

        return jnp.exp(
            -self._squared_euclidean_distances_jax(
                x0, x1, scale_factors=self._scale_factors
            )
        )
