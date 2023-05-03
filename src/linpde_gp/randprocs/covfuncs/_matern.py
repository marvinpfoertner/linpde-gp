import functools
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
import probnum as pn

from ._jax import JaxCovarianceFunctionMixin, JaxIsotropicMixin


class Matern(
    JaxCovarianceFunctionMixin, JaxIsotropicMixin, pn.randprocs.covfuncs.Matern
):
    @functools.partial(jax.jit, static_argnums=0)
    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        if self.nu == np.inf:
            return jnp.exp(
                -self._squared_euclidean_distances_jax(  # pylint: disable=invalid-unary-operand-type
                    x0, x1, scale_factors=self._scale_factors
                )
            )

        scaled_distances = self._euclidean_distances_jax(
            x0, x1, scale_factors=self._scale_factors
        )

        if self.is_half_integer:
            # Evaluate the polynomial part using Horner's method
            coeffs = Matern._half_integer_coefficients_floating(self.p)

            res = coeffs[self.p]

            for i in range(self.p - 1, -1, -1):
                res *= scaled_distances
                res += coeffs[i]

            # Exponential part
            res *= jnp.exp(
                -scaled_distances  # pylint: disable=invalid-unary-operand-type
            )

            return res

        # The modified Bessel function of the second kind is not implemented in jax
        raise NotImplementedError()
