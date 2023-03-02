from jax import numpy as jnp
import numpy as np
from probnum.randprocs import covfuncs

from ._jax import JaxCovarianceFunctionMixin


class Zero(JaxCovarianceFunctionMixin, covfuncs.CovarianceFunction):
    def _evaluate(self, x0: np.ndarray, x1: np.ndarray | None) -> np.ndarray:
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        return np.zeros(
            broadcast_batch_shape + self.output_shape_0 + self.output_shape_1,
            dtype=np.result_type(x0, x1),
        )

    def _evaluate_jax(self, x0: jnp.ndarray, x1: jnp.ndarray | None) -> jnp.ndarray:
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        return jnp.zeros(
            broadcast_batch_shape + self.output_shape_0 + self.output_shape_1,
            dtype=np.result_type(x0, x1),
        )
