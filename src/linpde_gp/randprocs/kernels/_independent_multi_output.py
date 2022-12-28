import functools
from typing import Optional, Sequence

import jax
from jax import numpy as jnp
import numpy as np

from ._jax import JaxKernel

class IndependentMultiOutputKernel(JaxKernel):
    def __init__(self, *kernels: JaxKernel):
        assert len(kernels) > 0
        input_shape = None
        for kernel in kernels:
            if input_shape is None:
                input_shape = kernel.input_shape
            assert kernel.input_shape == input_shape
            assert kernel.output_shape == ()
        self._kernels = kernels
        super().__init__(input_shape, (len(kernels), len(kernels)))
    
    @property
    def kernels(self):
        return self._kernels
    
    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        result = np.zeros(broadcast_batch_shape + self._output_shape)
        n_outputs = self.output_shape[0]

        for cur_output in range(n_outputs):
            result[..., cur_output, cur_output] = self.kernels[cur_output](x0, x1)

        return result

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        result = jnp.zeros(broadcast_batch_shape + self._output_shape)
        n_outputs = self.output_shape[0]

        for cur_output in range(n_outputs):
            result = result.at[..., cur_output, cur_output].set(self.kernels[cur_output](x0, x1))

        return result