from typing import Optional

from jax import numpy as jnp
import numpy as np
import probnum as pn

from ._jax import JaxCovarianceFunction


class IndependentMultiOutputCovarianceFunction(JaxCovarianceFunction):
    def __init__(self, *covfuncs: JaxCovarianceFunction):
        assert len(covfuncs) > 0
        input_shape = None
        for covfunc in covfuncs:
            if input_shape is None:
                input_shape = covfunc.input_shape
            assert covfunc.input_shape == input_shape
            assert covfunc.output_shape_0 == ()
            assert covfunc.output_shape_1 == ()
        self._covfuncs = covfuncs
        super().__init__(
            input_shape=input_shape,
            output_shape_0=(len(covfuncs),),
            output_shape_1=(len(covfuncs),),
        )

    @property
    def covfuncs(self):
        return self._covfuncs

    def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        result = np.zeros(
            broadcast_batch_shape + self._output_shape_0 + self._output_shape_1
        )
        (n_outputs,) = self.output_shape_0

        for cur_output in range(n_outputs):
            result[..., cur_output, cur_output] = self.covfuncs[cur_output](x0, x1)

        return result

    def _evaluate_jax(self, x0: jnp.ndarray, x1: Optional[jnp.ndarray]) -> jnp.ndarray:
        # Shape checking
        broadcast_batch_shape = self._check_shapes(
            x0.shape, x1.shape if x1 is not None else None
        )

        result = jnp.zeros(
            broadcast_batch_shape + self._output_shape_0 + self._output_shape_1
        )
        (n_outputs,) = self.output_shape_0

        for cur_output in range(n_outputs):
            result = result.at[..., cur_output, cur_output].set(
                self.covfuncs[cur_output].jax(x0, x1)
            )

        return result

    def _evaluate_linop(
        self, x0: np.ndarray, x1: Optional[np.ndarray]
    ) -> pn.linops.LinearOperator:
        return pn.linops.BlockDiagonalMatrix(
            *(covfunc.linop(x0, x1) for covfunc in self.covfuncs)
        )
