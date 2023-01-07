import numpy as np
import jax.numpy as jnp
import probnum as pn

from linpde_gp import linfunctls

from .._pv_crosscov import ProcessVectorCrossCovariance


@linfunctls.FlattenedLinearFunctional.__call__.register(  # pylint: disable=no-member
    ProcessVectorCrossCovariance
)
def _(self, pv_crosscov: ProcessVectorCrossCovariance, /) -> np.ndarray:
    inner_result = self.inner_functional(pv_crosscov)
    inner_output_ndim = self.inner_functional.output_ndim

    inner_shape = inner_result.shape
    # Take the existing shape and flatten the linear functional component
    if not pv_crosscov.reverse:
        # k(x, .)
        target_shape = self.output_shape + inner_shape[inner_output_ndim:]
    else:
        # k(., x)
        target_shape = inner_shape[:-inner_output_ndim] + self.output_shape
    return inner_result.reshape(target_shape, order="C")


class Kernel_Identity_Flattened(ProcessVectorCrossCovariance):
    def __init__(
        self,
        kernel: pn.randprocs.kernels.Kernel,
        flatten: linfunctls.FlattenedLinearFunctional,
    ):
        self._kernel = kernel
        self._flatten = flatten

        randproc_output_shape = self._kernel.output_shape[
            : self._kernel.output_ndim - self._flatten.input_codomain_ndim
        ]

        super().__init__(
            randproc_input_shape=self._kernel.input_shape,
            randproc_output_shape=randproc_output_shape,
            randvar_shape=self._flatten.output_shape,
            reverse=False,
        )

    @property
    def kernel(self) -> pn.randprocs.kernels.Kernel:
        return self._kernel

    @property
    def flatten(self) -> linfunctls.FlattenedLinearFunctional:
        return self._flatten

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        inner_functional = self._flatten.inner_functional
        inner_pv_crosscov = inner_functional(self._kernel, argnum=1)
        assert isinstance(inner_pv_crosscov, ProcessVectorCrossCovariance)
        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_output_shape = self.randproc_output_shape

        output_shape = x_batch_shape + x_output_shape + self._flatten.output_shape
        return inner_pv_crosscov(x).reshape(output_shape, order='C')

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        inner_functional = self._flatten.inner_functional
        inner_pv_crosscov = inner_functional(self._kernel, argnum=1)
        assert isinstance(inner_pv_crosscov, ProcessVectorCrossCovariance)
        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_output_shape = self.randproc_output_shape

        output_shape = x_batch_shape + x_output_shape + self._flatten.output_shape
        return inner_pv_crosscov.jax(x).reshape(output_shape, order='C')

class Kernel_Flattened_Identity(ProcessVectorCrossCovariance):
    def __init__(
        self,
        kernel: pn.randprocs.kernels.Kernel,
        flatten: linfunctls.FlattenedLinearFunctional,
    ):
        self._kernel = kernel
        self._flatten = flatten

        randproc_output_shape = self._kernel.output_shape[
            self._flatten.input_codomain_ndim :
        ]

        super().__init__(
            randproc_input_shape=self._kernel.input_shape,
            randproc_output_shape=randproc_output_shape,
            randvar_shape=self._flatten.output_shape,
            reverse=True,
        )

    @property
    def kernel(self) -> pn.randprocs.kernels.Kernel:
        return self._kernel

    @property
    def flatten(self) -> linfunctls.FlattenedLinearFunctional:
        return self._flatten

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        inner_functional = self._flatten.inner_functional
        inner_pv_crosscov = inner_functional(self._kernel, argnum=0)
        assert isinstance(inner_pv_crosscov, ProcessVectorCrossCovariance)
        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_output_shape = self.randproc_output_shape

        output_shape = inner_functional.output_shape + x_batch_shape + x_output_shape
        return inner_pv_crosscov(x).reshape(output_shape, order='C')

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        inner_functional = self._flatten.inner_functional
        inner_pv_crosscov = inner_functional(self._kernel, argnum=0)
        assert isinstance(inner_pv_crosscov, ProcessVectorCrossCovariance)
        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_output_shape = self.randproc_output_shape

        output_shape = inner_functional.output_shape + x_batch_shape + x_output_shape
        return inner_pv_crosscov.jax(x).reshape(output_shape, order='C')