from jax import numpy as jnp
import numpy as np
import probnum as pn

from linpde_gp import linfuncops, linfunctls

from .._pv_crosscov import ProcessVectorCrossCovariance


class Kernel_Identity_Dirac(ProcessVectorCrossCovariance):
    def __init__(
        self,
        kernel: pn.randprocs.kernels.Kernel,
        dirac: linfunctls.DiracFunctional,
    ):
        self._kernel = kernel
        self._dirac = dirac

        randproc_output_shape = self._kernel.output_shape[
            : self._kernel.output_ndim - self._dirac.input_codomain_ndim
        ]

        super().__init__(
            randproc_input_shape=self._kernel.input_shape,
            randproc_output_shape=randproc_output_shape,
            randvar_shape=self._dirac.output_shape,
            reverse=False,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        # `kxX.shape` layout:
        # x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = 0

        X_batch_shape = self._dirac.X_batch_shape
        X_batch_ndim = self._dirac.X_batch_ndim
        X_batch_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._dirac.input_codomain_shape
        X_output_ndim = self._dirac.input_codomain_ndim
        X_output_offset = x_output_offset + x_output_ndim

        assert x.shape == x_batch_shape + self._kernel.input_shape

        x = np.expand_dims(
            x,
            axis=tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim)),
        )

        assert x.shape == x_batch_shape + X_batch_ndim * (1,) + self._kernel.input_shape

        X = self._dirac.X

        assert X.shape == X_batch_shape + self._kernel.input_shape

        kxX = self._kernel(x, X)

        assert kxX.shape == (
            x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        )

        # Transform to desired output layout
        # x_batch_shape + x_output_shape + X_batch_shape + X_output_shape
        kxX = kxX.transpose(
            tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
            + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
            + tuple(range(X_output_offset, X_output_offset + X_output_ndim))
        )

        assert kxX.shape == (
            x_batch_shape + x_output_shape + X_batch_shape + X_output_shape
        )

        return kxX

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        # `kxX.shape` layout:
        # x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = 0

        X_batch_shape = self._dirac.X_batch_shape
        X_batch_ndim = self._dirac.X_batch_ndim
        X_batch_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._dirac.input_codomain_shape
        X_output_ndim = self._dirac.input_codomain_ndim
        X_output_offset = x_output_offset + x_output_ndim

        assert x.shape == x_batch_shape + self._kernel.input_shape

        x = jnp.expand_dims(
            x,
            axis=tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim)),
        )

        assert x.shape == x_batch_shape + X_batch_ndim * (1,) + self._kernel.input_shape

        X = self._dirac.X

        assert X.shape == X_batch_shape + self._kernel.input_shape

        kxX = self._kernel(x, X)

        assert kxX.shape == (
            x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        )

        # Transform to desired output layout
        # x_batch_shape + x_output_shape + X_batch_shape + X_output_shape
        kxX = kxX.transpose(
            tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
            + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
            + tuple(range(X_output_offset, X_output_offset + X_output_ndim))
        )

        assert kxX.shape == (
            x_batch_shape + x_output_shape + X_batch_shape + X_output_shape
        )

        return kxX


class Kernel_Dirac_Indentity(ProcessVectorCrossCovariance):
    def __init__(
        self,
        kernel: pn.randprocs.kernels.Kernel,
        dirac: linfunctls.DiracFunctional,
    ):
        self._kernel = kernel
        self._dirac = dirac

        randproc_output_shape = self._kernel.output_shape[
            self._dirac.input_codomain_ndim :
        ]

        super().__init__(
            randproc_input_shape=self._kernel.input_shape,
            randproc_output_shape=randproc_output_shape,
            randvar_shape=self._dirac.output_shape,
            reverse=False,
        )

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        # `kXx.shape` layout:
        # X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        X_batch_shape = self._dirac.X_batch_shape
        X_batch_ndim = self._dirac.X_batch_ndim
        X_batch_offset = 0

        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._dirac.input_codomain_shape
        X_output_ndim = self._dirac.input_codomain_ndim
        X_output_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_output_offset + X_output_ndim

        assert x.shape == x_batch_shape + self._kernel.input_shape

        X = np.expand_dims(
            self._dirac.X,
            axis=tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim)),
        )

        assert X.shape == (
            self._dirac.X_batch_shape + x_batch_ndim * (1,) + self._kernel.input_shape
        )

        kxX = self._kernel(x, X)

        assert kxX.shape == (
            X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        )

        # Transform to desired output layout
        # X_batch_shape + X_output_shape + x_batch_shape + x_output_shape
        kxX = kxX.transpose(
            tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
            + tuple(range(X_output_offset, X_output_offset + X_output_ndim))
            + tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
        )

        assert kxX.shape == (
            X_batch_shape + X_output_shape + x_batch_shape + x_output_shape
        )

        return kxX

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        # `kXx.shape` layout:
        # X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        X_batch_shape = self._dirac.X_batch_shape
        X_batch_ndim = self._dirac.X_batch_ndim
        X_batch_offset = 0

        x_batch_shape = x.shape[: x.ndim - self._kernel.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._dirac.input_codomain_shape
        X_output_ndim = self._dirac.input_codomain_ndim
        X_output_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_output_offset + X_output_ndim

        assert x.shape == x_batch_shape + self._kernel.input_shape

        X = jnp.expand_dims(
            self._dirac.X,
            axis=tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim)),
        )

        assert X.shape == (
            self._dirac.X_batch_shape + x_batch_ndim * (1,) + self._kernel.input_shape
        )

        kxX = self._kernel(x, X)

        assert kxX.shape == (
            X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        )

        # Transform to desired output layout
        # X_batch_shape + X_output_shape + x_batch_shape + x_output_shape
        kxX = kxX.transpose(
            tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
            + tuple(range(X_output_offset, X_output_offset + X_output_ndim))
            + tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
        )

        assert kxX.shape == (
            X_batch_shape + X_output_shape + x_batch_shape + x_output_shape
        )

        return kxX


@linfunctls.DiracFunctional.__call__.register(
    Kernel_Identity_Dirac
)  # pylint: disable=no-member
def _(self, k_id_dirac: Kernel_Identity_Dirac, /, argnum: int = 0) -> np.ndarray:
    if argnum != 0:
        raise ValueError("TODO")

    return k_id_dirac(self.X)


@linfunctls.DiracFunctional.__call__.register  # pylint: disable=no-member
def _(self, k_dirac_id: Kernel_Dirac_Indentity, /, argnum: int = 0) -> np.ndarray:
    if argnum != 1:
        raise ValueError("TODO")

    return k_dirac_id(self.X)


@linfuncops.LinearFunctionOperator.__call__.register(Kernel_Identity_Dirac)
def _(self, k_id_dirac: Kernel_Identity_Dirac, /, argnum: int = 0) -> np.ndarray:
    if argnum != 0:
        raise ValueError("TODO")

    return Kernel_Identity_Dirac(
        self(k_id_dirac._kernel, argnum=0),
        k_id_dirac._dirac,
    )
