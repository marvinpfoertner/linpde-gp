from jax import numpy as jnp
import numpy as np
import probnum as pn

from linpde_gp import linfunctls

from .._pv_crosscov import ProcessVectorCrossCovariance


@linfunctls._EvaluationFunctional.__call__.register(  # pylint: disable=no-member
    ProcessVectorCrossCovariance
)
def _(self, pv_crosscov: ProcessVectorCrossCovariance, /) -> pn.linops.LinearOperator:
    return pv_crosscov.evaluate_linop(self._X)


class CovarianceFunction_Identity_Evaluation(ProcessVectorCrossCovariance):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        evaluation_fctl: linfunctls._EvaluationFunctional,
    ):
        self._covfunc = covfunc
        self._evaluation_fctl = evaluation_fctl

        super().__init__(
            randproc_input_shape=self._covfunc.input_shape,
            randproc_output_shape=self._covfunc.output_shape_0,
            randvar_shape=self._evaluation_fctl.output_shape,
            reverse=False,
        )

    @property
    def covfunc(self) -> pn.randprocs.covfuncs.CovarianceFunction:
        return self._covfunc

    @property
    def evaluation_fctl(self) -> linfunctls._EvaluationFunctional:
        return self._evaluation_fctl

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        # `kxX.shape` layout:
        # x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        x_batch_shape = x.shape[: x.ndim - self._covfunc.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = 0

        X_batch_shape = self._evaluation_fctl.X_batch_shape
        X_batch_ndim = self._evaluation_fctl.X_batch_ndim
        X_batch_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._evaluation_fctl.input_codomain_shape
        X_output_ndim = self._evaluation_fctl.input_codomain_ndim
        X_output_offset = x_output_offset + x_output_ndim

        assert x.shape == x_batch_shape + self._covfunc.input_shape

        x = np.expand_dims(
            x,
            axis=tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim)),
        )

        assert (
            x.shape == x_batch_shape + X_batch_ndim * (1,) + self._covfunc.input_shape
        )

        X = self._evaluation_fctl.X

        assert X.shape == X_batch_shape + self._covfunc.input_shape

        kxX = self._covfunc(x, X)

        assert kxX.shape == (
            x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        )

        # Transform to desired output layout
        # Crucially, in contrast to the DiracFunctional, here the output shape
        # of the X component comes first
        # x_batch_shape + x_output_shape + X_output_shape + X_batch_shape
        kxX = kxX.transpose(
            tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
            + tuple(range(X_output_offset, X_output_offset + X_output_ndim))
            + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
        )

        assert kxX.shape == (
            x_batch_shape + x_output_shape + X_output_shape + X_batch_shape
        )

        return kxX

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        # `kxX.shape` layout:
        # x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        x_batch_shape = x.shape[: x.ndim - self._covfunc.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = 0

        X_batch_shape = self._evaluation_fctl.X_batch_shape
        X_batch_ndim = self._evaluation_fctl.X_batch_ndim
        X_batch_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._evaluation_fctl.input_codomain_shape
        X_output_ndim = self._evaluation_fctl.input_codomain_ndim
        X_output_offset = x_output_offset + x_output_ndim

        assert x.shape == x_batch_shape + self._covfunc.input_shape

        x = jnp.expand_dims(
            x,
            axis=tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim)),
        )

        assert (
            x.shape == x_batch_shape + X_batch_ndim * (1,) + self._covfunc.input_shape
        )

        X = self._evaluation_fctl.X

        assert X.shape == X_batch_shape + self._covfunc.input_shape

        kxX = self._covfunc(x, X)

        assert kxX.shape == (
            x_batch_shape + X_batch_shape + x_output_shape + X_output_shape
        )

        # Transform to desired output layout
        # Crucially, in contrast to the DiracFunctional, here the output shape
        # of the X component comes first
        # x_batch_shape + x_output_shape + X_output_shape + X_batch_shape
        kxX = kxX.transpose(
            tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
            + tuple(range(X_output_offset, X_output_offset + X_output_ndim))
            + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
        )

        assert kxX.shape == (
            x_batch_shape + x_output_shape + X_output_shape + X_batch_shape
        )

        return kxX

    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        return self.covfunc.linop(x, self._evaluation_fctl.X)


@linfunctls._EvaluationFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Evaluation
)
def _(self, pv_crosscov: CovarianceFunction_Identity_Evaluation, /) -> pn.linops.LinearOperator:
    return pv_crosscov.evaluate_linop(self.X)


class CovarianceFunction_Evaluation_Identity(ProcessVectorCrossCovariance):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        evaluation_fctl: linfunctls._EvaluationFunctional,
    ):
        self._covfunc = covfunc
        self._evaluation_fctl = evaluation_fctl

        randproc_output_shape = self._covfunc.output_shape[
            self._evaluation_fctl.input_codomain_ndim :
        ]

        super().__init__(
            randproc_input_shape=self._covfunc.input_shape,
            randproc_output_shape=randproc_output_shape,
            randvar_shape=self._evaluation_fctl.output_shape,
            reverse=True,
        )

    @property
    def covfunc(self) -> pn.randprocs.covfuncs.CovarianceFunction:
        return self._covfunc

    @property
    def evaluation_fctl(self) -> linfunctls._EvaluationFunctional:
        return self._evaluation_fctl

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        # `kXx.shape` layout:
        # X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        X_batch_shape = self._evaluation_fctl.X_batch_shape
        X_batch_ndim = self._evaluation_fctl.X_batch_ndim
        X_batch_offset = 0

        x_batch_shape = x.shape[: x.ndim - self._covfunc.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._evaluation_fctl.input_codomain_shape
        X_output_ndim = self._evaluation_fctl.input_codomain_ndim
        X_output_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_output_offset + X_output_ndim

        assert x.shape == x_batch_shape + self._covfunc.input_shape

        X = np.expand_dims(
            self._evaluation_fctl.X,
            axis=tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim)),
        )

        assert X.shape == (
            self._evaluation_fctl.X_batch_shape + x_batch_ndim * (1,) + self._covfunc.input_shape
        )

        kxX = self._covfunc(x, X)

        assert kxX.shape == (
            X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        )

        # Transform to desired output layout
        # Crucially, in contrast to the DiracFunctional, here the output shape
        # of the X component comes first
        # X_output_shape + X_batch_shape + x_batch_shape + x_output_shape
        kxX = kxX.transpose(
            tuple(range(X_output_offset, X_output_offset + X_output_ndim))
            + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
            + tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
        )

        assert kxX.shape == (
            X_output_shape + X_batch_shape + x_batch_shape + x_output_shape
        )

        return kxX

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        # `kXx.shape` layout:
        # X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        X_batch_shape = self._evaluation_fctl.X_batch_shape
        X_batch_ndim = self._evaluation_fctl.X_batch_ndim
        X_batch_offset = 0

        x_batch_shape = x.shape[: x.ndim - self._covfunc.input_ndim]
        x_batch_ndim = len(x_batch_shape)
        x_batch_offset = X_batch_offset + X_batch_ndim

        X_output_shape = self._evaluation_fctl.input_codomain_shape
        X_output_ndim = self._evaluation_fctl.input_codomain_ndim
        X_output_offset = x_batch_offset + x_batch_ndim

        x_output_shape = self.randproc_output_shape
        x_output_ndim = self.randproc_output_ndim
        x_output_offset = X_output_offset + X_output_ndim

        assert x.shape == x_batch_shape + self._covfunc.input_shape

        X = jnp.expand_dims(
            self._evaluation_fctl.X,
            axis=tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim)),
        )

        assert X.shape == (
            self._evaluation_fctl.X_batch_shape + x_batch_ndim * (1,) + self._covfunc.input_shape
        )

        kxX = self._covfunc(x, X)

        assert kxX.shape == (
            X_batch_shape + x_batch_shape + X_output_shape + x_output_shape
        )

        # Transform to desired output layout
        # Crucially, in contrast to the DiracFunctional, here the output shape
        # of the X component comes first
        # X_output_shape + X_batch_shape + x_batch_shape + x_output_shape
        kxX = kxX.transpose(
            tuple(range(X_output_offset, X_output_offset + X_output_ndim))
            + tuple(range(X_batch_offset, X_batch_offset + X_batch_ndim))
            + tuple(range(x_batch_offset, x_batch_offset + x_batch_ndim))
            + tuple(range(x_output_offset, x_output_offset + x_output_ndim))
        )

        assert kxX.shape == (
            X_output_shape + X_batch_shape + x_batch_shape + x_output_shape
        )

        return kxX
    
    def _evaluate_linop(self, x: np.ndarray) -> pn.linops.LinearOperator:
        return self.covfunc.linop(self._evaluation_fctl.X, x)


@linfunctls._EvaluationFunctional.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: CovarianceFunction_Evaluation_Identity, /) -> pn.linops.LinearOperator:
    return pv_crosscov.evaluate_linop(self.X)
