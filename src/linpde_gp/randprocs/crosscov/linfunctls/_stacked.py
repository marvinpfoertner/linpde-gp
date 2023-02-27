import numpy as np
import jax.numpy as jnp
import probnum as pn

from linpde_gp import linfunctls
from linpde_gp.linops import BlockMatrix

from .._pv_crosscov import ProcessVectorCrossCovariance


@linfunctls.StackedLinearFunctional.__call__.register(  # pylint: disable=no-member
    ProcessVectorCrossCovariance
)
def _(self, pv_crosscov: ProcessVectorCrossCovariance, /) -> np.ndarray:
    res_linfctl_1 = self.linfctl_1(pv_crosscov)
    res_linfctl_2 = self.linfctl_2(pv_crosscov)

    axis = 0
    if pv_crosscov.reverse:
        axis = -1
    
    return np.concatenate((res_linfctl_1, res_linfctl_2), axis=axis)


class CovarianceFunction_Identity_Stacked(ProcessVectorCrossCovariance):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        stacked: linfunctls.StackedLinearFunctional,
    ):
        self._covfunc = covfunc
        self._stacked = stacked

        L1 = self._stacked.linfctl_1
        L2 = self._stacked.linfctl_2

        self._kL1a = L1(self._covfunc, argnum=1)
        self._kL2a = L2(self._covfunc, argnum=1)
        assert isinstance(self._kL1a, ProcessVectorCrossCovariance)
        assert isinstance(self._kL2a, ProcessVectorCrossCovariance)

        super().__init__(
            randproc_input_shape=self._covfunc.input_shape,
            randproc_output_shape=self._covfunc.output_shape_0,
            randvar_shape=self._stacked.output_shape,
            reverse=False,
        )

    @property
    def kernel(self) -> pn.randprocs.kernels.Kernel:
        return self._covfunc

    @property
    def stacked(self) -> linfunctls.StackedLinearFunctional:
        return self._stacked

    @property
    def kL1a(self) -> ProcessVectorCrossCovariance:
        return self._kL1a

    @property
    def kL2a(self) -> ProcessVectorCrossCovariance:
        return self._kL2a

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate((self._kL1a(x), self._kL2a(x)), axis=-1)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate((self._kL1a(x), self._kL2a(x)), axis=-1)

@linfunctls.StackedLinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Identity_Stacked
)
def _(self, kSa: CovarianceFunction_Identity_Stacked, /) -> pn.linops.LinearOperator:
    L1kL1a = pn.linops.Matrix(self.L1(kSa.kL1a))
    L1kL1a.is_symmetric = True
    L1kL1a.is_positive_definite = True
    L1kL2a = pn.linops.Matrix(self.L1(kSa.kL2a))
    L2kL2a = pn.linops.Matrix(self.L2(kSa.kL2a))
    L2kL2a.is_symmetric = True
    L2kL2a.is_positive_definite = True

    return BlockMatrix(L1kL1a, L1kL2a, None, L2kL2a, is_spd=True)

class CovarianceFunction_Stacked_Identity(ProcessVectorCrossCovariance):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        stacked: linfunctls.StackedLinearFunctional,
    ):
        self._covfunc = covfunc
        self._stacked = stacked

        L1 = self._stacked.linfctl_1
        L2 = self._stacked.linfctl_2

        self._L1k = L1(self._covfunc, argnum=0)
        self._L2k = L2(self._covfunc, argnum=0)
        assert isinstance(self._L1k, ProcessVectorCrossCovariance)
        assert isinstance(self._L2k, ProcessVectorCrossCovariance)

        super().__init__(
            randproc_input_shape=self._covfunc.input_shape,
            randproc_output_shape=self._covfunc.output_shape_1,
            randvar_shape=self._stacked.output_shape,
            reverse=True,
        )

    @property
    def covfunc(self) -> pn.randprocs.kernels.Kernel:
        return self._covfunc

    @property
    def stacked(self) -> linfunctls.StackedLinearFunctional:
        return self._stacked

    @property
    def L1k(self) -> ProcessVectorCrossCovariance:
        return self._L1k

    @property
    def L2k(self) -> ProcessVectorCrossCovariance:
        return self._L2k

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate((self._L1k(x), self._L2k(x)), axis=0)

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate((self._L1k(x), self._L2k(x)), axis=0)

@linfunctls.StackedLinearFunctional.__call__.register(  # pylint: disable=no-member
    CovarianceFunction_Stacked_Identity
)
def _(self, Sk: CovarianceFunction_Stacked_Identity, /) -> pn.linops.LinearOperator:
    L1kL1a = pn.linops.Matrix(self.L1(Sk.L1k))
    L1kL1a.is_symmetric = True
    L1kL1a.is_positive_definite = True
    L1kL2a = pn.linops.Matrix(self.L2(Sk.L1k))
    L2kL2a = pn.linops.Matrix(self.L2(Sk.L2k))
    L2kL2a.is_symmetric = True
    L2kL2a.is_positive_definite = True

    return BlockMatrix(L1kL1a, L1kL2a, None, L2kL2a, is_spd=True)