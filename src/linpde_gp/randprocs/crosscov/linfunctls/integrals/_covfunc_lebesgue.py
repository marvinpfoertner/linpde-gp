import functools
import types

from jax import numpy as jnp
import numpy as np
import probnum as pn
import scipy.integrate

from linpde_gp import linfunctls

from ... import _pv_crosscov


class np_vectorize_method(np.vectorize):
    def __get__(self, obj, objtype=None):
        """https://docs.python.org/3/howto/descriptor.html#functions-and-methods"""
        if obj is None:
            return self
        return types.MethodType(self, obj)


class CovarianceFunction_Identity_LebesgueIntegral(
    _pv_crosscov.ProcessVectorCrossCovariance
):
    def __init__(
        self,
        k: pn.randprocs.covfuncs.CovarianceFunction,
        L: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        randproc_output_shape = k.output_shape_1 if reverse else k.output_shape_0

        if randproc_output_shape != ():
            raise NotImplementedError()

        super().__init__(
            randproc_input_shape=k.input_shape_1 if reverse else k.input_shape_0,
            randproc_output_shape=randproc_output_shape,
            randvar_shape=(),
            reverse=reverse,
        )

        self._k = k
        self._L = L

    @functools.partial(np_vectorize_method, excluded={0})
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return scipy.integrate.quad(
            lambda t: self._k(x, t),
            a=self._L.domain[0],
            b=self._L.domain[1],
        )[0]

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, pv_crosscov: CovarianceFunction_Identity_LebesgueIntegral, /):
    if pv_crosscov.reverse:
        L0 = pv_crosscov._L
        L1 = self
    else:
        L0 = self
        L1 = pv_crosscov._L

    return scipy.integrate.dblquad(
        pv_crosscov._k,
        *L0.domain,
        *L1.domain,
    )[0]
