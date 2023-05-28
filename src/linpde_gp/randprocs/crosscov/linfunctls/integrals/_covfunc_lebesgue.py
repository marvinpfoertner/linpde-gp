import functools
import types

from jax import numpy as jnp
import numpy as np
import probnum as pn
import scipy.integrate

from linpde_gp import linfunctls

from .._base import LinearFunctionalProcessVectorCrossCovariance


class np_vectorize_method(np.vectorize):
    def __get__(self, obj, objtype=None):
        """https://docs.python.org/3/howto/descriptor.html#functions-and-methods"""
        if obj is None:
            return self
        return types.MethodType(self, obj)


class CovarianceFunction_Identity_LebesgueIntegral(
    LinearFunctionalProcessVectorCrossCovariance
):
    def __init__(
        self,
        covfunc: pn.randprocs.covfuncs.CovarianceFunction,
        integral: linfunctls.LebesgueIntegral,
        reverse: bool = False,
    ):
        if integral.output_shape != ():
            raise NotImplementedError()

        super().__init__(
            covfunc=covfunc,
            linfunctl=integral,
            reverse=reverse,
        )

    @property
    def integral(self) -> linfunctls.LebesgueIntegral:
        return self.linfunctl

    @functools.partial(np_vectorize_method, excluded={0})
    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return scipy.integrate.quad(
            lambda t: self.covfunc(x, t),
            a=self.integral.domain[0],
            b=self.integral.domain[1],
        )[0]

    def _evaluate_jax(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()


@linfunctls.LebesgueIntegral.__call__.register  # pylint: disable=no-member
def _(self, Lk_or_kL: CovarianceFunction_Identity_LebesgueIntegral, /):
    if Lk_or_kL.reverse:  # Lk
        integral0 = Lk_or_kL.integral
        integral1 = self
    else:  # kL'
        integral0 = self
        integral1 = Lk_or_kL.integral

    return scipy.integrate.dblquad(
        lambda x1, x0: Lk_or_kL.covfunc(x0, x1),
        *integral0.domain,
        *integral1.domain,
    )[0]
