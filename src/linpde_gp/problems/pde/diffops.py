import functools

import jax
import linpde_gp
import probnum as pn
from jax import numpy as jnp

from ...typing import JaxFunction


def laplace(f: JaxFunction, argnum: int = 0) -> JaxFunction:
    Hf = jax.jit(jax.hessian(f, argnum))

    @jax.jit
    def _hessian_trace(*args, **kwargs):
        return jnp.trace(jnp.atleast_2d(Hf(*args, **kwargs)))

    return _hessian_trace


class LaplaceOperator:
    def __call__(self, f: JaxFunction, argnum: int = 0) -> JaxFunction:
        return laplace(f, argnum=argnum)

    @functools.singledispatchmethod
    def project(self, basis: "linpde_gp.bases.Basis") -> pn.linops.LinearOperator:
        raise NotImplementedError()


class HeatOperator:
    def __call__(self, f: JaxFunction, argnum: int = 0) -> JaxFunction:
        @jax.jit
        def _f(*args, **kwargs) -> jnp.ndarray:
            t, x = args[argnum : argnum + 2]
            tx = jnp.concatenate((t[None], x), axis=0)

            return f(
                *args[:argnum],
                tx,
                *args[argnum + 2 :],
                **kwargs,
            )

        @jax.jit
        def _f_heat(*args, **kwargs) -> jnp.ndarray:
            tx = args[argnum]
            t, x = tx[0], tx[1:]
            args = args[:argnum] + (t, x) + args[argnum + 1 :]

            df_dt = jax.grad(_f, argnums=argnum)(*args, **kwargs)
            laplace_f = laplace(_f, argnum=argnum + 1)(*args, **kwargs)

            return df_dt - laplace_f

        return _f_heat

    @functools.singledispatchmethod
    def project(self, basis: "linpde_gp.bases.Basis") -> pn.linops.LinearOperator:
        raise NotImplementedError()
