import functools

import jax
import linpde_gp
import probnum as pn
from jax import numpy as jnp

from ... import linfuncops
from ...typing import JaxFunction


def laplace(f: JaxFunction, argnum: int = 0) -> JaxFunction:
    Hf = jax.jit(jax.hessian(f, argnum))

    @jax.jit
    def _hessian_trace(*args, **kwargs):
        return jnp.trace(jnp.atleast_2d(Hf(*args, **kwargs)))

    return _hessian_trace


class LaplaceOperator(linfuncops.JaxLinearOperator):
    def __init__(self) -> None:
        super().__init__(L=laplace)

    @functools.singledispatchmethod
    def __call__(self, f, *, argnum=0, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def project(self, basis: "linpde_gp.bases.Basis") -> pn.linops.LinearOperator:
        raise NotImplementedError()


def heat(f: JaxFunction, argnum: int = 0) -> JaxFunction:
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

    df_dt = jax.grad(_f, argnums=argnum)
    laplace_f = laplace(_f, argnum=argnum + 1)

    @jax.jit
    def _f_heat(*args, **kwargs) -> jnp.ndarray:
        tx = args[argnum]
        t, x = tx[0], tx[1:]
        args = args[:argnum] + (t, x) + args[argnum + 1 :]

        return df_dt(*args, **kwargs) - laplace_f(*args, **kwargs)

    return _f_heat


class HeatOperator(linfuncops.JaxLinearOperator):
    def __init__(self) -> None:
        super().__init__(L=heat)

    @functools.singledispatchmethod
    def project(self, basis: "linpde_gp.bases.Basis") -> pn.linops.LinearOperator:
        raise NotImplementedError()


class DirectionalDerivative(linfuncops.JaxLinearOperator):
    def __init__(self, direction):
        self._direction = jnp.asarray(direction)

        super().__init__(L=self._impl)

    def _impl(self, f: JaxFunction, argnum: int = 0) -> JaxFunction:
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

        f_spatial_grad = jax.grad(_f, argnums=argnum + 1)

        @jax.jit
        def _f_dir_deriv(*args, **kwargs) -> jnp.ndarray:
            tx = args[argnum]
            t, x = tx[0], tx[1:]
            args = args[:argnum] + (t, x) + args[argnum + 1 :]

            return jnp.sum(self._direction * f_spatial_grad(*args, **kwargs))

        return _f_dir_deriv
