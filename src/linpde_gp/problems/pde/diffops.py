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
