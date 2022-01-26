from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import numpy as np
import probnum as pn
from jax import numpy as jnp

from .... import linfuncops
from ....typing import JaxFunction

if TYPE_CHECKING:
    import linpde_gp


def laplace_jax(f: JaxFunction, argnum: int = 0) -> JaxFunction:
    Hf = jax.jit(jax.hessian(f, argnum))

    @jax.jit
    def _hessian_trace(*args, **kwargs):
        return jnp.trace(jnp.atleast_2d(Hf(*args, **kwargs)))

    return _hessian_trace


class LaplaceOperator(linfuncops.JaxLinearOperator):
    def __init__(self) -> None:
        super().__init__(L=laplace_jax)

    @functools.singledispatchmethod
    def __call__(self, f, *, argnum=0, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def project(self, basis: linpde_gp.bases.Basis) -> pn.linops.LinearOperator:
        raise NotImplementedError()
