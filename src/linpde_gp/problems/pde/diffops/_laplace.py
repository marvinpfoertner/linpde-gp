from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp

import probnum as pn

from .... import linfuncops

if TYPE_CHECKING:
    import linpde_gp


def scaled_laplace_jax(
    f: linfuncops.JaxFunction, *, argnum: int = 0, alpha: float = 1.0
) -> linfuncops.JaxFunction:
    Hf = jax.jit(jax.hessian(f, argnum))

    @jax.jit
    def _scaled_hessian_trace(*args, **kwargs):
        return alpha * jnp.trace(jnp.atleast_2d(Hf(*args, **kwargs)))

    return _scaled_hessian_trace


class ScaledLaplaceOperator(linfuncops.JaxLinearOperator):
    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha

        super().__init__(L=functools.partial(scaled_laplace_jax, alpha=self._alpha))

    @functools.singledispatchmethod
    def __call__(self, f, **kwargs):
        return super().__call__(f, **kwargs)

    @functools.singledispatchmethod
    def project(self, basis: linpde_gp.bases.Basis) -> pn.linops.LinearOperator:
        raise NotImplementedError()
