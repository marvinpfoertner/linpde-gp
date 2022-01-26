from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import numpy as np
import probnum as pn
from jax import numpy as jnp

from .... import linfuncops
from ....typing import JaxFunction
from ._laplace import laplace_jax

if TYPE_CHECKING:
    import linpde_gp


def heat_jax(f: JaxFunction, argnum: int = 0) -> JaxFunction:
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
    laplace_f = laplace_jax(_f, argnum=argnum + 1)

    @jax.jit
    def _f_heat(*args, **kwargs) -> jnp.ndarray:
        tx = args[argnum]
        t, x = tx[0], tx[1:]
        args = args[:argnum] + (t, x) + args[argnum + 1 :]

        return df_dt(*args, **kwargs) - laplace_f(*args, **kwargs)

    return _f_heat


class HeatOperator(linfuncops.JaxLinearOperator):
    def __init__(self) -> None:
        super().__init__(L=heat_jax)

    @functools.singledispatchmethod
    def project(self, basis: linpde_gp.bases.Basis) -> pn.linops.LinearOperator:
        raise NotImplementedError()
